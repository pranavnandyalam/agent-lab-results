"""Document-wise ancestral sampling (temperature T, default 1.0, no top-k/top-p) from a trained GPT checkpoint,
with a batched per-row KV cache.

Data format (see src/tok.py): every document is its tokens followed by <|eod|> (id 0); no BOS. Here each
document is generated from the context [<|eod|>] (the token that precedes every doc in the training stream
except the very first) until the model emits <|eod|> (or --max_doc tokens, then <|eod|> is appended and the doc
counted as truncated). Output .bin = uint16 stream of doc tokens + <|eod|>, identical format to data/R_*.bin.

Batching: B rows, each row is one in-progress document with its own position (cache index == position id).
When a row's context reaches ctx tokens, the row is re-prefilled with its last ctx//2 - 1 context tokens
(sliding window; positions restart at 0, as in training windows that start mid-document).
Length-bias control: once (finished + in-flight) tokens reach --tokens, no new documents are started, all
in-flight docs are finished, and docs are written in START order (every started doc is kept), so the corpus
is an i.i.d. sample of documents, not biased toward short docs.

Usage:
  python -I src/sample.py --selftest --ckpt ckpt/G0_8M_s0
  python -I src/sample.py --ckpt ckpt/G0_8M_s0 --out data/S1.bin --tokens 4194304 --seed 0
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig

EOD_ID = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def load_model(ckpt_dir):
    ckpt_dir = resolve(ckpt_dir)
    args = json.load(open(os.path.join(ckpt_dir, "result.json")))["args"]
    c = GPTConfig(vocab=args["vocab"], ctx=args["ctx"], d=args["d"], n_layer=args["layers"], n_head=args["heads"])
    m = GPT(c)
    st = torch.load(os.path.join(ckpt_dir, "last.pt"), weights_only=True)
    m.load_state_dict(st["model"])
    m.eval()
    return m, st

class KVCache:
    def __init__(self, m, B):
        c = m.c
        self.k = [torch.zeros(B, c.n_head, c.ctx, c.d // c.n_head) for _ in range(c.n_layer)]
        self.v = [torch.zeros(B, c.n_head, c.ctx, c.d // c.n_head) for _ in range(c.n_layer)]

@torch.inference_mode()
def prefill(m, cache, rows, seqs):
    """seqs: LongTensor [R,T] (T<=ctx) written at cache positions 0..T-1 of `rows`. Returns logits [R,T,V]."""
    R, T = seqs.shape
    x = m.tok(seqs) + m.pos(torch.arange(T))
    for li, b in enumerate(m.blocks):
        C = x.shape[-1]
        q, k, v = b.qkv(b.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(R, T, b.h, C // b.h).transpose(1, 2) for t in (q, k, v))
        cache.k[li][rows, :, :T] = k
        cache.v[li][rows, :, :T] = v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + b.proj(y.transpose(1, 2).reshape(R, T, C))
        x = x + b.fc2(F.gelu(b.fc(b.ln2(x))))
    return m.head(m.ln_f(x))

@torch.inference_mode()
def step(m, cache, tok, pos):
    """One token per row. tok, pos: LongTensor [B]; row r's token is written at cache index pos[r] and
    attends to cache indices 0..pos[r]. Returns logits [B,V]."""
    B = tok.shape[0]
    ctx = m.c.ctx
    ar = torch.arange(B)
    x = (m.tok(tok) + m.pos(pos)).unsqueeze(1)  # [B,1,C]
    mask = (torch.arange(ctx).unsqueeze(0) <= pos.unsqueeze(1)).view(B, 1, 1, ctx)
    for li, b in enumerate(m.blocks):
        C = x.shape[-1]
        q, k, v = b.qkv(b.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, 1, b.h, C // b.h).transpose(1, 2) for t in (q, k, v))  # [B,H,1,dh]
        cache.k[li][ar, :, pos] = k[:, :, 0]
        cache.v[li][ar, :, pos] = v[:, :, 0]
        y = F.scaled_dot_product_attention(q, cache.k[li], cache.v[li], attn_mask=mask)
        x = x + b.proj(y.transpose(1, 2).reshape(B, 1, C))
        x = x + b.fc2(F.gelu(b.fc(b.ln2(x))))
    return m.head(m.ln_f(x))[:, 0]

@torch.inference_mode()
def selftest(m, atol=1e-4):
    data = np.fromfile(resolve("data/R_val.bin"), dtype=np.uint16).astype(np.int64)
    ctx = m.c.ctx
    res = {}
    # (a) incremental from empty cache vs full no-cache forward, B=2 rows, 64 tokens
    T = 64
    seqs = torch.from_numpy(np.stack([data[0:T], data[1000:1000 + T]]))
    full = m(seqs)
    cache = KVCache(m, 2)
    inc = torch.stack([step(m, cache, seqs[:, t], torch.full((2,), t)) for t in range(T)], 1)
    res["a_incremental_vs_full_maxabs"] = float((inc - full).abs().max())
    # (b) prefill 100 tokens then 20 incremental steps vs full forward of 120 tokens
    s = torch.from_numpy(data[5000:5120]).unsqueeze(0)
    full = m(s)
    cache = KVCache(m, 1)
    lp = prefill(m, cache, torch.tensor([0]), s[:, :100])
    inc = torch.stack([step(m, cache, s[:, t], torch.tensor([t])) for t in range(100, 120)], 1)
    res["b_prefill_vs_full_maxabs"] = float(max((lp - full[:, :100]).abs().max(), (inc - full[:, 100:]).abs().max()))
    # (c) rows at different positions: row1 restarts at step 30 with a new sequence while row0 continues
    s0 = torch.from_numpy(data[7000:7080]); s1a = torch.from_numpy(data[9000:9030]); s1b = torch.from_numpy(data[11000:11050])
    full0, full1b = m(s0[None])[0], m(s1b[None])[0]
    cache = KVCache(m, 2)
    errs = []
    for t in range(80):
        if t < 30:
            tok, pos = torch.stack([s0[t], s1a[t]]), torch.tensor([t, t])
        else:
            tok, pos = torch.stack([s0[t], s1b[t - 30]]), torch.tensor([t, t - 30])
        lg = step(m, cache, tok, pos)
        errs.append(float((lg[0] - full0[t]).abs().max()))
        if t >= 30:
            errs.append(float((lg[1] - full1b[t - 30]).abs().max()))
    res["c_mixed_positions_maxabs"] = max(errs)
    # (d) full-length context (ctx tokens) incremental vs full
    s = torch.from_numpy(data[20000:20000 + ctx]).unsqueeze(0)
    full = m(s)
    cache = KVCache(m, 1)
    inc = torch.stack([step(m, cache, s[:, t], torch.tensor([t])) for t in range(ctx)], 1)
    res["d_fullctx_maxabs"] = float((inc - full).abs().max())
    res["atol"] = atol
    res["pass"] = all(v <= atol for k, v in res.items() if k.endswith("maxabs"))
    return res

@torch.inference_mode()
def sample(m, n_tokens, B, seed, temperature=1.0, max_doc=2048, log_every=30.0):
    """Returns (list of np.uint16 docs in start order, each ending with EOD; stats)."""
    ctx = m.c.ctx
    keep = ctx // 2 - 1  # context tokens kept on re-prefill
    g = torch.Generator().manual_seed(seed)
    cache = KVCache(m, B)
    bufs = [[EOD_ID] for _ in range(B)]  # context: leading EOD + doc tokens so far
    row_doc = list(range(B))              # start index of the doc in each row
    next_start = B
    active = np.ones(B, bool)
    done = {}                             # start index -> np array (doc + EOD)
    tok = torch.full((B,), EOD_ID, dtype=torch.long)
    pos = torch.zeros(B, dtype=torch.long)
    finished_tok, n_trunc, n_refill, n_steps = 0, 0, 0, 0
    starting = True
    t0 = tlast = time.time()
    while active.any():
        # rows whose cache is full: sliding re-prefill (all such rows have identical lengths)
        full = [r for r in range(B) if active[r] and int(pos[r]) == ctx]
        if full:
            seqs = torch.tensor([bufs[r][-keep - 1:-1] for r in full], dtype=torch.long)
            prefill(m, cache, torch.tensor(full), seqs)
            pos[full] = keep
            n_refill += len(full)
        logits = step(m, cache, tok, pos).float()
        n_steps += 1
        probs = torch.softmax(logits / temperature, -1)
        nxt = torch.multinomial(probs, 1, generator=g)[:, 0]
        nl = nxt.tolist()
        inflight = 0
        for r in range(B):
            if not active[r]:
                continue
            t = nl[r]
            bufs[r].append(t)
            ndoc = len(bufs[r]) - 1  # doc tokens incl. this one
            if t == EOD_ID or ndoc >= max_doc:
                if t != EOD_ID:
                    bufs[r].append(EOD_ID); n_trunc += 1
                d = np.asarray(bufs[r][1:], dtype=np.uint16)
                done[row_doc[r]] = d
                finished_tok += d.size
                if starting:
                    bufs[r] = [EOD_ID]; row_doc[r] = next_start; next_start += 1
                    tok[r] = EOD_ID; pos[r] = 0
                else:
                    active[r] = False
            else:
                tok[r] = t; pos[r] += 1
                inflight += ndoc
        if starting and finished_tok + inflight >= n_tokens:
            starting = False
        if time.time() - tlast > log_every:
            tlast = time.time()
            print(json.dumps({"finished_tokens": finished_tok, "docs": len(done), "steps": n_steps,
                              "tok_per_s": round(finished_tok / (tlast - t0), 1)}), flush=True)
    secs = time.time() - t0
    docs = [done[i] for i in sorted(done)]
    assert sorted(done) == list(range(len(done)))
    tot = int(sum(d.size for d in docs))
    return docs, {"tokens": tot, "docs": len(docs), "truncated_docs": n_trunc, "refills": n_refill,
                  "steps": n_steps, "batch": B, "seconds": round(secs, 1), "tok_per_s": round(tot / secs, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--tokens", type=int, default=4194304)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_doc", type=int, default=2048)
    ap.add_argument("--n_text", type=int, default=5, help="docs decoded to <out>.samples.txt")
    ap.add_argument("--meta", help="json path for run metadata")
    a = ap.parse_args()
    m, st = load_model(a.ckpt)
    if a.selftest:
        r = selftest(m)
        print(json.dumps(r))
        if a.meta:
            json.dump(r, open(resolve(a.meta), "w"), indent=1)
        sys.exit(0 if r["pass"] else 1)
    torch.manual_seed(a.seed)
    docs, stats = sample(m, a.tokens, a.batch, a.seed, a.temperature, a.max_doc)
    arr = np.concatenate(docs)
    out = resolve(a.out)
    arr.tofile(out)
    lens = np.array([d.size for d in docs])
    stats.update({"ckpt": a.ckpt, "ckpt_run_hash": st.get("run_hash"), "seed": a.seed,
                  "temperature": a.temperature, "max_doc": a.max_doc, "target_tokens": a.tokens, "out": a.out,
                  "out_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                  "doc_len_mean": float(lens.mean()), "doc_len_pct50_90_99": [float(x) for x in np.percentile(lens, [50, 90, 99])],
                  "doc_len_max": int(lens.max()), "torch": torch.__version__, "threads": torch.get_num_threads()})
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(resolve("data/tok.json"))
    with open(out + ".samples.txt", "w", encoding="utf-8") as f:
        for i in range(min(a.n_text, len(docs))):
            f.write(f"=== doc {i} ({docs[i].size} tokens) ===\n{tk.decode(docs[i][:-1].tolist())}\n\n")
    print(json.dumps(stats), flush=True)
    if a.meta:
        json.dump(stats, open(resolve(a.meta), "w"), indent=1)

if __name__ == "__main__":
    main()
