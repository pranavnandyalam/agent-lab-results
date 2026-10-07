"""Generic trainer over a token stream described by a JSON plan.

Plan JSON (paths relative to the project dir, or absolute): {"phases": [{"mode": "concat"|"interleave",
                        "sources": [{"bin": path, "offset": int, "tokens": int}, ...]}, ...]}
Each source slice is cut into non-overlapping (ctx+1)-token windows. Within a phase, windows are
shuffled (seeded): "concat" shuffles within each source and keeps sources in order; "interleave"
shuffles all windows of the phase together. Phases run in order. Default (--lr_schedule global): ONE
warmup(5%)+cosine LR schedule spans the whole run. --lr_schedule per_phase: the same warmup(5%)+cosine
schedule restarted independently at the start of each plan phase (phase length in steps =
phase windows / bs, boundaries at floor(cumulative windows / bs)). Checkpoints (resumable) go to --ckpt. Eval PPL on --eval splits at the end
(per-doc NLL sums saved for document-level bootstrap)."""
import argparse, hashlib, json, math, os, time
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig

EOD_ID = 0  # <|eod|> is the first special token in tok.json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project dir
# args that change the training trajectory; hashed (with plan + source bin contents) into last.pt
HASH_ARGS = ["seed", "lr", "bs", "wd", "d", "layers", "heads", "ctx", "vocab"]

def resolve(p):
    """Paths in plans/CLI may be relative to the project dir."""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def rel(p):
    """Store paths relative to the project dir when inside it."""
    ap_ = os.path.abspath(resolve(p))
    r = os.path.relpath(ap_, ROOT)
    return ap_ if r.startswith("..") else r

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def run_hash(plan, a):
    bins = sorted({s["bin"] for ph in plan["phases"] for s in ph["sources"]})
    hargs = {k: getattr(a, k) for k in HASH_ARGS}
    if getattr(a, "lr_schedule", "global") != "global":  # only non-default values enter the hash (old hashes unchanged)
        hargs["lr_schedule"] = a.lr_schedule
    blob = {"plan": plan, "args": hargs,
            "bin_sha256": {rel(b): file_sha256(resolve(b)) for b in bins}}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()

def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")

def build_windows(plan, ctx, seed):
    """Returns list of (bin_path, start) for every training window, in training order."""
    rng = np.random.default_rng(seed)
    order = []
    for ph in plan["phases"]:
        per_src = []
        for s in ph["sources"]:
            n = s["tokens"] // ctx  # each window = ctx inputs (+1 target from the next token)
            avail = len(load_bin(resolve(s["bin"]))) - s["offset"] - 1
            if n * ctx > avail:
                raise SystemExit(f"source {s} needs {n*ctx} tokens, only {avail} available")
            per_src.append([(s["bin"], s["offset"] + i * ctx) for i in range(n)])
        if ph["mode"] == "interleave":
            w = [x for src in per_src for x in src]
            order += [w[i] for i in rng.permutation(len(w))]
        elif ph["mode"] == "concat":
            for src in per_src:
                order += [src[i] for i in rng.permutation(len(src))]
        else:
            raise SystemExit(f"bad mode {ph['mode']}")
    return order

def lr_at(step, total, peak, warm_frac=0.05, min_ratio=0.1):
    warm = max(1, int(round(warm_frac * total)))
    if step < warm:
        return peak * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

def phase_bounds(plan, ctx, bs):
    """Step index where each phase starts, plus the total step count, e.g. [0, 512, 1024]."""
    cum, b = 0, [0]
    for ph in plan["phases"]:
        cum += sum(s["tokens"] // ctx for s in ph["sources"])
        b.append(cum // bs)
    return b

def lr_sched(step, total, peak, bounds=None):
    """global: one schedule over [0, total). per_phase (bounds given): lr_at restarted inside each phase."""
    if bounds is None:
        return lr_at(step, total, peak)
    for k in range(len(bounds) - 1):
        if bounds[k] <= step < bounds[k + 1]:
            return lr_at(step - bounds[k], bounds[k + 1] - bounds[k], peak)
    return lr_at(step, total, peak)

@torch.no_grad()
def evaluate(model, bin_path, ctx, bs=64, max_tokens=None):
    """Mean NLL over the whole split with non-overlapping ctx windows; returns ppl and per-doc sums."""
    model.eval()
    data = np.asarray(load_bin(resolve(bin_path)), dtype=np.int64)
    if max_tokens:
        data = data[: max_tokens + 1]
    doc_id = np.concatenate([[0], np.cumsum(data[:-1] == EOD_ID)])  # doc of each token
    n_win = (len(data) - 1) // ctx
    nll = np.zeros(n_win * ctx)
    for i in range(0, n_win, bs):
        j = min(n_win, i + bs)
        x = torch.from_numpy(np.stack([data[k*ctx:(k+1)*ctx] for k in range(i, j)]))
        y = torch.from_numpy(np.stack([data[k*ctx+1:(k+1)*ctx+1] for k in range(i, j)]))
        _, l = model(x, y, reduction="none")
        nll[i*ctx:j*ctx] = l.double().numpy()
    tgt_doc = doc_id[1:n_win*ctx+1]  # doc of each target token
    nd = int(tgt_doc.max()) + 1
    doc_sum = np.bincount(tgt_doc, weights=nll, minlength=nd)
    doc_cnt = np.bincount(tgt_doc, minlength=nd)
    model.train()
    return float(math.exp(nll.mean())), float(nll.mean()), doc_sum, doc_cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--ckpt", required=True, help="run dir for checkpoints/outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval", nargs="*", default=[], help="name=path.bin pairs")
    ap.add_argument("--eval_max_tokens", type=int, default=None)
    ap.add_argument("--lr_schedule", choices=["global", "per_phase"], default="global")
    a = ap.parse_args()
    a.ckpt = resolve(a.ckpt)
    os.makedirs(a.ckpt, exist_ok=True)
    plan = json.load(open(resolve(a.plan)))
    rhash = run_hash(plan, a)
    torch.manual_seed(a.seed)
    model = GPT(GPTConfig(vocab=a.vocab, ctx=a.ctx, d=a.d, n_layer=a.layers, n_head=a.heads))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    windows = build_windows(plan, a.ctx, a.seed)
    total = len(windows) // a.bs
    bounds = None
    if a.lr_schedule == "per_phase":
        bounds = phase_bounds(plan, a.ctx, a.bs)
        bounds[-1] = total
        print(f"per_phase LR schedule, phase step bounds {bounds}", flush=True)
    step, ck = 0, os.path.join(a.ckpt, "last.pt")
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=True)
        if st.get("run_hash") != rhash:
            raise SystemExit(f"refusing to resume {ck}: run_hash mismatch "
                             f"(ckpt {st.get('run_hash')} vs current {rhash}); plan/args/data changed")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = int(st["step"])
        print(f"resumed at step {step}/{total}", flush=True)
    print(f"non-emb params {model.n_params()}  total params {model.n_params(False)}  "
          f"steps {total}  tokens/step {a.bs*a.ctx}", flush=True)
    bins = {}
    log = open(os.path.join(a.ckpt, "train_log.jsonl"), "a")
    t0, tok_done = time.time(), 0
    model.train()
    while step < total:
        batch = windows[step * a.bs:(step + 1) * a.bs]
        xs, ys = [], []
        for b, s in batch:
            if b not in bins:
                bins[b] = load_bin(resolve(b))
            w = np.asarray(bins[b][s:s + a.ctx + 1], dtype=np.int64)
            xs.append(w[:-1]); ys.append(w[1:])
        x, y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
        lr = lr_sched(step, total, a.lr, bounds)
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1; tok_done += a.bs * a.ctx
        if step % a.log_every == 0 or step == total:
            el = time.time() - t0
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": lr, "tok_per_s": round(tok_done / el, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if step % a.ckpt_every == 0 or step == total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "run_hash": rhash}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    train_time = time.time() - t0
    res = {"steps": total, "train_tokens": total * a.bs * a.ctx, "train_seconds_this_session": round(train_time, 1),
           "tok_per_s_this_session": round(tok_done / train_time, 1) if tok_done else None, "seed": a.seed,
           "run_hash": rhash, "args": {**vars(a), "plan": rel(a.plan), "ckpt": rel(a.ckpt),
           "eval": [e.split("=", 1)[0] + "=" + rel(e.split("=", 1)[1]) for e in a.eval]}, "eval": {}}
    for e in a.eval:
        name, path = e.split("=", 1)
        te = time.time()
        ppl, nll, ds, dc = evaluate(model, path, a.ctx, max_tokens=a.eval_max_tokens)
        np.savez(os.path.join(a.ckpt, f"eval_{name}_perdoc.npz"), nll_sum=ds, count=dc)
        res["eval"][name] = {"ppl": ppl, "nll": nll, "seconds": round(time.time() - te, 1)}
    print(json.dumps(res["eval"]), flush=True)
    if tok_done == 0 and not a.eval and os.path.exists(os.path.join(a.ckpt, "result.json")):
        print("nothing trained or evaluated this session; keeping existing result.json", flush=True)
        return
    json.dump(res, open(os.path.join(a.ckpt, "result.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
