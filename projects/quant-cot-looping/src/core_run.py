"""Core run: Qwen3-0.6B on 16 GSM8K TEST problems (permutation seed 0 of the test split at DS_REV, first 16),
split into 2 chunks of 8 problems; one batch = one chunk x seeds 0 1 2 = 24 generations.

Decode loop identical to src/timed_trial.py (temperature 0.6 -> top-k 20 -> top-p 0.95, multinomial with a
per-row torch.Generator seeded with the sample seed, left padding, explicit position ids, DynamicCache,
compaction of finished rows). Output: results/core/<tag>_chunk<k>.json; skipped if it already exists (resumable).

Levels (tag: --bits --group): fp32: 16 (no quant) | w5g64: 5 64 | w4g64: 4 64 | w3g32: 3 32
Usage:
  python -I src/core_run.py --bits 4 --group 64 --tag w4g64 --problems-chunk 0
Smoke only: --limit-problems 1 --seeds 0 --max-new 32 --out-dir ~/scratch/x
"""
import argparse, json, os, sys, time, re
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from transformers import DynamicCache, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from trial import load, build, env, DS_REV, HERE  # sets HF offline, threads=4
from quant import fake_quantize_model

torch.set_num_threads(4)
N_TEST, CHUNK = 16, 8


def load_test_chunk(k, seed=0):
    from huggingface_hub import snapshot_download
    p = snapshot_download("openai/gsm8k", repo_type="dataset", revision=DS_REV, allow_patterns=["main/*.parquet"])
    df = pd.read_parquet(os.path.join(p, "main", "test-00000-of-00001.parquet"))
    idx16 = [int(i) for i in np.random.default_rng(seed).permutation(len(df))[:N_TEST]]
    idx = idx16[k * CHUNK:(k + 1) * CHUNK]
    return idx16, idx, [df.iloc[i]["question"] for i in idx], [df.iloc[i]["answer"] for i in idx]


@torch.no_grad()
def run(a):
    out_dir = os.path.expanduser(a.out_dir)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(HERE, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{a.tag}_chunk{a.problems_chunk}.json")
    if os.path.exists(out):
        print(f"SKIP exists: {out}"); return
    tok, model = load("fp32")
    nq = fake_quantize_model(model, a.bits, a.group) if a.bits < 16 else 0
    idx16, pidx, qs, golds = load_test_chunk(a.problems_chunk, seed=0)
    if a.limit_problems:                                                  # smoke test only
        pidx, qs, golds = pidx[:a.limit_problems], qs[:a.limit_problems], golds[:a.limit_problems]
    rows = [(pi, s) for pi in range(len(qs)) for s in a.seeds]          # problem-major
    B = len(rows)
    enc = build(tok, [qs[pi] for pi, _ in rows])
    gens = [torch.Generator().manual_seed(s) for _, s in rows]
    warpers = [TemperatureLogitsWarper(0.6), TopKLogitsWarper(20), TopPLogitsWarper(0.95)]
    eos_list = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")})
    pad = tok.pad_token_id
    ids, am = enc["input_ids"], enc["attention_mask"]
    prompt_ids = [ids[r][am[r].bool()].tolist() for r in range(B)]
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    cache = DynamicCache()
    t0 = time.perf_counter()
    logits = model(input_ids=ids, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True).logits[:, -1].float()
    t_prefill = time.perf_counter() - t0
    nxt = pos[:, -1] + 1
    active = list(range(B))
    finished = [False] * B
    out_toks = [[] for _ in range(B)]
    slot_steps, steps, trace = 0, 0, []
    for step in range(a.max_new):
        sc = logits
        for w in warpers:
            sc = w(None, sc)
        probs = torch.softmax(sc, -1)
        t = torch.stack([torch.multinomial(probs[k], 1, generator=gens[r]) for k, r in enumerate(active)]).squeeze(1)
        slot_steps += len(active); steps += 1
        done = torch.zeros(len(active), dtype=torch.bool)
        for k, r in enumerate(active):
            if finished[r]:
                t[k] = pad; done[k] = True
            else:
                out_toks[r].append(int(t[k]))
                if int(t[k]) in eos_list:
                    finished[r] = True; done[k] = True
        el = time.perf_counter() - t0
        if step % 64 == 0 or done.all():
            trace.append({"step": step, "active_rows": len(active), "elapsed_s": round(el, 2)})
        if all(finished[r] for r in active) or step == a.max_new - 1:
            break
        if done.any():
            keep = torch.nonzero(~done).squeeze(1)
            cache.batch_select_indices(keep)
            am, nxt, t = am[keep], nxt[keep], t[keep]
            active = [active[int(k)] for k in keep]
        am = torch.cat([am, torch.ones(len(active), 1, dtype=am.dtype)], 1)
        logits = model(input_ids=t[:, None], attention_mask=am, position_ids=nxt[:, None],
                       past_key_values=cache, use_cache=True).logits[:, -1].float()
        nxt = nxt + 1
    t_total = time.perf_counter() - t0
    lens = [len(o) for o in out_toks]
    trunc = [not finished[r] for r in range(B)]
    texts = [tok.decode(o, skip_special_tokens=False) for o in out_toks]
    boxed = [(re.findall(r"\\boxed\{([^{}]*)\}", tx) or [None])[-1] for tx in texts]
    gold_num = [g.split("####")[-1].strip().replace(",", "") for g in golds]
    rec = {"tag": a.tag, "bits": a.bits, "group": a.group if a.bits < 16 else None, "dtype": "fp32",
           "problems_chunk": a.problems_chunk, "gsm8k_split": "test", "gsm8k_test_idx_all16": idx16,
           "gsm8k_test_idx": pidx, "batch": B, "seeds": a.seeds, "compact": True, "max_new_tokens": a.max_new,
           "n_linear_quantized": nq, "eos_ids": eos_list,
           "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "per_row_generator_seed": "sample seed"},
           "prompt_len_padded": int(ids.shape[1]), "decode_steps_run": steps, "slot_steps": slot_steps,
           "t_prefill_s": t_prefill, "t_total_s": t_total, "t_decode_s": t_total - t_prefill,
           "gen_tokens": lens, "truncated": trunc, "n_truncated": int(sum(trunc)),
           "mean_gen_tokens": float(np.mean(lens)), "median_gen_tokens": float(np.median(lens)),
           "max_gen_tokens": int(max(lens)), "useful_tokens": int(sum(lens)),
           "agg_useful_tok_s_total": sum(lens) / t_total, "trace": trace,
           "rows": [{"problem": pidx[pi], "seed": s, "gen_tokens": lens[r], "truncated": trunc[r], "boxed": boxed[r],
                     "gold": gold_num[pi], "text": texts[r], "prompt_ids": prompt_ids[r], "token_ids": out_toks[r]}
                    for r, (pi, s) in enumerate(rows)],
           "env": env(), "argv": sys.argv}
    json.dump(rec, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    print(json.dumps({k: rec[k] for k in ("tag", "problems_chunk", "batch", "t_total_s", "n_truncated",
                                          "mean_gen_tokens", "useful_tokens", "gen_tokens")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True, help="16 = fp32, no quantization")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--problems-chunk", type=int, choices=[0, 1], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--limit-problems", type=int, default=0, help="smoke test only")
    ap.add_argument("--out-dir", default="results/core")
    run(ap.parse_args())
