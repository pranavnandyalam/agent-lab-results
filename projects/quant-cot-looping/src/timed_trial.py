"""Full-length timed sampling trial (compute planning) for Qwen3-0.6B on CPU.

Custom batched decode loop (same sampling as HF generate: temperature -> top-k -> top-p warpers, multinomial),
left padding, explicit position ids, DynamicCache. Each row has its own torch.Generator seeded with its sample
seed, so a (problem, seed) row is reproducible regardless of batch composition (up to fp batch nondeterminism).
--compact (default on) drops finished rows from the batch and KV cache (cache.batch_select_indices), so finished
sequences stop consuming compute; --no-compact mimics HF generate (finished rows keep decoding pad tokens).

Usage:
  python -I src/timed_trial.py --bits 4 --n-problems 8 --seeds 0 1 2 --max-new 2048 --out results/timed_trial_w4.json
  python -I src/timed_trial.py --bits 16 --n-problems 8 --seeds 0 --max-new 2048 --out results/timed_trial_fp32.json
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


def load_train(n, seed=0):
    from huggingface_hub import snapshot_download
    p = snapshot_download("openai/gsm8k", repo_type="dataset", revision=DS_REV, allow_patterns=["main/*.parquet"])
    df = pd.read_parquet(os.path.join(p, "main", "train-00000-of-00001.parquet"))
    idx = [int(i) for i in np.random.default_rng(seed).permutation(len(df))[:n]]
    return idx, [df.iloc[i]["question"] for i in idx], [df.iloc[i]["answer"] for i in idx]


@torch.no_grad()
def run(a):
    tok, model = load("fp32")
    nq = fake_quantize_model(model, a.bits, a.group) if a.bits < 16 else 0
    pidx, qs, golds = load_train(a.n_problems, seed=0)
    rows = [(pi, s) for pi in range(len(qs)) for s in a.seeds]          # problem-major
    B = len(rows)
    enc = build(tok, [qs[pi] for pi, _ in rows])
    gens = [torch.Generator().manual_seed(s) for _, s in rows]
    warpers = [TemperatureLogitsWarper(0.6), TopKLogitsWarper(20), TopPLogitsWarper(0.95)]
    eos_set = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} | set(a.test_extra_eos)  # extra: testing only
    eos = torch.tensor(sorted(eos_set))
    pad = tok.pad_token_id
    ids, am = enc["input_ids"], enc["attention_mask"]
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    cache = DynamicCache()
    t0 = time.perf_counter()
    logits = model(input_ids=ids, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True).logits[:, -1].float()
    t_prefill = time.perf_counter() - t0
    nxt = pos[:, -1] + 1
    active = list(range(B))            # original row index for each live batch row
    finished = [False] * B
    out_toks = [[] for _ in range(B)]
    slot_steps, steps, trace, stopped = 0, 0, [], False
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
                if int(t[k]) in eos.tolist():
                    finished[r] = True; done[k] = True
        el = time.perf_counter() - t0
        if step % 64 == 0 or done.all():
            trace.append({"step": step, "active_rows": len(active), "elapsed_s": round(el, 2)})
        if all(finished[r] for r in active) or step == a.max_new - 1:
            break
        if el > a.deadline:
            stopped = True; break
        if a.compact and done.any():
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
    rec = {"bits": a.bits, "group": a.group, "dtype": "fp32", "batch": B, "seeds": a.seeds, "compact": a.compact,
           "max_new_tokens": a.max_new, "n_linear_quantized": nq, "gsm8k_train_idx": pidx,
           "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "per_row_generator_seed": "sample seed"},
           "prompt_len_padded": int(ids.shape[1]), "decode_steps_run": steps, "stopped_by_deadline": stopped,
           "slot_steps": slot_steps, "slot_steps_if_no_compaction": B * steps,
           "t_prefill_s": t_prefill, "t_total_s": t_total, "t_decode_s": t_total - t_prefill,
           "gen_tokens": lens, "truncated": trunc, "n_truncated": int(sum(trunc)),
           "mean_gen_tokens": float(np.mean(lens)), "median_gen_tokens": float(np.median(lens)), "max_gen_tokens": int(max(lens)),
           "useful_tokens": int(sum(lens)), "agg_useful_tok_s_total": sum(lens) / t_total,
           "agg_useful_tok_s_decode": sum(lens) / (t_total - t_prefill),
           "slot_tok_s_decode": slot_steps / (t_total - t_prefill), "trace": trace,
           "rows": [{"problem": pidx[pi], "seed": s, "gen_tokens": lens[r], "truncated": trunc[r], "boxed": boxed[r],
                     "gold": gold_num[pi], "text": texts[r]} for r, (pi, s) in enumerate(rows)],
           "env": env(), "argv": sys.argv}
    out = os.path.join(HERE, a.out) if not os.path.isabs(a.out) else a.out
    json.dump(rec, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("rows", "env", "trace", "gsm8k_train_idx", "truncated")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--n-problems", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--deadline", type=float, default=1e9, help="stop decoding after this many seconds (partial save)")
    ap.add_argument("--compact", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--test-extra-eos", type=int, nargs="*", default=[], help="testing only: extra token ids treated as EOS")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
