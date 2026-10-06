"""CLI. Modes:
  prepare                      filter RewardBench, split dev/resamples -> results/splits.json
  trial  --model M --n 20      timed trial on first n dev pairs (4 core passes each) -> results/trial_<M>.json
  run    --model M --resample s   resumable eval of one (model, resample) -> results/raw_<M>_r<s>.json
"""
import argparse, json, os, sys, time, platform, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sjr import config as C
from sjr import data as D
from sjr import metrics as M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
CORE_PASSES = [("better", "cf"), ("better", "rf"), ("W1", "cf"), ("W1", "rf")]
EXTRA_PASSES = [("W2", "cf"), ("W2", "rf"), ("length", "cf"), ("length", "rf")]

def by_id():
    return {r["id"]: r for r in D.load_rb()}

def prepare():
    from sjr.judge import load_tokenizer
    tok = load_tokenizer(C.FILTER_TOKENIZER)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))
    rows = D.load_rb()
    kept, counts = D.filter_pairs(rows, ntok, D.make_full_len(tok))
    dev, res = D.split_pool(kept)
    kmap = {r["id"]: r for r in kept}
    def desc(ids):
        secs = {}
        for i in ids: secs[kmap[i]["section"]] = secs.get(kmap[i]["section"], 0) + 1
        return {"n": len(ids), "by_section": secs,
                "mean_max_full_tokens": statistics.mean(kmap[i]["full_tokens"] for i in ids)}
    pool_secs = {}
    for r in kept: pool_secs[r["section"]] = pool_secs.get(r["section"], 0) + 1
    out = {"rb_revision": C.RB_REV, "filter_tokenizer": C.FILTER_TOKENIZER, "counts": counts,
           "pool_by_section": pool_secs, "dev_seed": C.DEV_SEED, "dev": dev,
           "resamples": {str(s): v for s, v in res.items()},
           "describe": {"dev": desc(dev), **{f"r{s}": desc(v) for s, v in res.items()}},
           "mean_max_full_tokens_eval300": statistics.mean(kmap[i]["full_tokens"] for v in res.values() for i in v)}
    with open(os.path.join(RES, "splits.json"), "w") as f: json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ["counts", "pool_by_section", "describe", "mean_max_full_tokens_eval300"]}, indent=1))

def load_splits():
    with open(os.path.join(RES, "splits.json")) as f: return json.load(f)

def run_items(judge, items, ckpt):
    """items: list of (pair_id, criterion, order). Resumable via jsonl checkpoint."""
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                try: d = json.loads(line)
                except json.JSONDecodeError: continue
                done[(d["id"], d["criterion"], d["order"])] = d
    rb = by_id()
    with open(ckpt, "a") as f:
        for (pid, crit, order) in items:
            if (pid, crit, order) in done: continue
            r = rb[pid]
            a, b = (r["chosen"], r["rejected"]) if order == "cf" else (r["rejected"], r["chosen"])
            out = judge.judge(r["prompt"], a, b, crit)
            d = {"id": pid, "criterion": crit, "order": order, "subset": r["subset"], **out}
            f.write(json.dumps(d) + "\n"); f.flush()
            done[(pid, crit, order)] = d
    return [done[k] for k in items]

def flags_from(recs, worse="W1"):
    per = {}
    for d in recs: per.setdefault(d["id"], {})[(d["criterion"], d["order"])] = d["letter"]
    return {pid: M.pair_flags(j, worse) for pid, j in per.items()
            if all(k in j for k in [("better", "cf"), ("better", "rf"), (worse, "cf"), (worse, "rf")])}

def env_info():
    import torch, transformers
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "threads": torch.get_num_threads(), "dtype": "float32", "machine": platform.machine()}

def trial(model, n):
    from sjr.judge import Judge, set_threads
    set_threads(4)
    sp = load_splits()
    ids = sp["dev"][:n]
    os.makedirs(C.SCRATCH, exist_ok=True)
    ckpt = os.path.join(C.SCRATCH, f"trial_{model}.jsonl")
    if os.path.exists(ckpt): os.remove(ckpt)  # trial is a timing measurement: always fresh
    t_load = time.perf_counter(); judge = Judge(model); t_load = time.perf_counter() - t_load
    items = [(i, c, o) for i in ids for (c, o) in CORE_PASSES]
    t0 = time.perf_counter(); recs = run_items(judge, items, ckpt); wall = time.perf_counter() - t0
    flags = list(flags_from(recs).values())
    secs = [d["sec"] for d in recs]; toks = [d["n_tokens"] for d in recs]
    out = {"mode": "trial", "model": model, "revision": C.MODELS[model][1], "dev_pair_ids": ids,
           "token_ids": judge.ids, "env": env_info(), "load_sec": t_load, "n_passes": len(recs),
           "wall_sec": wall, "sec_per_pass_mean": statistics.mean(secs), "sec_per_pass_wall": wall / len(recs),
           "mean_prompt_tokens": statistics.mean(toks), "max_prompt_tokens": max(toks),
           "mass_AB_mean": statistics.mean(d["mass_AB"] for d in recs),
           "mass_AB_min": min(d["mass_AB"] for d in recs),
           "metrics_W1_devtrial": M.summarize(flags), "records": recs}
    with open(os.path.join(RES, f"trial_{model}.json"), "w") as f: json.dump(out, f, indent=1)
    return out

def run(model, resample):
    from sjr.judge import Judge, set_threads
    set_threads(4)
    sp = load_splits()
    ids = sp["resamples"][str(resample)]
    passes = CORE_PASSES + (EXTRA_PASSES if resample == 0 else [])
    items = [(i, c, o) for i in ids for (c, o) in passes]
    os.makedirs(C.SCRATCH, exist_ok=True)
    ckpt = os.path.join(C.SCRATCH, f"run_{model}_r{resample}.jsonl")
    judge = Judge(model)
    recs = run_items(judge, items, ckpt)
    out = {"mode": "run", "model": model, "revision": C.MODELS[model][1], "resample": resample,
           "token_ids": judge.ids, "env": env_info(), "records": recs}
    with open(os.path.join(RES, f"raw_{model}_r{resample}.json"), "w") as f: json.dump(out, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "trial", "run"])
    ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--resample", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "prepare": prepare()
    elif a.mode == "trial":
        o = trial(a.model, a.n)
        print(json.dumps({k: v for k, v in o.items() if k != "records"}, indent=1))
    else: run(a.model, a.resample)
