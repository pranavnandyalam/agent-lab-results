"""CLI. Modes:
  prepare                      filter RewardBench, split dev/resamples -> results/splits.json
  trial  --model M --n 20      timed trial on first n dev pairs (4 core passes each) -> results/trial_<M>.json
  run    --model M --resample s   resumable eval of one (model, resample) -> results/raw_<M>_r<s>.json
  prepare_noncode              build the 200-pair non-code set -> results/splits_noncode.json (see sjr.data.noncode_select)
  --pairs noncode              with trial/run: use splits_noncode.json; core passes only (better/W1 x cf/rf), identical
                               prompt and readout; outputs results/trial_nc_<M>.json, results/raw_nc_<M>_r<k>.json;
                               checkpoint ~/scratch/small-judge-reversal/run_nc_<M>_r<k>.jsonl (resumable).
                               The non-code set defines one draw, k=1 (--resample 1).
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

def load_splits(pairs="main"):
    with open(os.path.join(RES, "splits.json" if pairs == "main" else "splits_noncode.json")) as f: return json.load(f)

def prepare_noncode():
    from sjr.judge import load_tokenizer
    tok = load_tokenizer(C.FILTER_TOKENIZER)
    sp = load_splits()
    used = set(sp["dev"]) | {i for v in sp["resamples"].values() for i in v}
    full_len = D.make_full_len(tok, crits=("better", "W1"))
    rows = D.load_rb()
    ids, counts, meta = D.noncode_select(rows, full_len, used)
    rmap = {r["id"]: r for r in rows}
    ntok = lambda t: len(tok.encode(t, add_special_tokens=False))
    fl = [full_len(rmap[i]) for i in ids]
    subs = {}
    for i in ids: subs[rmap[i]["subset"]] = subs.get(rmap[i]["subset"], 0) + 1
    out = {"rb_revision": C.RB_REV, "filter_tokenizer": C.FILTER_TOKENIZER, "seed": D.NC_SEED, "n": D.NC_N,
           "rules": {"sections": list(D.NC_SECTIONS), "excluded": "safety subsets, hep-* (code), safety-keyword prompts, "
                     "ids in splits.json (dev + resamples 0-2)", "per_response_token_filter": None,
                     "max_full_prompt_tokens": D.NC_MAX_FULL, "full_prompt_tokens_over": "better,W1 x cf,rf (Qwen2.5 template)",
                     "substantively_different": f"whitespace-normalised responses differ and difflib char ratio < {D.NC_MAX_SIM}",
                     "selection": "proportional allocation over sections (largest remainder), seeded sample within section"},
           "counts": counts, **meta, "by_subset": subs,
           "describe": {"mean_full_tokens": statistics.mean(fl), "max_full_tokens": max(fl),
                        "mean_resp_tokens": statistics.mean(ntok(rmap[i][k]) for i in ids for k in ("chosen", "rejected")),
                        "share_pairs_resp_gt150": sum(max(ntok(rmap[i]["chosen"]), ntok(rmap[i]["rejected"])) > 150 for i in ids) / len(ids),
                        "mean_char_sim": statistics.mean(D.text_sim(rmap[i]["chosen"], rmap[i]["rejected"]) for i in ids)},
           "resamples": {"1": ids}}
    with open(os.path.join(RES, "splits_noncode.json"), "w") as f: json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "resamples"}, indent=1))

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

def trial(model, n, pairs="main"):
    from sjr.judge import Judge, set_threads
    import random
    set_threads(4)
    sp = load_splits(pairs)
    nc = "_nc" if pairs == "noncode" else ""
    # noncode: seeded sample (seed 0) of the set so the timing covers all sections; main: first n dev pairs
    ids = sp["dev"][:n] if pairs == "main" else sorted(random.Random(0).sample(sp["resamples"]["1"], n))
    os.makedirs(C.SCRATCH, exist_ok=True)
    ckpt = os.path.join(C.SCRATCH, f"trial{nc}_{model}.jsonl")
    if os.path.exists(ckpt): os.remove(ckpt)  # trial is a timing measurement: always fresh
    t_load = time.perf_counter(); judge = Judge(model); t_load = time.perf_counter() - t_load
    items = [(i, c, o) for i in ids for (c, o) in CORE_PASSES]
    t0 = time.perf_counter(); recs = run_items(judge, items, ckpt); wall = time.perf_counter() - t0
    flags = list(flags_from(recs).values())
    secs = [d["sec"] for d in recs]; toks = [d["n_tokens"] for d in recs]
    out = {"mode": "trial", "pairs": pairs, "model": model, "revision": C.MODELS[model][1], "dev_pair_ids": ids,
           "token_ids": judge.ids, "env": env_info(), "load_sec": t_load, "n_passes": len(recs),
           "wall_sec": wall, "sec_per_pass_mean": statistics.mean(secs), "sec_per_pass_wall": wall / len(recs),
           "mean_prompt_tokens": statistics.mean(toks), "max_prompt_tokens": max(toks),
           "mass_AB_mean": statistics.mean(d["mass_AB"] for d in recs),
           "mass_AB_min": min(d["mass_AB"] for d in recs),
           "metrics_W1_devtrial": M.summarize(flags), "records": recs}
    if pairs == "noncode": out["projection"] = project_nc(out, sp)
    with open(os.path.join(RES, f"trial{nc}_{model}.json"), "w") as f: json.dump(out, f, indent=1)
    return out

def project_nc(tr, sp):
    """Projected wall time for the full non-code set per model: trial sec/pass x (model's measured sec/token in the
    main raw runs / trial model's sec/token there) x mean trial prompt tokens ratio. Models without main raw runs: None."""
    def sec_per_tok(m):
        S = T = 0.0
        for k in (0, 1, 2):
            fp = os.path.join(RES, f"raw_{m}_r{k}.json")
            if not os.path.exists(fp): continue
            for r in json.load(open(fp))["records"]: S += r["sec"]; T += r["n_tokens"]
        return S / T if T else None
    base = sec_per_tok(tr["model"])
    n_pass = len(sp["resamples"]["1"]) * len(CORE_PASSES)
    out = {"n_passes_full_set": n_pass, "trial_sec_per_pass_wall": tr["sec_per_pass_wall"],
           "trial_mean_prompt_tokens": tr["mean_prompt_tokens"], "per_model_min": {}}
    for m in C.MODELS:
        spt = sec_per_tok(m)
        out["per_model_min"][m] = (round(tr["sec_per_pass_wall"] * spt / base * n_pass / 60, 1) if spt and base else None)
    out["method"] = ("trial wall sec/pass x n_passes, scaled by the ratio of forward sec/token measured in each model's "
                     "main raw runs (same machine, 4 threads); ignores model load time and attention's superlinear term")
    return out

def run(model, resample, pairs="main"):
    from sjr.judge import Judge, set_threads
    set_threads(4)
    sp = load_splits(pairs)
    ids = sp["resamples"][str(resample)]
    nc = "_nc" if pairs == "noncode" else ""
    passes = CORE_PASSES + (EXTRA_PASSES if (resample == 0 and pairs == "main") else [])
    items = [(i, c, o) for i in ids for (c, o) in passes]
    os.makedirs(C.SCRATCH, exist_ok=True)
    ckpt = os.path.join(C.SCRATCH, f"run{nc}_{model}_r{resample}.jsonl")
    judge = Judge(model)
    t0 = time.perf_counter()
    recs = run_items(judge, items, ckpt)
    out = {"mode": "run", "pairs": pairs, "model": model, "revision": C.MODELS[model][1], "resample": resample,
           "token_ids": judge.ids, "env": env_info(), "wall_sec_this_invocation": time.perf_counter() - t0,
           "records": recs}
    with open(os.path.join(RES, f"raw{nc}_{model}_r{resample}.json"), "w") as f: json.dump(out, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "prepare_noncode", "trial", "run"])
    ap.add_argument("--pairs", choices=["main", "noncode"], default="main")
    ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--resample", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "prepare": prepare()
    elif a.mode == "prepare_noncode": prepare_noncode()
    elif a.mode == "trial":
        o = trial(a.model, a.n, a.pairs)
        print(json.dumps({k: v for k, v in o.items() if k != "records"}, indent=1))
    else: run(a.model, a.resample, a.pairs)
