"""Generation-based validation of the logit readout (referee fix 1).
Greedy decode (<=16 new tokens) with the identical chat prompt (Qwen3: enable_thinking=False) on the first N pairs
of resample 0, criteria better/W1, both orders. Parse first standalone 'A'/'B'.
  gen   --model M [--n 100]   -> results/gen_check_<M>.json (checkpoint every 10 pairs, resumable)
  report                      -> results/gen_check.md (+ gen_check_summary.json)
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sjr import config as C
from sjr import metrics as M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
PASSES = [("better", "cf"), ("better", "rf"), ("W1", "cf"), ("W1", "rf")]
MAX_NEW = 16
VERDICT_RE = re.compile(r"(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])")

def parse(text):
    """First standalone capital A or B (covers 'A', 'A.', '**B**', 'Response A'). None if absent."""
    m = VERDICT_RE.search(text)
    return m.group(1) if m else None

def out_path(model): return os.path.join(RES, f"gen_check_{model}.json")

def gen(model, n):
    import torch
    from sjr.judge import Judge, set_threads
    from sjr.prompts import chat_input
    from run import by_id, load_splits, env_info
    set_threads(4)
    torch.manual_seed(0)
    ids = load_splits()["resamples"]["0"][:n]
    path = out_path(model)
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for d in json.load(f)["records"]: done[(d["id"], d["criterion"], d["order"])] = d
    judge = Judge(model)
    tok, mdl = judge.tok, judge.model
    rb = by_id()
    meta = {"mode": "gen_check", "model": model, "revision": C.MODELS[model][1], "resample": 0, "n_pairs": n,
            "max_new_tokens": MAX_NEW, "decoding": "greedy (do_sample=False)", "seed": 0,
            "enable_thinking": False if judge.is_qwen3 else None, "env": env_info()}
    def save():
        recs = [done[(i, c, o)] for i in ids for (c, o) in PASSES if (i, c, o) in done]
        tmp = path + ".tmp"
        with open(tmp, "w") as f: json.dump({**meta, "records": recs}, f, indent=0)
        os.replace(tmp, path)
    for k, pid in enumerate(ids):
        r = rb[pid]
        for (crit, order) in PASSES:
            if (pid, crit, order) in done: continue
            a, b = (r["chosen"], r["rejected"]) if order == "cf" else (r["rejected"], r["chosen"])
            text = chat_input(tok, r["prompt"], a, b, crit, judge.is_qwen3)
            enc = tok(text, return_tensors="pt", add_special_tokens=False)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = mdl.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None,
                                   top_p=None, top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            dt = time.perf_counter() - t0
            new = out[0, enc["input_ids"].shape[1]:].tolist()
            g = tok.decode(new, skip_special_tokens=True)
            done[(pid, crit, order)] = {"id": pid, "criterion": crit, "order": order, "subset": r["subset"],
                                        "generation": g, "generation_raw": tok.decode(new, skip_special_tokens=False),
                                        "gen_token_ids": new, "verdict": parse(g), "sec": dt}
        if (k + 1) % 10 == 0 or k + 1 == len(ids):
            save(); print(f"[{model}] {k+1}/{len(ids)} pairs", flush=True)
    save()

def flags_of(per):
    return [M.pair_flags(j, "W1") for j in per.values() if all(j.get(p) in ("A", "B") for p in PASSES)]

def summ_block(per):
    fl = flags_of(per)
    s = M.summarize(fl)
    lo, hi, _ = M.bootstrap(fl, "acc_better_poscons") if fl else (float("nan"),) * 3
    s["acc_ci"] = [lo, hi]
    return s

def report():
    lines = ["# Generation-based validation of the logit readout (referee fix 1)", "",
             f"Greedy decoding, max_new_tokens={MAX_NEW}, identical chat prompt (Qwen3: enable_thinking=False), "
             "resample 0, criteria `better` and `W1` (worse), both orders (4 judgments/pair). Verdict = first "
             "standalone `A`/`B` in the decoded text (regex `(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])`). "
             "Logit readout = argmax over tokens `A` vs `B` from raw_<model>_r0.json. Taxonomy/metrics via "
             "src/sjr/metrics.py on pairs where all 4 generated verdicts parsed; logit rows use the same pairs. "
             "acc = chosen preferred under `better` in both orders; reversal = acc AND rejected named `worse` in both "
             "orders (correct_reversal share); position-locked = same letter in all 4 judgments. CI: bootstrap over "
             "pairs (2000, seed 0).", ""]
    summary = {}
    for model in ["Qwen3-1.7B", "Qwen2.5-1.5B-Instruct"]:
        p = out_path(model)
        if not os.path.exists(p):
            lines += [f"## {model}", "", "not run", ""]; continue
        g = json.load(open(p))
        raw = json.load(open(os.path.join(RES, f"raw_{model}_r0.json")))
        logit = {(d["id"], d["criterion"], d["order"]): d for d in raw["records"]}
        recs = g["records"]
        n_j = len(recs)
        parsed = [d for d in recs if d["verdict"] in ("A", "B")]
        agree = [d for d in parsed if d["verdict"] == logit[(d["id"], d["criterion"], d["order"])]["letter"]]
        # pairs complete (all 4 judgments generated)
        per_g, per_l = {}, {}
        for d in recs:
            per_g.setdefault(d["id"], {})[(d["criterion"], d["order"])] = d["verdict"]
        complete = [i for i, j in per_g.items() if all(p_ in j for p_ in PASSES)]
        per_g = {i: per_g[i] for i in complete}
        okpairs = [i for i in complete if all(per_g[i][p_] in ("A", "B") for p_ in PASSES)]
        for i in complete:
            per_l[i] = {p_: logit[(i, *p_)]["letter"] for p_ in PASSES}
        sg = summ_block({i: per_g[i] for i in okpairs})
        sl = summ_block({i: per_l[i] for i in okpairs})
        sl_all = summ_block(per_l)
        # per-pair category agreement
        cat_agree = sum(M.classify(per_g[i], "W1") == M.classify(per_l[i], "W1") for i in okpairs)
        def marg(crit, src):
            v = [src[i][(crit, o)] for i in okpairs for o in ("cf", "rf")]
            return sum(x == "A" for x in v) / len(v) if v else float("nan")
        by_crit = {}
        for crit in ("better", "W1"):
            pc = [d for d in parsed if d["criterion"] == crit]
            by_crit[crit] = {"n": len(pc), "agree": sum(d["verdict"] == logit[(d["id"], d["criterion"], d["order"])]["letter"] for d in pc) / len(pc) if pc else float("nan")}
        unpars = [d["generation"] for d in recs if d["verdict"] is None]
        gens = {}
        for d in recs: gens[d["generation"].strip()[:30]] = gens.get(d["generation"].strip()[:30], 0) + 1
        top_gens = sorted(gens.items(), key=lambda x: -x[1])[:8]
        S = {"n_judgments": n_j, "n_pairs_complete": len(complete), "n_parsed": len(parsed),
             "parse_rate": len(parsed) / n_j if n_j else float("nan"),
             "agree_with_logit": len(agree) / len(parsed) if parsed else float("nan"), "agree_by_criterion": by_crit,
             "n_pairs_all4_parsed": len(okpairs), "pair_category_agreement": cat_agree / len(okpairs) if okpairs else float("nan"),
             "P_A_gen": {c: marg(c, per_g) for c in ("better", "W1")},
             "P_A_logit": {c: marg(c, per_l) for c in ("better", "W1")},
             "metrics_gen": sg, "metrics_logit_same_pairs": sl, "metrics_logit_all_complete": sl_all,
             "unparsable_examples": unpars[:5], "top_generations": top_gens,
             "mean_sec_per_judgment": sum(d["sec"] for d in recs) / n_j if n_j else float("nan")}
        summary[model] = S
        f3 = lambda x: f"{x:.2f}"
        lines += [f"## {model} (rev {g['revision'][:12]})", "",
                  f"- judgments generated: {n_j} ({len(complete)} complete pairs); parsed: {len(parsed)}/{n_j} = {f3(S['parse_rate'])}",
                  f"- agreement generated vs logit argmax: {len(agree)}/{len(parsed)} = {f3(S['agree_with_logit'])} "
                  f"(better: {f3(by_crit['better']['agree'])}, n={by_crit['better']['n']}; W1: {f3(by_crit['W1']['agree'])}, n={by_crit['W1']['n']})",
                  f"- per-pair taxonomy category agreement (gen vs logit): {cat_agree}/{len(okpairs)} = {f3(S['pair_category_agreement'])}",
                  f"- P(letter A) better/W1: generated {f3(S['P_A_gen']['better'])}/{f3(S['P_A_gen']['W1'])}; "
                  f"logit {f3(S['P_A_logit']['better'])}/{f3(S['P_A_logit']['W1'])} (n={len(okpairs)} pairs x 2 orders)",
                  f"- most frequent generations (first 30 chars, count): {top_gens}",
                  f"- unparsable examples: {unpars[:5]}", "",
                  "| readout | n pairs | acc(better, both orders) [CI] | correct_reversal | position_locked | criterion_blind | reversed_consistent | other | cond_flip | pos_locked share of non-CR |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, s in [("generated", sg), ("logit (same pairs)", sl), ("logit (all complete pairs)", sl_all)]:
            lines.append(f"| {name} | {s['n']} | {f3(s['acc_better_poscons'])} [{f3(s['acc_ci'][0])},{f3(s['acc_ci'][1])}] | "
                         f"{f3(s['share_correct_reversal'])} | {f3(s['share_position_locked'])} | {f3(s['share_criterion_blind'])} | "
                         f"{f3(s['share_reversed_consistent'])} | {f3(s['share_other'])} | {f3(s['cond_flip_rate'])} (n={s['n_cond']}) | "
                         f"{f3(s['pos_locked_share_of_noncr'])} (n={s['n_noncr']}) |")
        lines.append("")
    with open(os.path.join(RES, "gen_check.md"), "w") as f: f.write("\n".join(lines))
    with open(os.path.join(RES, "gen_check_summary.json"), "w") as f: json.dump(summary, f, indent=1)
    print("\n".join(lines))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "report"])
    ap.add_argument("--model", default="Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    if a.mode == "gen": gen(a.model, a.n)
    else: report()
