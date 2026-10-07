"""Letter-prior-calibrated re-analysis (referee fix 1,2,4). Run: python -I src/calibrated.py -> results/calibrated.json/.md
d_c = (score_cf - score_rf)/2 : >0 means judge prefers the chosen response, letter offset removed."""
import json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
MODELS = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]  # 4B = positive control, resample 1 only
rng = np.random.default_rng(0)
def load(m):
    R = {}
    for s in (0, 1, 2):
        if not os.path.exists(f"{RES}/raw_{m}_r{s}.json"): continue
        for r in json.load(open(f"{RES}/raw_{m}_r{s}.json"))["records"]:
            R.setdefault(r["id"], {})[(r["criterion"], r["order"])] = r
    return R
def boot(fn, n, B=2000):
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
out, L = {}, ["# Calibrated re-analysis (letter offset removed)", "",
 "d = (score_cf - score_rf)/2 per criterion; `better` vs `W1`. corr = Pearson(d_better,d_W1); same_sign = share with sign(d_better)==sign(d_W1) (criterion-blind if high; reversal if ~0); acc_cal = share with d_better>0; reverse_cal = share with d_better>0 & d_W1<0. Hep = HumanEvalPack ids (hep-*). CI = bootstrap over pairs, 95%.", "",
 "| model | subset | n | corr(better,W1) [CI] | same_sign [CI] | acc_cal(better) | reverse_cal [CI] | P(letter A) better/W1 |", "|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    R = load(m); ids = sorted(i for i in R if all(k in R[i] for k in [("better","cf"),("better","rf"),("W1","cf"),("W1","rf")]))
    sub = {i: R[i][("better","cf")]["subset"] for i in ids}
    db = np.array([(R[i][("better","cf")]["score"]-R[i][("better","rf")]["score"])/2 for i in ids])
    dw = np.array([(R[i][("W1","cf")]["score"]-R[i][("W1","rf")]["score"])/2 for i in ids])
    mass = np.array([min(R[i][(c,o)]["mass_AB"] for c in ("better","W1") for o in ("cf","rf")) for i in ids])
    isH = np.array([str(sub[i]).startswith("hep") for i in ids])
    pa = {c: float(np.mean([R[i][(c,o)]["letter"]=="A" for i in ids for o in ("cf","rf")])) for c in ("better","W1")}
    for name, mk in [("all", np.ones(len(ids),bool)), ("non-hep", ~isH), ("hep", isH), ("mass>0.9 all 4 passes", mass>0.9)]:
        b, w = db[mk], dw[mk]; n = len(b)
        if n < 10: continue
        corr = lambda ix: np.corrcoef(b[ix], w[ix])[0,1]
        ss = lambda ix: np.mean(np.sign(b[ix]) == np.sign(w[ix]))
        rv = lambda ix: np.mean((b[ix]>0)&(w[ix]<0))
        full = np.arange(n)
        r = {"n": n, "corr": float(corr(full)), "corr_ci": boot(corr, n), "same_sign": float(ss(full)), "same_sign_ci": boot(ss, n),
             "acc_cal": float(np.mean(b>0)), "reverse_cal": float(rv(full)), "reverse_cal_ci": boot(rv, n), "pA": pa}
        out.setdefault(m, {})[name] = r
        L.append(f"| {m} | {name} | {n} | {r['corr']:.2f} [{r['corr_ci'][0]:.2f},{r['corr_ci'][1]:.2f}] | {r['same_sign']:.2f} [{r['same_sign_ci'][0]:.2f},{r['same_sign_ci'][1]:.2f}] | {r['acc_cal']:.2f} | {r['reverse_cal']:.2f} [{r['reverse_cal_ci'][0]:.2f},{r['reverse_cal_ci'][1]:.2f}] | {pa['better']:.2f}/{pa['W1']:.2f} |")
    # letter-biased null: independent letters at observed per-criterion rates
    pb = float(np.mean([R[i][("better",o)]["letter"]=="A" for i in ids for o in ("cf","rf")]))
    pw = float(np.mean([R[i][("W1",o)]["letter"]=="A" for i in ids for o in ("cf","rf")]))
    obs = np.mean([len({R[i][(c,o)]["letter"] for c in ("better","W1") for o in ("cf","rf")})==1 for i in ids])
    exp = pb**2*pw**2 + (1-pb)**2*(1-pw)**2
    out[m]["letter_null"] = {"pA_better": pb, "pA_W1": pw, "all_four_same_letter_expected": float(exp), "observed": float(obs),
        "pos_consistent_acc_expected_better": float(pb*(1-pb))}
    L.append(f"| {m} | letter-null (all four passes same letter) | {len(ids)} | expected {exp:.2f} | observed {obs:.2f} | | | |")
json.dump(out, open(f"{RES}/calibrated.json","w"), indent=1)
open(f"{RES}/calibrated.md","w").write("\n".join(L)+"\n")
print("\n".join(L))
