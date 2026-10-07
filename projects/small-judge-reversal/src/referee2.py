"""Referee round 2 analyses from existing logits (no model inference).
Run: OMP_NUM_THREADS=2 .venv/bin/python -I src/referee2.py -> results/referee2.{json,md}
d_q = (score_cf - score_rf)/2, score = A-minus-B logit; >0 = judge favours chosen C as the answer to q (as calibrated.py).
Bootstraps: over pairs, 2000 reps, each analysis block re-seeded with np.random.default_rng(0)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sjr import data as D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
MODELS = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen3-0.6B", "Qwen3-1.7B"]
B = 2000
rb = {r["id"]: r for r in D.load_rb()}

def load(m):
    R, rs = {}, {}
    for s in (0, 1, 2):
        for r in json.load(open(f"{RES}/raw_{m}_r{s}.json"))["records"]:
            R.setdefault(r["id"], {})[(r["criterion"], r["order"])] = r
            rs[r["id"]] = s
    return R, rs

def dq(R, i, q):
    return (R[i][(q, "cf")]["score"] - R[i][(q, "rf")]["score"]) / 2

def pc(R, i, q):
    """order-averaged P(judge picks C | {A,B}) for question q: sigmoid(score) = P(A) in the A/B pair."""
    sg = lambda x: 1 / (1 + np.exp(-x))
    return (sg(R[i][(q, "cf")]["score"]) + sg(-R[i][(q, "rf")]["score"])) / 2

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r

def spearman(x, y): return corr(rank(x), rank(y))

def ci(fn, n, rng):
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)], float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

def lev(a, b):
    """token-level Levenshtein distance (insert/delete/substitute = 1)."""
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]

f2 = lambda x: "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
fci = lambda c: f"[{f2(c[0])}, {f2(c[1])}]"
out = {"meta": {"bootstrap_reps": B, "seed": 0, "d_definition": "(score_cf-score_rf)/2, >0 favours chosen"}}
L = ["# Referee round 2 analyses (from existing logits; no new inference)", "",
     "d_q = (s_q^cf - s_q^rf)/2 (A-minus-B logit, letter offset cancels; >0 = favours chosen C). Bootstrap over pairs, 2000 reps, seed 0 (re-seeded per block). Script: `src/referee2.py`.", ""]

data = {}
for m in MODELS:
    R, rs = load(m)
    ids = sorted(i for i in R if all(k in R[i] for k in [("better", "cf"), ("better", "rf"), ("W1", "cf"), ("W1", "rf")]))
    data[m] = (R, rs, ids)
ids_all = data[MODELS[0]][2]
assert all(data[m][2] == ids_all for m in MODELS), "pair sets differ across models"

# ---------- (a) length positive control ----------
L += ["## (a) Length positive control (resample 0, 'Which response is longer?')", "",
      "dlen_char = len(C)-len(R) in characters. r = Pearson, rho = Spearman. acc_len_cal = share of non-tied pairs with sign(d_length)==sign(dlen_char).", "",
      "| judge | n (non-tied) | r(d_length, dlen_char) [CI] | rho(d_length, dlen_char) | acc_len_cal [CI] | r(d_length, d_better) [CI] (n=100) | r(d_better, dlen_char) | median abs d_length | median abs d_better (r0) |",
      "|---|---|---|---|---|---|---|---|---|"]
out["a_length"] = {}
for m in MODELS:
    R, rs, _ = data[m]
    ids0 = sorted(i for i in R if rs[i] == 0 and ("length", "cf") in R[i] and ("length", "rf") in R[i])
    dl = np.array([dq(R, i, "length") for i in ids0]); db = np.array([dq(R, i, "better") for i in ids0])
    lc = np.array([len(rb[i]["chosen"]) - len(rb[i]["rejected"]) for i in ids0], float)
    nt = lc != 0
    rng = np.random.default_rng(0)
    dln, lcn = dl[nt], lc[nt]; n = int(nt.sum())
    o = {"n_pairs_r0": len(ids0), "n_nontied": n,
         "r_dlen_lenchar": corr(dln, lcn), "r_dlen_lenchar_ci": ci(lambda ix: corr(dln[ix], lcn[ix]), n, rng),
         "rho_dlen_lenchar": spearman(dln, lcn),
         "acc_len_cal": float(np.mean(np.sign(dln) == np.sign(lcn))),
         "acc_len_cal_ci": ci(lambda ix: np.mean(np.sign(dln[ix]) == np.sign(lcn[ix])), n, rng),
         "r_dlen_dbetter": corr(dl, db), "r_dlen_dbetter_ci": ci(lambda ix: corr(dl[ix], db[ix]), len(dl), rng),
         "r_dbetter_lenchar": corr(db[nt], lcn),
         "median_abs_d_length": float(np.median(np.abs(dl))), "median_abs_d_better_r0": float(np.median(np.abs(db)))}
    out["a_length"][m] = o
    L.append(f"| {m} | {n} | {f2(o['r_dlen_lenchar'])} {fci(o['r_dlen_lenchar_ci'])} | {f2(o['rho_dlen_lenchar'])} | {f2(o['acc_len_cal'])} {fci(o['acc_len_cal_ci'])} | {f2(o['r_dlen_dbetter'])} {fci(o['r_dlen_dbetter_ci'])} | {f2(o['r_dbetter_lenchar'])} | {o['median_abs_d_length']:.3f} | {o['median_abs_d_better_r0']:.3f} |")

# ---------- (b) within-family size trend ----------
L += ["", "## (b) Calibrated size trend within family (paired bootstrap over the same 300 pairs)", "",
      "diff = larger minus smaller judge. corr = r(d_better, d_W1); same_sign = share sign(d_better)==sign(d_W1).", "",
      "| family | n | metric | small | large | diff (large-small) [95% CI] |", "|---|---|---|---|---|---|"]
out["b_size_trend"] = {}
for fam, (s, l) in {"Qwen2.5 0.5B->1.5B": ("Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"),
                    "Qwen3 0.6B->1.7B": ("Qwen3-0.6B", "Qwen3-1.7B")}.items():
    V = {}
    for m in (s, l):
        R, _, ids = data[m]
        V[m] = (np.array([dq(R, i, "better") for i in ids_all]), np.array([dq(R, i, "W1") for i in ids_all]))
    n = len(ids_all)
    mets = {"corr": lambda b, w: corr(b, w), "same_sign": lambda b, w: float(np.mean(np.sign(b) == np.sign(w))),
            "acc_cal": lambda b, w: float(np.mean(b > 0))}
    out["b_size_trend"][fam] = {}
    for name, fn in mets.items():
        rng = np.random.default_rng(0)
        d = lambda ix: fn(V[l][0][ix], V[l][1][ix]) - fn(V[s][0][ix], V[s][1][ix])
        o = {"n": n, "small": fn(*V[s]), "large": fn(*V[l]), "diff": d(np.arange(n)), "diff_ci": ci(d, n, rng)}
        out["b_size_trend"][fam][name] = o
        L.append(f"| {fam} | {n} | {name} | {f2(o['small'])} | {f2(o['large'])} | {o['diff']:+.2f} {fci(o['diff_ci'])} |")

# ---------- (c) per-judge slope, ratio, probability check ----------
L += ["", "## (c) Per judge: OLS slope, magnitude ratio, order-averaged probabilities (all 300 pairs)", "",
      "slope = OLS of d_W1 on d_better (with intercept). ratio = median over pairs of |d_W1|/|d_better| (pairs with d_better != 0). "
      "P_C(q) = order-averaged P(pick C) = [sigmoid(s_cf) + sigmoid(-s_rf)]/2 (A-vs-B two-way softmax). "
      "Sum S = P_C(better)+P_C(W1): criterion-following => S~1 with P_C(better) far from 0.5; criterion-blind => P_C(better)~P_C(W1) so S~2*P_C(better); "
      "a letter-only judge gives P_C=0.5 for both, S=1. So S~1 alone does not identify criterion-blindness; Delta = P_C(better)-P_C(W1) (blind ~0, following >0) is reported alongside.", "",
      "| judge | n | slope [CI] | median abs ratio | mean P_C(better) | mean P_C(W1) | mean S [CI] | mean abs(S-1) | mean Delta [CI] | mean abs(P_C(better)-0.5) |",
      "|---|---|---|---|---|---|---|---|---|---|"]
out["c_per_judge"] = {}
for m in MODELS:
    R, _, _ = data[m]
    b = np.array([dq(R, i, "better") for i in ids_all]); w = np.array([dq(R, i, "W1") for i in ids_all])
    pb = np.array([pc(R, i, "better") for i in ids_all]); pw = np.array([pc(R, i, "W1") for i in ids_all])
    S, De = pb + pw, pb - pw; n = len(b)
    slope = lambda ix: float(np.polyfit(b[ix], w[ix], 1)[0])
    nz = b != 0
    rng = np.random.default_rng(0)
    o = {"n": n, "slope": slope(np.arange(n)), "slope_ci": ci(slope, n, rng),
         "median_ratio": float(np.median(np.abs(w[nz]) / np.abs(b[nz]))), "n_ratio": int(nz.sum()),
         "mean_PC_better": float(pb.mean()), "mean_PC_W1": float(pw.mean()),
         "mean_S": float(S.mean()), "mean_S_ci": ci(lambda ix: S[ix].mean(), n, rng), "mean_abs_S_minus_1": float(np.mean(np.abs(S - 1))),
         "mean_Delta": float(De.mean()), "mean_Delta_ci": ci(lambda ix: De[ix].mean(), n, rng),
         "mean_abs_PC_better_minus_half": float(np.mean(np.abs(pb - 0.5)))}
    out["c_per_judge"][m] = o
    L.append(f"| {m} | {n} | {o['slope']:.2f} {fci(o['slope_ci'])} | {o['median_ratio']:.2f} | {o['mean_PC_better']:.3f} | {o['mean_PC_W1']:.3f} | {o['mean_S']:.3f} {fci(o['mean_S_ci'])} | {o['mean_abs_S_minus_1']:.3f} | {o['mean_Delta']:+.3f} {fci(o['mean_Delta_ci'])} | {o['mean_abs_PC_better_minus_half']:.3f} |")

# ---------- (d) HEP textual overlap ----------
hep = [i for i in ids_all if str(rb[i]["subset"]).startswith("hep")]
ed = {i: lev(rb[i]["chosen"].split(), rb[i]["rejected"].split()) for i in hep}
ev = np.array([ed[i] for i in hep])
out["d_hep_overlap"] = {"n_hep": len(hep), "edit_distance_quantiles": {q: float(np.percentile(ev, q)) for q in (0, 10, 25, 50, 75, 90, 100)},
                        "n_edit0": int((ev == 0).sum()), "per_judge": {}}
L += ["", "## (d) HEP pairs: token edit distance between C and R, stratified", "",
      f"Whitespace tokens, token-level Levenshtein. n_hep = {len(hep)}; edit distance quantiles (0/10/25/50/75/90/100%): "
      + "/".join(f"{out['d_hep_overlap']['edit_distance_quantiles'][q]:g}" for q in (0, 10, 25, 50, 75, 90, 100))
      + f"; pairs with distance 0: {out['d_hep_overlap']['n_edit0']}.", "",
      "| judge | stratum | n | r(d_better,d_W1) [CI] | same_sign | acc_cal(better) [CI] | median abs d_better |", "|---|---|---|---|---|---|---|"]
strata = [("ed<5", ev < 5), ("5<=ed<20", (ev >= 5) & (ev < 20)), ("ed>=20", ev >= 20), ("all hep", np.ones(len(ev), bool)), ("hep excl. ed<5", ev >= 5),
          ("fine: ed=1", ev == 1), ("fine: ed=2", ev == 2), ("fine: ed>=3", ev >= 3)]
for m in MODELS:
    R, _, _ = data[m]
    b = np.array([dq(R, i, "better") for i in hep]); w = np.array([dq(R, i, "W1") for i in hep])
    out["d_hep_overlap"]["per_judge"][m] = {}
    for name, mk in strata:
        bb, ww = b[mk], w[mk]; n = len(bb)
        rng = np.random.default_rng(0)
        o = {"n": n, "corr": corr(bb, ww), "corr_ci": ci(lambda ix: corr(bb[ix], ww[ix]), n, rng) if n >= 10 else [float("nan")] * 2,
             "same_sign": float(np.mean(np.sign(bb) == np.sign(ww))) if n else float("nan"),
             "acc_cal": float(np.mean(bb > 0)) if n else float("nan"),
             "acc_cal_ci": ci(lambda ix: np.mean(bb[ix] > 0), n, rng) if n >= 10 else [float("nan")] * 2,
             "median_abs_d_better": float(np.median(np.abs(bb))) if n else float("nan")}
        out["d_hep_overlap"]["per_judge"][m][name] = o
        L.append(f"| {m} | {name} | {n} | {f2(o['corr'])} {fci(o['corr_ci'])} | {f2(o['same_sign'])} | {f2(o['acc_cal'])} {fci(o['acc_cal_ci'])} | {o['median_abs_d_better']:.3f} |")
# non-hep reference for edit distance
nh = [i for i in ids_all if i not in set(hep)]
nhe = np.array([lev(rb[i]["chosen"].split(), rb[i]["rejected"].split()) for i in nh])
out["d_hep_overlap"]["non_hep_edit_distance_median"] = float(np.median(nhe)); out["d_hep_overlap"]["n_non_hep"] = len(nh)
L += ["", f"Reference: non-HEP pairs (n={len(nh)}) median token edit distance {np.median(nhe):g}."]

json.dump(out, open(f"{RES}/referee2.json", "w"), indent=1)
open(f"{RES}/referee2.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
