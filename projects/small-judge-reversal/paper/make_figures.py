"""Generate paper figures from committed raw results (no hand-drawn data).

Inputs : ../results/analysis.json (written by src/analyze.py), ../results/raw_<model>_r<k>.json
Outputs: figures/fig_taxonomy.pdf, figures/fig_letter_share.pdf, figures/fig_calibrated.pdf,
         figures/fig_positive_control.pdf (Qwen3-4B vs Qwen3-1.7B, resample 1), figures/paper_stats.json
         (letter shares per criterion, summed per-pass forward times, and calibrated r / |d| / letter-offset
         scale checks, quoted in main.tex)
Run    : python -I paper/make_figures.py   (needs numpy + matplotlib; versions used: paper/requirements-paper.txt)
"""
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

MODELS = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen3-0.6B", "Qwen3-1.7B"]
SHORT = {"Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
         "Qwen3-0.6B": "Qwen3-0.6B", "Qwen3-1.7B": "Qwen3-1.7B"}
CATS = ["correct_reversal", "position_locked", "criterion_blind", "reversed_consistent", "other"]
CAT_LABEL = {"correct_reversal": "correct reversal", "position_locked": "position-locked",
             "criterion_blind": "criterion-blind", "reversed_consistent": "reversed-consistent",
             "other": "other"}
COLORS = {"correct_reversal": "#1b9e77", "position_locked": "#d95f02", "criterion_blind": "#7570b3",
          "reversed_consistent": "#e7298a", "other": "#bdbdbd"}

plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})

# Figure 1: pair-level taxonomy shares (W1), 300 pooled pairs per model + analytic random judge.
an = json.load(open(os.path.join(RES, "analysis.json")))
rows = [(SHORT[m], an["models"][m]["pooled"]) for m in MODELS]
rows.append(("random judge\n(analytic)", an["reference"]["random_judge"]))
fig, ax = plt.subplots(figsize=(3.45, 1.9))
for y, (name, p) in enumerate(rows):
    left = 0.0
    for c in CATS:
        w = p["share_" + c]
        ax.barh(y, w, left=left, color=COLORS[c], edgecolor="white", linewidth=0.4,
                label=CAT_LABEL[c] if y == 0 else None)
        left += w
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("share of pairs (n=300 per model)")
ax.legend(ncol=3, fontsize=6, loc="lower center", bbox_to_anchor=(0.42, 1.0), frameon=False,
          handlelength=1.0, columnspacing=0.8)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig_taxonomy.pdf"))
plt.close(fig)

# Figure 2: share of passes answered "A", per criterion, from raw per-pass records (all resamples).
CRITS = [("better", "better"), ("W1", "worse (W1)"), ("W2", "rejected (W2)"), ("length", "longer")]
share, fwd_sec = {}, {}
for m in MODELS:
    cnt = defaultdict(Counter)
    fwd_sec[m] = 0.0
    for s in (0, 1, 2):
        for r in json.load(open(os.path.join(RES, f"raw_{m}_r{s}.json")))["records"]:
            cnt[r["criterion"]][r["letter"]] += 1
            fwd_sec[m] += r["sec"]
    share[m] = {c: (cnt[c]["A"] / sum(cnt[c].values()), sum(cnt[c].values())) for c, _ in CRITS}
fig, ax = plt.subplots(figsize=(3.45, 1.85))
width = 0.2
for k, (c, lab) in enumerate(CRITS):
    xs = [i + (k - 1.5) * width for i in range(len(MODELS))]
    ax.bar(xs, [share[m][c][0] for m in MODELS], width, label=lab,
           color=["#4d4d4d", "#d95f02", "#fdae6b", "#7570b3"][k])
ax.axhline(0.5, color="black", lw=0.5, ls=":")
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([SHORT[m].replace("-", "\n", 1) for m in MODELS])
ax.set_ylim(0, 1.05)
ax.set_ylabel('share "A"')
ax.legend(ncol=4, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
          handlelength=1.0, columnspacing=0.8)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig_letter_share.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# Figure 3 (added for referee round 1): letter-calibrated content preference under "better" vs W1.
# d = (score_cf - score_rf)/2, same definition as src/calibrated.py (>0: prefers the chosen response;
# under W1, >0 means the chosen response is named as the WORSE one). A criterion-following judge would
# put points in the lower-right quadrant (d_better>0, d_W1<0); criterion-blindness gives a positive slope.
import numpy as np
cal = {}
fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.85))
for ax, m in zip(axs, MODELS):
    R = {}
    for s in (0, 1, 2):
        for r in json.load(open(os.path.join(RES, f"raw_{m}_r{s}.json")))["records"]:
            R.setdefault(r["id"], {})[(r["criterion"], r["order"])] = r
    keys = [("better", "cf"), ("better", "rf"), ("W1", "cf"), ("W1", "rf")]
    ids = sorted(i for i in R if all(k in R[i] for k in keys))
    db = np.array([(R[i][("better", "cf")]["score"] - R[i][("better", "rf")]["score"]) / 2 for i in ids])
    dw = np.array([(R[i][("W1", "cf")]["score"] - R[i][("W1", "rf")]["score"]) / 2 for i in ids])
    hep = np.array([str(R[i][("better", "cf")]["subset"]).startswith("hep") for i in ids])
    # scale check (referee round 2): size of content shift d vs the letter offset o = (score_cf + score_rf)/2
    off_b = np.array([(R[i][("better", "cf")]["score"] + R[i][("better", "rf")]["score"]) / 2 for i in ids])
    cal[m] = {"n": len(ids), "n_hep": int(hep.sum()), "corr": round(float(np.corrcoef(db, dw)[0, 1]), 2),
              "median_abs_d_better": round(float(np.median(np.abs(db))), 3),
              "median_abs_d_W1": round(float(np.median(np.abs(dw))), 3),
              "mean_offset_better": round(float(np.mean(off_b)), 2)}
    ax.scatter(db[hep], dw[hep], s=2, color="#7570b3", alpha=0.6, lw=0, label="HumanEvalPack")
    ax.scatter(db[~hep], dw[~hep], s=2, color="#d95f02", alpha=0.8, lw=0, label="other")
    lim = float(np.max(np.abs(np.concatenate([db, dw])))) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="black", lw=0.4); ax.axvline(0, color="black", lw=0.4)
    ax.set_title(f"{SHORT[m]}  r={cal[m]['corr']:.2f}", fontsize=7)
    ax.set_xlabel(r"$d_\mathrm{better}$", fontsize=7)
    ax.tick_params(labelsize=6)
axs[0].set_ylabel(r"$d_\mathrm{W1}$", fontsize=7)
axs[0].legend(fontsize=5, loc="lower right", frameon=False, markerscale=3, handletextpad=0.2)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig_calibrated.pdf"))
plt.close(fig)

# Positive control (cycle 22): Qwen3-4B, resample 1 only (100 pairs). To compare like with like, every judge is
# re-scored on the SAME 100 resample-1 pairs (same d definition as src/calibrated.py). Point estimates for the 4B
# judge equal results/calibrated.json; the bootstrap here uses its own generator (seed 0, 2000 resamples), so the
# CIs for the 4B judge are quoted from results/calibrated.json in the paper and the CIs below are used only for
# the r1-only rows of the small judges and for the paired differences (4B minus 1.7B, same pairs).
PC, PC_RES = "Qwen3-4B", 1
rng = np.random.default_rng(0)
keys = [("better", "cf"), ("better", "rf"), ("W1", "cf"), ("W1", "rf")]


def load_r(m, s):
    R = {}
    for r in json.load(open(os.path.join(RES, f"raw_{m}_r{s}.json")))["records"]:
        R.setdefault(r["id"], {})[(r["criterion"], r["order"])] = r
    return R


def dvec(R, ids):
    db = np.array([(R[i][("better", "cf")]["score"] - R[i][("better", "rf")]["score"]) / 2 for i in ids])
    dw = np.array([(R[i][("W1", "cf")]["score"] - R[i][("W1", "rf")]["score"]) / 2 for i in ids])
    return db, dw


def metrics(b, w):
    return {"corr": float(np.corrcoef(b, w)[0, 1]), "same_sign": float(np.mean(np.sign(b) == np.sign(w))),
            "acc_cal": float(np.mean(b > 0)), "reverse_cal": float(np.mean((b > 0) & (w < 0))),
            "slope": float(np.polyfit(b, w, 1)[0]),
            "median_ratio": float(np.median(np.abs(w) / np.abs(b)))}


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


Rpc = load_r(PC, PC_RES)
ids_pc = sorted(i for i in Rpc if all(k in Rpc[i] for k in keys))
pc = {"n_pairs": len(ids_pc), "resample": PC_RES, "per_judge_r1": {}, "diff_4B_minus_1.7B_r1": {}}
D = {}
for m in MODELS + [PC]:
    R = Rpc if m == PC else load_r(m, PC_RES)
    assert all(all(k in R[i] for k in keys) for i in ids_pc), m
    b, w = dvec(R, ids_pc)
    D[m] = (b, w)
    point = metrics(b, w)
    boots = [metrics(b[ix], w[ix]) for ix in (rng.integers(0, len(b), len(b)) for _ in range(2000))]
    e = {k: round(v, 3) for k, v in point.items()}
    for k in ("corr", "same_sign", "reverse_cal", "slope"):
        e[k + "_ci"] = ci([x[k] for x in boots])
    e["n_hep"] = int(sum(str(R[i][("better", "cf")]["subset"]).startswith("hep") for i in ids_pc))
    e["median_abs_d_better"] = round(float(np.median(np.abs(b))), 3)
    e["mean_offset_better"] = round(float(np.mean(
        [(R[i][("better", "cf")]["score"] + R[i][("better", "rf")]["score"]) / 2 for i in ids_pc])), 2)
    pc["per_judge_r1"][m] = e
# paired bootstrap over the same 100 pairs: 4B minus Qwen3-1.7B
b4, w4 = D[PC]
b17, w17 = D["Qwen3-1.7B"]
diffs = []
for _ in range(2000):
    ix = rng.integers(0, len(b4), len(b4))
    m4, m17 = metrics(b4[ix], w4[ix]), metrics(b17[ix], w17[ix])
    diffs.append({k: m4[k] - m17[k] for k in ("corr", "same_sign", "acc_cal", "reverse_cal", "slope")})
m4, m17 = metrics(b4, w4), metrics(b17, w17)
for k in ("corr", "same_sign", "acc_cal", "reverse_cal", "slope"):
    pc["diff_4B_minus_1.7B_r1"][k] = {"diff": round(m4[k] - m17[k], 3), "ci": ci([x[k] for x in diffs])}
recs = json.load(open(os.path.join(RES, f"raw_{PC}_r{PC_RES}.json")))["records"]
mass = np.array([r["mass_AB"] for r in recs if r["criterion"] in ("better", "W1")])
pc["Qwen3-4B_mass"] = {"mean_mass_AB": round(float(mass.mean()), 3), "share_mass_lt_0.5": round(float(np.mean(mass < 0.5)), 3),
                       "n_passes": int(len(mass))}
pc["Qwen3-4B_forward_seconds_sum"] = round(sum(r["sec"] for r in recs))

# Figure 4: same-pair scatter, Qwen3-1.7B vs Qwen3-4B on the 100 resample-1 pairs.
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75))
for ax, m, lab in zip(axs, ["Qwen3-1.7B", PC], ["Qwen3-1.7B", "Qwen3-4B"]):
    b, w = D[m]
    hp = np.array([str(Rpc[i][("better", "cf")]["subset"]).startswith("hep") for i in ids_pc])
    ax.scatter(b[hp], w[hp], s=3, color="#7570b3", alpha=0.7, lw=0, label="HumanEvalPack")
    ax.scatter(b[~hp], w[~hp], s=3, color="#d95f02", alpha=0.9, lw=0, label="other")
    lim = float(np.max(np.abs(np.concatenate([b, w])))) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="black", lw=0.4); ax.axvline(0, color="black", lw=0.4)
    ax.set_title(f"{lab}  r={pc['per_judge_r1'][m]['corr']:.2f}", fontsize=7)
    ax.set_xlabel(r"$d_\mathrm{better}$", fontsize=7)
    ax.tick_params(labelsize=6)
axs[0].set_ylabel(r"$d_\mathrm{W1}$", fontsize=7)
axs[0].legend(fontsize=5, loc="lower right", frameon=False, markerscale=3, handletextpad=0.2)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig_positive_control.pdf"))
plt.close(fig)

stats = {"calibrated_check": cal, "positive_control_r1": pc,
         "letter_share_A": {m: {c: {"share_A": round(v[0], 3), "n_passes": v[1]} for c, v in d.items()}
                            for m, d in share.items()},
         "forward_seconds_sum": {m: round(v) for m, v in fwd_sec.items()},
         "forward_hours_total": round(sum(fwd_sec.values()) / 3600, 2)}
with open(os.path.join(OUT, "paper_stats.json"), "w") as f:
    json.dump(stats, f, indent=1)
print(json.dumps({SHORT[m]: {c: [round(v[0], 3), v[1]] for c, v in d.items()} for m, d in share.items()}))
