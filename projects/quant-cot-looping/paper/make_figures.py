"""Generate the paper figures from the committed raw results (read-only).

Inputs : ../results/core/<level>_chunk<k>.json (raw generations, token ids) and ../results/core/analysis.json
Outputs: figures/fig_length_split.pdf, figures/fig_h1_share.pdf, figures/fig_trace_scatter.pdf
Per-trace loop tokens and correctness are recomputed with the frozen detector (src/detector.py) and the
scoring rule (src/analyze_core.py); the script asserts that per-level means match analysis.json exactly.

Usage (needs numpy + matplotlib; any venv):
  python -I projects/quant-cot-looping/paper/make_figures.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from analyze_core import load_batches, masks, correct, LEVELS, EFF_BITS  # noqa: E402

PRIMARY = "primary_A20x4+B"
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "figure.dpi": 150})
LABEL = {"fp32": "fp32\n(32)", "w5g64": "w5g64\n(5.5)", "w4g64": "w4g64\n(4.5)", "w3g32": "w3g32\n(4.0)"}
COL = {"fp32": "#1b9e77", "w5g64": "#7570b3", "w4g64": "#d95f02", "w3g32": "#e7298a"}

ana = json.load(open(os.path.join(PROJ, "results", "core", "analysis.json")))
data, present, missing = load_batches()
assert not missing, missing

# ---- per-trace table ----
T = {}
for lv in LEVELS:
    rows = []
    for r in data[lv]:
        m = masks(r)
        rows.append(dict(n=r["gen_tokens"], loop=int(sum(m[PRIMARY])), trunc=bool(r["truncated"]), ok=correct(r)))
    T[lv] = rows
    s = ana["detectors"][PRIMARY]["levels"][lv]
    assert abs(np.mean([t["n"] for t in rows]) - s["mean_tokens"]) < 1e-9, lv
    assert abs(np.mean([t["loop"] for t in rows]) - s["mean_loop_tokens"]) < 1e-9, lv
    assert sum(t["trunc"] for t in rows) == s["n_truncated"], lv
    assert abs(np.mean([t["ok"] for t in rows]) - s["accuracy"]) < 1e-9, lv
print("per-level means match analysis.json")

# ---- Fig 1: mean tokens split into loop / non-loop, with per-trace lengths ----
fig, ax = plt.subplots(figsize=(3.4, 2.2))
rng = np.random.default_rng(0)
for i, lv in enumerate(LEVELS):
    s = ana["detectors"][PRIMARY]["levels"][lv]
    nl = s["mean_tokens"] - s["mean_loop_tokens"]
    ax.bar(i, nl, width=0.6, color="#bbbbbb", edgecolor="k", lw=0.4, label="mean non-loop" if i == 0 else None)
    ax.bar(i, s["mean_loop_tokens"], bottom=nl, width=0.6, color="#d62728", edgecolor="k", lw=0.4,
           label="mean loop (20-gram)" if i == 0 else None)
    x = i + rng.uniform(-0.22, 0.22, len(T[lv]))
    ax.scatter(x, [t["n"] for t in T[lv]], s=4, color="k", alpha=0.35, lw=0,
               label="trace length" if i == 0 else None)
    ax.text(i, 2120, f"{s['n_truncated']}/48 cap", ha="center", va="bottom", fontsize=6.5)
ax.axhline(2048, ls=":", lw=0.6, color="k")
ax.set_xticks(range(len(LEVELS)), [LABEL[lv] for lv in LEVELS])
ax.set_ylabel("generated tokens")
ax.set_xlabel("level (effective bits/weight)")
ax.set_ylim(0, 2300)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, ncol=3, frameon=False,
          handlelength=1.2, columnspacing=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_length_split.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---- Fig 2: H1 share with bootstrap CI, three detectors ----
DET = [(PRIMARY, "20-gram (primary)", "o"), ("sens_A16x4+B", "16-gram", "s"), ("sens_A12x4+B", "12-gram", "^")]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
for j, lv in enumerate(["w5g64", "w4g64"]):
    for k, (det, name, mk) in enumerate(DET):
        h = ana["detectors"][det]["H1"][lv]
        x = j + (k - 1) * 0.18
        lo, hi = h["share_q_ci95"]
        ax.errorbar(x, h["share_q"], yerr=[[h["share_q"] - lo], [hi - h["share_q"]]], fmt=mk, ms=4,
                    color=["k", "#555555", "#999999"][k], capsize=2, lw=0.8, label=name if j == 0 else None)
ax.axhline(0.5, ls="--", lw=0.7, color="#d62728")
ax.text(0.5, 0.53, "H1 threshold 0.5", color="#d62728", fontsize=6, ha="center")
ax.axhline(0, lw=0.4, color="k")
ax.set_xticks([0, 1], ["w5g64 (5.5 bits)", "w4g64 (4.5 bits)"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("loop share of added tokens")
ax.legend(fontsize=6, loc="upper left", ncol=3, frameon=False)
ax.set_ylim(-0.2, 1.1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_h1_share.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---- Fig 3: per-trace length vs loop fraction ----
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for lv in LEVELS:
    n = np.array([t["n"] for t in T[lv]]); lf = np.array([t["loop"] / t["n"] for t in T[lv]])
    ok = np.array([t["ok"] for t in T[lv]])
    jit = rng.uniform(-12, 12, len(n)) * (n == 2048)
    ax.scatter(n[ok] + jit[ok], lf[ok], s=10, marker="o", facecolors="none", edgecolors=COL[lv], lw=0.7)
    ax.scatter(n[~ok] + jit[~ok], lf[~ok], s=10, marker="x", color=COL[lv], lw=0.7, label=lv)
ax.scatter([], [], s=10, marker="o", facecolors="none", edgecolors="k", lw=0.7, label="correct (o)")
ax.scatter([], [], s=10, marker="x", color="k", lw=0.7, label="incorrect (x)")
ax.axvline(2048, ls=":", lw=0.6, color="k")
ax.set_xlabel("generated tokens (points at the 2048 cap jittered)")
ax.set_ylabel("loop fraction (20-gram)")
ax.legend(fontsize=6, loc="upper left", frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_trace_scatter.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
