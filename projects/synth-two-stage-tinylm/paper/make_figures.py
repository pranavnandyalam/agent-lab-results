"""Generate paper figures from raw results (no hand-drawn values).

Fig. 1 (fig_ppl.pdf): R_val perplexity per arm, N=1M and N=4M; per-seed points from
  results/grid/<arm>_N<N>_s<seed>/eval_R_val_perdoc.npz (same computation as src/analyze.py),
  bar = mean over seeds.
Fig. 2 (fig_diffs.pdf): paired differences (a - b) with hierarchical-bootstrap 95% CIs parsed
  verbatim from results/analysis.md (output of src/analyze.py) and per-seed differences.

Usage (needs numpy + matplotlib):  python -I paper/make_figures.py  (from anywhere)
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRID = os.path.join(ROOT, "results", "grid")
OUT = os.path.join(HERE, "figures")
ARMS = ["real-only", "mixed", "S-R", "R-S", "real-only-2N", "real-only-2ep", "mixed-realtail"]
LABEL = {"real-only": "real-only\n(N)", "mixed": "mixed\n(2N)", "S-R": "S→R\n(2N)", "R-S": "R→S\n(2N)",
         "real-only-2N": "real-2N\n(2N)", "real-only-2ep": "real-2ep\n(2N)", "mixed-realtail": "mix→real\n(3N)"}
COL = {"real-only": "0.6", "mixed": "#4c72b0", "S-R": "#dd8452", "R-S": "#8172b2",
       "real-only-2N": "#55a868", "real-only-2ep": "#2d6a3e", "mixed-realtail": "0.35"}

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def ppl(arm, N, s):
    z = np.load(os.path.join(GRID, f"{arm}_N{N}_s{s}", "eval_R_val_perdoc.npz"))
    return float(np.exp(z["nll_sum"].sum() / z["count"].sum()))


def fig_ppl():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for ax, N in zip(axes, ["1M", "4M"]):
        for i, arm in enumerate(ARMS):
            v = [ppl(arm, N, s) for s in range(3)]
            ax.bar(i, np.mean(v), color=COL[arm], width=0.7, alpha=0.85)
            ax.scatter([i - 0.15, i, i + 0.15], v, s=6, color="k", zorder=3)
            ax.text(i, np.mean(v) * 1.02, f"{np.mean(v):.2f}", ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=5.5)
        ax.set_title(f"N = {N} real tokens", fontsize=7)
        ax.set_ylabel("R_val perplexity (lower is better)" if N == "1M" else "")
        ax.set_ylim(0, max(ppl("real-only", N, s) for s in range(3)) * 1.12)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_ppl.pdf"))
    plt.close(fig)


ROW = re.compile(r"^\| (1M|4M) \| (\S+) − (\S+) \| 3 \| ([+-][\d.]+) \| \[([+-][\d.]+), ([+-][\d.]+)\] \| ([^|]+) \|")
SHOW = [("S-R", "mixed"), ("S-R", "R-S"), ("S-R", "real-only-2N"), ("S-R", "real-only-2ep"),
        ("mixed", "real-only-2ep"), ("real-only-2ep", "real-only-2N")]
NAME = {"S-R": "S→R", "R-S": "R→S", "mixed": "mixed", "real-only-2N": "real-2N", "real-only-2ep": "real-2ep"}


def fig_diffs():
    rows = {}
    for line in open(os.path.join(ROOT, "results", "analysis.md"), encoding="utf-8"):
        m = ROW.match(line)
        if m:
            N, a, b, mean, lo, hi, per = m.groups()
            rows[(N, a, b)] = (float(mean), float(lo), float(hi), [float(x) for x in per.split(",")])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9), sharey=True)
    for ax, N in zip(axes, ["1M", "4M"]):
        for j, (a, b) in enumerate(SHOW):
            mean, lo, hi, per = rows[(N, a, b)]
            y = len(SHOW) - 1 - j
            ax.plot([lo, hi], [y, y], color="k", lw=1.0)
            ax.plot(mean, y, "o", color="#dd8452", ms=3.5, zorder=3)
            ax.scatter(per, [y + 0.22] * 3, s=14, color="0.3", marker="|", linewidths=0.8, zorder=2)
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.set_yticks(range(len(SHOW)))
        ax.set_yticklabels([f"{NAME[a]} − {NAME[b]}" for a, b in SHOW[::-1]])
        ax.set_xlabel("paired Δ perplexity (negative = first arm better)")
        ax.set_title(f"N = {N}: mean, 95% CI, per-seed ticks", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_diffs.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_ppl()
    fig_diffs()
    for f in sorted(os.listdir(OUT)):
        print(f, os.path.getsize(os.path.join(OUT, f)), "bytes")
