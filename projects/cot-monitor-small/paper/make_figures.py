"""Generate paper figures from the committed analysis output (no hand-entered data).

Input : ../results/analysis_test.json (written by `python -I src/analyze.py --split test --prompt v2`)
Output: figures/fig_auroc.pdf  (AUROC per contrast and view, 95% skeleton-bootstrap CIs, BoW reference line)
        figures/fig_effects.pdf (K_cot and O with 95% skeleton-bootstrap CIs)
Run   : python -I paper/make_figures.py   (needs numpy + matplotlib; not in the project venv.
        Versions used: matplotlib 3.11.2, numpy 2.5.3, Python 3.14.4, in a throwaway venv.)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

MONS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen3-0.6b", "qwen3-1.7b"]
LABEL = {"qwen2.5-0.5b": "Q2.5-0.5B", "qwen2.5-1.5b": "Q2.5-1.5B", "qwen3-0.6b": "Q3-0.6B", "qwen3-1.7b": "Q3-1.7B"}
UNIT = "skeleton"  # pre-registered primary bootstrap unit (12 clusters)

an = json.load(open(os.path.join(RES, "analysis_test.json")))
assert an["split"] == "test" and an["prompt"] == "v2", (an["split"], an["prompt"])
pm = an["per_monitor"]
for m in MONS:
    assert pm[m]["complete"], m

plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "pdf.fonttype": 42})

# ---------------------------------------------------------------- Figure 1: AUROC by contrast/view
CONTRASTS = [  # (key in boot dict, label, marker, color)
    ("P1_BvsA_cot", "B vs A, CoT-only", "o", "#1b9e77"),
    ("BncvsA_cot", "Bnc vs A, CoT-only", "s", "#d95f02"),
    ("P2_CvsA_both", "C vs A, CoT+action", "^", "#7570b3"),
    ("P2_CvsA_action", "C(=D) vs A, action-only", "v", "#e7298a"),
    ("P3_DvsA_both", "D vs A, CoT+action", "D", "#666666"),
]
fig, ax = plt.subplots(figsize=(3.45, 2.35))
nc = len(CONTRASTS)
width = 0.78
for i, m in enumerate(MONS):
    for j, (k, lab, mk, col) in enumerate(CONTRASTS):
        d = pm[m]["boot"][UNIT][k]
        x = i + (j - (nc - 1) / 2) * width / nc
        e, (lo, hi) = d["est"], d["ci"]
        ax.errorbar([x], [e], yerr=[[e - lo], [hi - e]], fmt=mk, color=col, ms=3.2, lw=0.8, capsize=1.5,
                    label=lab if i == 0 else None)
bow = an["bow"]["B"]["skeleton"]["auroc"]
ax.axhline(bow, color="#1b9e77", ls=":", lw=0.8)
ax.text(0.5, bow + 0.012, "BoW, B vs A", color="#1b9e77", fontsize=6, ha="center", va="bottom")
ax.axhline(0.5, color="black", lw=0.5, ls="--")
ax.axhline(0.95, color="#999999", lw=0.5, ls="-.")
ax.text(-0.45, 0.957, "H2 ceiling 0.95", color="#777777", fontsize=6, va="bottom")
ax.set_xticks(range(len(MONS)))
ax.set_xticklabels([LABEL[m] for m in MONS])
ax.set_xlim(-0.5, len(MONS) - 0.5)
ax.set_ylim(0.35, 1.02)
ax.set_ylabel("AUROC (95% CI)")
ax.legend(ncol=2, fontsize=5.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
          handletextpad=0.3, columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_auroc.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: K_cot and O
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.55), sharey=True)
for ax, (k, title, ref) in zip(axes, [("K_cot", "K (CoT-only)", 0.15), ("O", "O = action - CoT+action", None)]):
    for i, m in enumerate(MONS):
        d = pm[m]["boot"][UNIT][k]
        e, (lo, hi) = d["est"], d["ci"]
        ax.errorbar([e], [i], xerr=[[e - lo], [hi - e]], fmt="o", color="#333333", ms=3, lw=0.8, capsize=1.5)
    ax.axvline(0, color="black", lw=0.5, ls="--")
    if ref is not None:
        ax.axvline(ref, color="#999999", lw=0.5, ls="-.")
        ax.text(ref + 0.01, 1.5, "0.15\nfloor", fontsize=5.5, color="#777777", va="center")
    ax.set_title(title, fontsize=7)
    ax.set_yticks(range(len(MONS)))
    ax.set_yticklabels([LABEL[m] for m in MONS])
axes[0].invert_yaxis()  # shared y axis: inverts both panels
axes[0].set_xlabel("AUROC difference")
axes[1].set_xlabel("AUROC difference")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_effects.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
