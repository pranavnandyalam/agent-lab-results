"""Generate paper figures from committed raw results (no hand-drawn data).

Inputs : ../results/analysis.json (written by src/analyze.py), ../results/raw_<model>_r<k>.json
Outputs: figures/fig_taxonomy.pdf, figures/fig_letter_share.pdf, figures/paper_stats.json
         (letter shares per criterion and summed per-pass forward times, quoted in main.tex)
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
ax.set_ylabel('share of passes "A"')
ax.legend(ncol=4, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
          handlelength=1.0, columnspacing=0.8)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig_letter_share.pdf"))
plt.close(fig)

stats = {"letter_share_A": {m: {c: {"share_A": round(v[0], 3), "n_passes": v[1]} for c, v in d.items()}
                            for m, d in share.items()},
         "forward_seconds_sum": {m: round(v) for m, v in fwd_sec.items()},
         "forward_hours_total": round(sum(fwd_sec.values()) / 3600, 2)}
with open(os.path.join(OUT, "paper_stats.json"), "w") as f:
    json.dump(stats, f, indent=1)
print(json.dumps({SHORT[m]: {c: [round(v[0], 3), v[1]] for c, v in d.items()} for m, d in share.items()}))
