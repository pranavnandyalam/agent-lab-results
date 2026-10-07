"""Builds results/referee_r1_ablation.md table from results/grid/*/result.json (N=4M). Std = sample std (ddof=1)."""
import json, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = os.path.join(ROOT, "results/grid")
ROWS = [("S-R", "S->R, global schedule (lr 3e-3)"), ("S-R-perphase", "S->R, per-phase warmup+cosine restart (lr 3e-3)"),
        ("mixed", "mixed, global (lr 3e-3)"), ("real-only-2ep", "real-only-2ep, global (lr 3e-3)"),
        ("mixed-lr1e-3", "mixed, global, lr 1e-3"), ("S-R-lr1e-3", "S->R, global, lr 1e-3"),
        ("real-only-2ep-lr1e-3", "real-only-2ep, global, lr 1e-3")]
out = ["| arm | description | seeds done | R_val ppl per seed | R_val mean ± std | R_dev mean |", "|---|---|---|---|---|---|"]
raw = {}
for arm, desc in ROWS:
    v, d, seeds = [], [], []
    for s in (0, 1, 2):
        f = os.path.join(G, f"{arm}_N4M_s{s}", "result.json")
        if os.path.exists(f):
            r = json.load(open(f)); v.append(r["eval"]["R_val"]["ppl"]); d.append(r["eval"]["R_dev"]["ppl"]); seeds.append(s)
    raw[arm] = dict(zip(seeds, v))
    if not v:
        out.append(f"| {arm} | {desc} | none | - | - | - |"); continue
    sd = f"{st.stdev(v):.2f}" if len(v) > 1 else "n/a"
    out.append(f"| {arm} | {desc} | {','.join(map(str, seeds))} | " + " / ".join(f"s{s}: {x:.3f}" for s, x in zip(seeds, v))
               + f" | {st.mean(v):.2f} ± {sd} | {st.mean(d):.2f} |")
print("\n".join(out)); json.dump(raw, open(os.path.join(ROOT, "results/referee_r1/r_val_ppl_raw.json"), "w"), indent=1)
