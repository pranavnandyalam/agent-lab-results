"""Referee round-2 analysis: per-arm R_val ppl (mean ± sample std, n seeds) and paired differences with
hierarchical (seeds, then docs) bootstrap 95% CIs, reusing the helpers of src/analyze.py (ppl, _p, hier; rng seed 0,
2000 reps). Runs are looked up first in results/referee_r2/<run>/, then results/grid/<run>/ (run = <arm>_N<N>_s<seed>).
Missing runs are skipped; seeds are paired by intersection. Rows with n<3 are preliminary (never 'rule met').
R_dev is NOT used. Usage: .venv/bin/python -I src/analyze_r2.py > results/referee_r2/analysis.md"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ppl, _p, hier  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DIRS = [os.path.join(RES, "referee_r2"), os.path.join(RES, "grid")]
SEEDS = range(3)

def load(arm, N, s):
    for d in DIRS:
        p = f"{d}/{arm}_N{N}_s{s}/eval_R_val_perdoc.npz"
        if os.path.exists(p):
            z = np.load(p); return z["nll_sum"], z["count"]
    return None

def where(arm, N, s):
    for d in DIRS:
        if os.path.exists(f"{d}/{arm}_N{N}_s{s}/eval_R_val_perdoc.npz"):
            return os.path.basename(d)
    return None

def contrast(terms, N):
    """terms: list of (coef, arm). Per seed: sum coef*ppl(arm). CI: hierarchical bootstrap over seeds then docs."""
    seeds = [s for s in SEEDS if all(load(a, N, s) is not None for _, a in terms)]
    if not seeds: return None
    D = [[(c, load(a, N, s)) for c, a in terms] for s in seeds]
    per = [sum(c * ppl(*z) for c, z in d) for d in D]
    nd = len(D[0][0][1][1])
    lo, hi = hier(lambda k, ix: sum(c * _p(z, ix) for c, z in D[k]), len(seeds), nd)
    return per, float(np.mean(per)), lo, hi, seeds

ARMS = {  # N -> arms shown in the per-arm table (comparators from results/grid included for context)
    "4M": ["mixed", "S-R", "real-only-2ep", "S-R-perphase", "mixed-constLR", "S-R-constLR",
           "mixed-lr1e-3", "S-R-lr1e-3", "real-only-2ep-lr1e-3", "mixed-lr6e-3", "S-R-lr6e-3", "real-only-2ep-lr6e-3"],
    "1M": ["real-only", "real-only-2ep", "mixed", "S-R", "S-R-perphase"],
}
D = lambda a, b: [(1, a), (-1, b)]
COMPS = [  # (section, N, label, terms)
    ("A1 ordering vs annealing (constant LR = 5% warmup then flat 3e-3)", "4M", "S-R-constLR − mixed-constLR", D("S-R-constLR", "mixed-constLR")),
    ("A1 ordering vs annealing (constant LR = 5% warmup then flat 3e-3)", "4M", "S-R − mixed (cosine, reference)", D("S-R", "mixed")),
    ("A1 ordering vs annealing (constant LR = 5% warmup then flat 3e-3)", "4M", "[S-R-constLR − mixed-constLR] − [S-R − mixed] (does the ordering gap depend on annealing?)",
     [(1, "S-R-constLR"), (-1, "mixed-constLR"), (-1, "S-R"), (1, "mixed")]),
    ("A1 ordering vs annealing (constant LR = 5% warmup then flat 3e-3)", "4M", "S-R-constLR − S-R (effect of removing decay, S->R)", D("S-R-constLR", "S-R")),
    ("A1 ordering vs annealing (constant LR = 5% warmup then flat 3e-3)", "4M", "mixed-constLR − mixed (effect of removing decay, mixed)", D("mixed-constLR", "mixed")),
    ("A1 ordering vs annealing (constant LR = 5% warmup then flat 3e-3)", "4M", "S-R-constLR − real-only-2ep (cosine)", D("S-R-constLR", "real-only-2ep")),
    ("Schedule restart vs global schedule (S->R, lr 3e-3)", "4M", "S-R-perphase − S-R", D("S-R-perphase", "S-R")),
    ("Schedule restart vs global schedule (S->R, lr 3e-3)", "4M", "S-R-perphase − mixed", D("S-R-perphase", "mixed")),
    ("Schedule restart vs global schedule (S->R, lr 3e-3)", "4M", "S-R-perphase − real-only-2ep", D("S-R-perphase", "real-only-2ep")),
    ("LR sensitivity, lr 1e-3 (seed 0 from results/grid, seeds 1,2 from referee_r2)", "4M", "S-R-lr1e-3 − mixed-lr1e-3", D("S-R-lr1e-3", "mixed-lr1e-3")),
    ("LR sensitivity, lr 1e-3 (seed 0 from results/grid, seeds 1,2 from referee_r2)", "4M", "S-R-lr1e-3 − real-only-2ep-lr1e-3", D("S-R-lr1e-3", "real-only-2ep-lr1e-3")),
    ("LR sensitivity, lr 1e-3 (seed 0 from results/grid, seeds 1,2 from referee_r2)", "4M", "mixed-lr1e-3 − real-only-2ep-lr1e-3", D("mixed-lr1e-3", "real-only-2ep-lr1e-3")),
    ("LR sensitivity, lr 6e-3", "4M", "S-R-lr6e-3 − mixed-lr6e-3", D("S-R-lr6e-3", "mixed-lr6e-3")),
    ("LR sensitivity, lr 6e-3", "4M", "S-R-lr6e-3 − real-only-2ep-lr6e-3", D("S-R-lr6e-3", "real-only-2ep-lr6e-3")),
    ("LR sensitivity, lr 6e-3", "4M", "mixed-lr6e-3 − real-only-2ep-lr6e-3", D("mixed-lr6e-3", "real-only-2ep-lr6e-3")),
    ("LR sensitivity, lr 6e-3", "4M", "S-R-lr6e-3 − S-R (3e-3)", D("S-R-lr6e-3", "S-R")),
    ("LR sensitivity, lr 6e-3", "4M", "mixed-lr6e-3 − mixed (3e-3)", D("mixed-lr6e-3", "mixed")),
    ("LR sensitivity, lr 6e-3", "4M", "real-only-2ep-lr6e-3 − real-only-2ep (3e-3)", D("real-only-2ep-lr6e-3", "real-only-2ep")),
    ("N=1M per-phase S->R (H2-primary: real stage uses exactly the real-only(N) schedule)", "1M", "S-R-perphase − real-only (H2-primary)", D("S-R-perphase", "real-only")),
    ("N=1M per-phase S->R (H2-primary: real stage uses exactly the real-only(N) schedule)", "1M", "S-R-perphase − S-R", D("S-R-perphase", "S-R")),
    ("N=1M per-phase S->R (H2-primary: real stage uses exactly the real-only(N) schedule)", "1M", "S-R-perphase − mixed", D("S-R-perphase", "mixed")),
    ("N=1M per-phase S->R (H2-primary: real stage uses exactly the real-only(N) schedule)", "1M", "S-R-perphase − real-only-2ep", D("S-R-perphase", "real-only-2ep")),
]

def main():
    print("# Referee round 2 analysis (R_val perplexity; lower is better)\n")
    print("Generated by `src/analyze_r2.py`. CIs: hierarchical bootstrap (resample seeds, then R_val documents; 2000 reps, "
          "rng seed 0; helpers from `src/analyze.py`). 'rule met' = all per-seed diffs share a sign AND CI excludes 0 AND n=3. "
          "With n=1 the CI is a document-only bootstrap and the row is a hint, not a result.\n")
    print("## Per-arm R_val ppl\n")
    print("| N | arm | n seeds | mean ± std (ddof=1) | per-seed (source dir) |\n|---|---|---|---|---|")
    for N, arms in ARMS.items():
        for arm in arms:
            v = [(s, ppl(*load(arm, N, s)), where(arm, N, s)) for s in SEEDS if load(arm, N, s) is not None]
            if not v:
                print(f"| {N} | {arm} | 0 | not run | |"); continue
            x = [p for _, p, _ in v]
            sd = f"{np.std(x, ddof=1):.2f}" if len(x) > 1 else "n/a"
            print(f"| {N} | {arm} | {len(x)} | {np.mean(x):.2f} ± {sd} | " + ", ".join(f"s{s}: {p:.3f} ({w})" for s, p, w in v) + " |")
    sec = None
    for section, N, label, terms in COMPS:
        if section != sec:
            sec = section
            print(f"\n## {section}\n")
            print("| N | contrast | n | seeds | mean | 95% CI | per-seed | rule met |\n|---|---|---|---|---|---|---|---|")
        r = contrast(terms, N)
        if r is None:
            print(f"| {N} | {label} | 0 | | not available | | | |"); continue
        per, m, lo, hi, seeds = r
        met = (all(x > 0 for x in per) or all(x < 0 for x in per)) and (lo > 0 or hi < 0) and len(per) == 3
        print(f"| {N} | {label} | {len(per)} | {','.join(map(str, seeds))} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
              + ", ".join(f"{x:+.3f}" for x in per) + f" | {met} |")
    print("\nNotes: constLR = same 5% linear warmup as the cosine runs, then constant peak LR (3e-3) to the last step "
          "(train.py --lr_const). mixed has one phase, so per-phase = global for mixed. No multiple-comparison correction.")

if __name__ == "__main__":
    main()
