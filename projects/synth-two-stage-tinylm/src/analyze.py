"""Family A analysis: per-arm R_val ppl (mean±std over seeds), paired diffs vs pairs, document bootstrap.
Pre-registered rule: sign-agreement across all seeds AND doc-bootstrap 95% CI (resampling docs, shared across seeds) excludes 0.
Usage: .venv/bin/python -I src/analyze.py > results/analysis.md   (R_dev is NOT used.)"""
import glob, os, numpy as np
G = os.path.join(os.path.dirname(__file__), "..", "results", "grid")
ARMS = ["real-only", "mixed", "S-R", "R-S", "mixed-realtail"]
def load(arm, N, s):
    p = f"{G}/{arm}_N{N}_s{s}/eval_R_val_perdoc.npz"
    if not os.path.exists(p): return None
    z = np.load(p); return z["nll_sum"], z["count"]
def ppl(nll, cnt): return float(np.exp(nll.sum() / cnt.sum()))
rng = np.random.default_rng(0)
def paired(a_arm, b_arm, N, seeds):
    """diff = ppl(a) - ppl(b), per seed and bootstrap CI over docs (same doc indices for all seeds)."""
    A = [load(a_arm, N, s) for s in seeds]; B = [load(b_arm, N, s) for s in seeds]
    ok = [i for i in range(len(seeds)) if A[i] is not None and B[i] is not None]
    if not ok: return None
    A = [A[i] for i in ok]; B = [B[i] for i in ok]
    cnt = A[0][1]; nd = len(cnt)
    per = [ppl(*a) - ppl(*b) for a, b in zip(A, B)]
    boots = []
    for _ in range(2000):
        ix = rng.integers(0, nd, nd)
        boots.append(np.mean([np.exp(a[0][ix].sum()/a[1][ix].sum()) - np.exp(b[0][ix].sum()/b[1][ix].sum()) for a, b in zip(A, B)]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return per, float(np.mean(per)), float(lo), float(hi), len(ok)
print("# Family A analysis (R_val perplexity; lower is better; PRELIMINARY unless n=3)\n")
print("| N | arm | n seeds | R_val ppl mean ± std | per-seed |\n|---|---|---|---|---|")
for N in ["1M", "4M"]:
    for arm in ARMS:
        v = [ppl(*load(arm, N, s)) for s in range(3) if load(arm, N, s)]
        if v: print(f"| {N} | {arm} | {len(v)} | {np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v)>1 else float('nan'):.2f} | {', '.join(f'{x:.2f}' for x in v)} |")
print("\n## Paired differences (a − b, ppl; negative = a better)\n")
print("| N | a vs b | n | mean diff | doc-bootstrap 95% CI | per-seed diffs | rule met (sign agree & CI excl. 0) |\n|---|---|---|---|---|---|---|")
for N in ["1M", "4M"]:
    for a, b in [("S-R","mixed"),("S-R","real-only"),("mixed","real-only"),("S-R","R-S"),("S-R","mixed-realtail")]:
        r = paired(a, b, N, range(3))
        if r:
            per, m, lo, hi, n = r
            met = (all(x > 0 for x in per) or all(x < 0 for x in per)) and (lo > 0 or hi < 0) and n == 3
            print(f"| {N} | {a} − {b} | {n} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {', '.join(f'{x:+.2f}' for x in per)} | {met} |")
