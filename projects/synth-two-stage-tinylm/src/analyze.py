"""Family A analysis: per-arm R_val ppl (mean±std over seeds), paired diffs vs pairs, document bootstrap.
Pre-registered rule: sign-agreement across all seeds AND hierarchical-bootstrap 95% CI (hierarchical: seeds resampled, then docs) excludes 0.
Usage: .venv/bin/python -I src/analyze.py > results/analysis.md   (R_dev is NOT used.)"""
import glob, os, numpy as np
G = os.path.join(os.path.dirname(__file__), "..", "results", "grid")
ARMS = ["real-only", "real-only-2N", "real-only-2ep", "mixed", "S-R", "R-S", "mixed-realtail"]
def load(arm, N, s):
    p = f"{G}/{arm}_N{N}_s{s}/eval_R_val_perdoc.npz"
    if not os.path.exists(p): return None
    z = np.load(p); return z["nll_sum"], z["count"]
def ppl(nll, cnt): return float(np.exp(nll.sum() / cnt.sum()))
rng = np.random.default_rng(0)
def _p(z, ix): return np.exp(z[0][ix].sum() / z[1][ix].sum())
def hier(diff_fn, nseeds, nd, reps=2000):
    """Hierarchical bootstrap: resample seeds with replacement, then docs (fresh doc draw per rep, shared across drawn seeds)."""
    rng = np.random.default_rng(0)  # fixed per comparison: intervals do not shift when rows are added
    out = []
    for _ in range(reps):
        sx = rng.integers(0, nseeds, nseeds); ix = rng.integers(0, nd, nd)
        out.append(np.mean([diff_fn(s, ix) for s in sx]))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]
def paired(a_arm, b_arm, N, seeds):
    """diff = ppl(a) - ppl(b) per seed; CI by hierarchical (seeds, then docs) bootstrap. n=3 seeds => seed level is coarse."""
    A = [load(a_arm, N, s) for s in seeds]; B = [load(b_arm, N, s) for s in seeds]
    ok = [i for i in range(len(seeds)) if A[i] is not None and B[i] is not None]
    if not ok: return None
    A = [A[i] for i in ok]; B = [B[i] for i in ok]; nd = len(A[0][1])
    per = [ppl(*a) - ppl(*b) for a, b in zip(A, B)]
    lo, hi = hier(lambda s, ix: _p(A[s], ix) - _p(B[s], ix), len(ok), nd)
    return per, float(np.mean(per)), lo, hi, len(ok)
def h1_contrast(a_arm="S-R", b_arm="mixed"):
    """g(N)=ppl(a)-ppl(b); contrast g(4M)-g(1M), seeds paired. Negative = ordering benefit grows with N."""
    seeds = [s for s in range(3) if all(load(x, N, s) for x in (a_arm, b_arm) for N in ("1M", "4M"))]
    if not seeds: return None
    D = [{N: (load(a_arm, N, s), load(b_arm, N, s)) for N in ("1M", "4M")} for s in seeds]
    per = [ppl(*d["4M"][0]) - ppl(*d["4M"][1]) - (ppl(*d["1M"][0]) - ppl(*d["1M"][1])) for d in D]
    nd = len(D[0]["4M"][0][1])
    lo, hi = hier(lambda s, ix: _p(D[s]["4M"][0], ix) - _p(D[s]["4M"][1], ix) - _p(D[s]["1M"][0], ix) + _p(D[s]["1M"][1], ix), len(seeds), nd)
    return per, float(np.mean(per)), lo, hi, len(seeds)
def main():
    print("# Family A analysis (R_val perplexity; lower is better; PRELIMINARY unless n=3)\n")
    print("| N | arm | n seeds | R_val ppl mean ± std | per-seed |\n|---|---|---|---|---|")
    for N in ["1M", "4M"]:
        for arm in ARMS:
            v = [ppl(*load(arm, N, s)) for s in range(3) if load(arm, N, s)]
            if v: print(f"| {N} | {arm} | {len(v)} | {np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v)>1 else float('nan'):.2f} | {', '.join(f'{x:.2f}' for x in v)} |")
    print("\n## Paired differences (a − b, ppl; negative = a better)\n")
    print("| N | a vs b | n | mean diff | hierarchical-bootstrap 95% CI | per-seed diffs | rule met (sign agree & CI excl. 0) |\n|---|---|---|---|---|---|---|")
    for N in ["1M", "4M"]:
        for a, b in [("S-R","mixed"),("S-R","real-only"),("mixed","real-only"),("S-R","R-S"),("S-R","mixed-realtail"),("mixed","real-only-2N"),("S-R","real-only-2N"),("mixed","real-only-2ep"),("S-R","real-only-2ep"),("real-only-2ep","real-only"),("real-only-2ep","real-only-2N")]:
            r = paired(a, b, N, range(3))
            if r:
                per, m, lo, hi, n = r
                met = (all(x > 0 for x in per) or all(x < 0 for x in per)) and (lo > 0 or hi < 0) and n == 3
                print(f"| {N} | {a} − {b} | {n} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {', '.join(f'{x:+.2f}' for x in per)} | {met} |")

    print("\n## H1 contrast: g(4M) - g(1M), g = ppl(S-R) - ppl(mixed) (negative = S-R advantage grows with N)\n")
    r = h1_contrast()
    if r:
        per, m, lo, hi, n = r
        print(f"n seeds = {n}; mean {m:+.2f}; hierarchical-bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]; per-seed {', '.join(f'{x:+.2f}' for x in per)}")
    print("\nNote: with 3 seeds the seed-level bootstrap is coarse; per-seed signs are the primary evidence. Rows with n<3 are preliminary and never count as 'rule met'.")
    print("real-only-2ep = Family B: the same N real tokens, 2 epochs (matched total tokens/steps, no extra fresh real data).")
    print("real-only-2N = real-only with matched TOTAL tokens (2N; 4M capped at R_train length 8.38M) to separate step-count from synthetic-data effects.")

if __name__ == "__main__":  # importable (helpers reused by src/analyze_r2.py)
    main()
