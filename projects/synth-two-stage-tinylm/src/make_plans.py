"""Family A plans (PLAN arm table) for N in {1M, 4M} (N=2M dropped, PLAN cut 1). Writes plans/A_<arm>_N<N>.json
and plans/A_slices.json. Token counts are exact multiples of ctx (256) windows: 1M = 1048576 = 4096 windows.

Slices (only R_train and S1f; never R_dev/R_val/R_gen):
  REAL  = R_train[0 : N]            real tokens of real-only, mixed, S->R, R->S, and the TAIL of mixed->real-tail
  SYN   = S1f[0 : N]                synthetic tokens of every arm that uses them
  REAL2 = R_train[off2 : off2 + N]  real tokens of the mixed PHASE of mixed->real-tail, off2 = min(N, len-1-N)
          N=1M: off2 = N, fully disjoint from REAL. N=4M: R_train (8376976 tok) < 2N+1, so off2 = len-1-N and
          REAL2 overlaps REAL by N-off2 tokens (logged in A_slices.json); those tokens are seen twice in that arm.
Nested budgets: the 1M slices are prefixes of the 4M slices."""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = 256
NS = {"1M": 1 << 20, "4M": 1 << 22}
R, S = "data/R_train.bin", "data/S1f.bin"

def ntok(p):
    return len(np.memmap(os.path.join(ROOT, p), dtype=np.uint16, mode="r"))

def src(b, off, n):
    return {"bin": b, "offset": int(off), "tokens": int(n)}

def main():
    lr_, ls_ = ntok(R), ntok(S)
    os.makedirs(os.path.join(ROOT, "plans"), exist_ok=True)
    info = {"ctx": CTX, "R_train_tokens": lr_, "S1f_tokens": ls_, "N": {}}
    for tag, N in NS.items():
        assert N % CTX == 0 and N + 1 <= lr_ and N + 1 <= ls_
        off2 = min(N, lr_ - 1 - N)
        real, syn, real2 = src(R, 0, N), src(S, 0, N), src(R, off2, N)
        arms = {
            "real-only": [{"mode": "concat", "sources": [real]}],
            "mixed": [{"mode": "interleave", "sources": [real, syn]}],
            "S-R": [{"mode": "concat", "sources": [syn]}, {"mode": "concat", "sources": [real]}],
            "R-S": [{"mode": "concat", "sources": [real]}, {"mode": "concat", "sources": [syn]}],
            "mixed-realtail": [{"mode": "interleave", "sources": [real2, syn]}, {"mode": "concat", "sources": [real]}],
        }
        for arm, ph in arms.items():
            json.dump({"phases": ph}, open(os.path.join(ROOT, f"plans/A_{arm}_N{tag}.json"), "w"))
        info["N"][tag] = {"N": N, "windows": N // CTX, "REAL": [0, N], "SYN": [0, N], "REAL2": [off2, off2 + N],
                          "REAL2_overlap_with_REAL_tokens": max(0, N - off2),
                          "total_tokens": {"real-only": N, "mixed": 2 * N, "S-R": 2 * N, "R-S": 2 * N,
                                           "mixed-realtail": 3 * N}}
    json.dump(info, open(os.path.join(ROOT, "plans/A_slices.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))

if __name__ == "__main__":
    main()
