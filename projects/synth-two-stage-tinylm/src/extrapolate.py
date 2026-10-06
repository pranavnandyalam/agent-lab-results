"""Extrapolate full-grid training time from a measured tokens/sec (PLAN rev 3 arm table)."""
import json, sys
tps = float(sys.argv[1]); M = 1_000_000
Ns, seeds = [1, 2, 4], 3
arm_mult = {"real-only": 1, "mixed": 2, "S->R": 2, "R->S": 2, "mixed->real-tail": 3}  # tokens / N
famA = sum(m * n for n in Ns for m in arm_mult.values()) * seeds * M
famB = sum(2 * n for n in Ns) * seeds * M                     # real-only, 2N total via repeats
restart = sum(2 * n for n in [1, 4]) * seeds * M * 2           # ~12 runs: S->R + real-only-schedule variants
lr_tune = 6 * 2 * M                                           # 6 runs at ~2M tokens on R_dev
g0 = 8.93 * M                                                 # one epoch of R_gen
items = {"famA_45runs": famA, "famB_9runs": famB, "restart_12runs": restart, "lr_tune_6runs": lr_tune, "G0": g0}
out = {k: {"Mtok": v / M, "hours": round(v / tps / 3600, 2)} for k, v in items.items()}
tot = sum(items.values()); out["TOTAL"] = {"Mtok": tot / M, "hours": round(tot / tps / 3600, 2)}
cut1 = {k: v for k, v in items.items()}
cut1["famA_45runs"] = sum(m * n for n in [1, 4] for m in arm_mult.values()) * seeds * M
out["TOTAL_without_N2M_famA"] = {"hours": round(sum(cut1.values()) / tps / 3600, 2)}
out["tok_per_s"] = tps
print(json.dumps(out, indent=1))
