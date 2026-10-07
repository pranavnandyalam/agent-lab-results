"""Unit check for train.py --lr_schedule: per-phase bounds/LR values and run_hash invariance for the default."""
import json, sys, types, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import train as T
plan = json.load(open(os.path.join(ROOT, "plans/A_S-R_N4M.json")))
b = T.phase_bounds(plan, 256, 32); print("bounds", b); assert b == [0, 512, 1024]
pk = 3e-3
for s in [0, 24, 25, 511, 512, 512 + 24, 512 + 25, 1023]:
    print(s, "per_phase", round(T.lr_sched(s, 1024, pk, b), 7), "global", round(T.lr_sched(s, 1024, pk), 7))
assert T.lr_sched(512, 1024, pk, b) == T.lr_sched(0, 1024, pk, b) == T.lr_at(0, 512, pk)
assert all(T.lr_sched(512 + i, 1024, pk, b) == T.lr_sched(i, 1024, pk, b) for i in range(512))
assert all(T.lr_sched(i, 1024, pk) == T.lr_at(i, 1024, pk) for i in range(1024))
assert abs(T.lr_sched(511, 1024, pk, b) - 0.1 * pk) < 1e-4 * pk
r = json.load(open(os.path.join(ROOT, "results/grid/S-R_N4M_s0/result.json")))
a = types.SimpleNamespace(**r["args"]); a.lr_schedule = "global"
h = T.run_hash(plan, a); print("default hash matches committed S-R_N4M_s0:", h == r["run_hash"]); assert h == r["run_hash"]
a.lr_schedule = "per_phase"; print("per_phase hash differs:", T.run_hash(plan, a) != r["run_hash"])
print("OK")
