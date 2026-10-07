"""Unit check for train.py --lr_const: warmup then flat; default schedule and run_hash unchanged."""
import json, sys, types, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import train as T
plan = json.load(open(os.path.join(ROOT, "plans/A_S-R_N4M.json")))
pk = 3e-3
for s in [0, 25, 51, 52, 512, 1023]:
    print(s, "const", round(T.lr_sched(s, 1024, pk, None, True), 7), "global", round(T.lr_sched(s, 1024, pk), 7))
assert all(T.lr_sched(i, 1024, pk, None, True) == T.lr_at(i, 1024, pk) for i in range(51))  # same warmup (51 steps)
assert all(T.lr_sched(i, 1024, pk, None, True) == pk for i in range(51, 1024))
r = json.load(open(os.path.join(ROOT, "results/grid/S-R_N4M_s0/result.json")))
a = types.SimpleNamespace(**r["args"]); a.lr_schedule = "global"
h = T.run_hash(plan, a); print("default hash matches committed S-R_N4M_s0:", h == r["run_hash"]); assert h == r["run_hash"]
a.lr_const = False; assert T.run_hash(plan, a) == r["run_hash"]
a.lr_const = True; print("lr_const hash differs:", T.run_hash(plan, a) != r["run_hash"]); assert T.run_hash(plan, a) != r["run_hash"]
print("OK")
