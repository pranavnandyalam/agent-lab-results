"""Build data/instances.jsonl: one record per scoring pass (deterministic, no randomness).

Per instance: action-only (a0, a1) = 2 passes; CoT-only (ben, rev, revnc, bencue) = 4;
CoT+action (A, B, C, D, Bnc, Acue) = 6. Total 12 passes/instance.
Condition labels for action-only / CoT-only passes are the content keys; analysis maps them
to conditions (e.g. action-only B vs A compares a0 with a0 -> AUROC 0.5 by construction).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cues import find_cues  # noqa: E402
from templates import instances, pools  # noqa: E402

COND = {"A": ("ben", "a0"), "B": ("rev", "a0"), "C": ("rev", "a1"), "D": ("ben", "a1"),
        "Bnc": ("revnc", "a0"), "Acue": ("bencue", "a0")}
COT_COND = {"ben": "A", "rev": "B", "revnc": "Bnc", "bencue": "Acue"}


def cap(s: str) -> str:
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)


def main(out):
    P = {"test": pools("test"), "dev": pools("dev")}
    recs, problems = [], []
    for split, fam, j, p, sl in instances():
        iid = f"{split}-{fam}-{j:02d}"
        R = {k: cap(P[split][k][p].format(check=sl["check"], goal=sl["goal"])) for k in P[split]}
        for k, txt in R.items():
            c = find_cues(txt)
            if k in ("ben", "revnc") and c:
                problems.append((iid, k, c, txt))
            if k in ("rev", "bencue") and not c:
                problems.append((iid, k, "no cue", txt))
        for k in ("task", "check", "goal"):
            if find_cues(sl[k]):
                problems.append((iid, k, find_cues(sl[k])))
        base = dict(split=split, family=fam, inst=iid, p=f"{fam}-{split}-p{p:02d}", task=sl["task"])
        for a in ("a0", "a1"):
            recs.append(dict(base, id=f"{iid}|action|{a}", view="action", cond=a, reasoning=None, action=sl[a]))
        for r, c in COT_COND.items():
            recs.append(dict(base, id=f"{iid}|cot|{c}", view="cot", cond=c, reasoning=R[r], action=None))
        for c, (r, a) in COND.items():
            recs.append(dict(base, id=f"{iid}|both|{c}", view="both", cond=c, reasoning=R[r], action=sl[a]))
    if problems:
        for x in problems:
            print("PROBLEM", x)
        raise SystemExit(1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(recs), "records ->", out)
    print(Counter((r["split"], r["view"]) for r in recs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "instances.jsonl"))
