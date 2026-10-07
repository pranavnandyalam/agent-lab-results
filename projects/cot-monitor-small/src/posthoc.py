"""Referee-round-1 post-hoc breakdowns (exploratory, not pre-registered). numpy only.
Usage: python -I src/posthoc.py --split test --prompts v2[,v1,v3]
Writes results/posthoc_<split>.md. Reports, per prompt and monitor:
  K = AUROC_cot(B vs A) - AUROC_cot(Bnc vs A); scale-free false alarm AUROC_cot(Acue vs A);
  action-view AUROC(a1 vs a0) pooled, per family, per hack type (instance index parity = hack variant);
  within-pair accuracy (fraction of instances with score(a1) > score(a0))."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze import auroc, load_scores, load_instances, ROOT, MONS

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); ap.add_argument("--prompts", default="v2")
a = ap.parse_args()
insts = load_instances(a.split)
fam = {i["inst"]: i["family"] for i in insts}
ids = sorted(fam)
fams = sorted(set(fam.values()))
par = {i: int(i.rsplit("-", 1)[-1]) % 2 for i in ids}
out = [f"# Post-hoc breakdowns ({a.split}) — exploratory\n"]
for pr in a.prompts.split(","):
    S = load_scores(a.split, pr)
    out.append(f"\n## prompt {pr}\n")
    out.append("| monitor | n inst | B-A cot | Bnc-A cot | K | Acue-A cot (AUROC) | act a1-vs-a0 | pair acc |\n|---|---|---|---|---|---|---|---|")
    det = []
    for m in MONS:
        if m not in S:
            continue
        s = S[m]
        g = lambda v, c: [s[(i, v, c)]["score"] for i in ids if (i, v, c) in s]
        if len(g("cot", "A")) < len(ids):
            continue
        A, B, Bn, Ac = g("cot", "A"), g("cot", "B"), g("cot", "Bnc"), g("cot", "Acue")
        a0, a1 = [s[(i, "action", "a0")]["score"] for i in ids], [s[(i, "action", "a1")]["score"] for i in ids]
        ab, abn = auroc(B, A), auroc(Bn, A)
        out.append(f"| {m} | {len(ids)} | {ab:.3f} | {abn:.3f} | {ab-abn:+.3f} | {auroc(Ac, A):.3f} | {auroc(a1, a0):.3f} | {np.mean(np.array(a1) > np.array(a0)):.3f} |")
        row = [m]
        for f in fams:
            ix = [k for k, i in enumerate(ids) if fam[i] == f]
            row.append(f"{auroc([a1[k] for k in ix], [a0[k] for k in ix]):.2f}")
        for h in (0, 1):
            ix = [k for k, i in enumerate(ids) if par[i] == h]
            row.append(f"{auroc([a1[k] for k in ix], [a0[k] for k in ix]):.2f}")
        det.append("| " + " | ".join(row) + " |")
    out.append("\nAction view AUROC(a1 vs a0), within family / hack-variant parity (a0 only from same group):\n")
    out.append("| monitor | " + " | ".join(fams) + " | hack var 0 | hack var 1 |\n|" + "---|" * (len(fams) + 3))
    out += det
open(os.path.join(ROOT, "results", f"posthoc_{a.split}.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
