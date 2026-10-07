"""Join committed blind audit labels with the (scratch) key; report precision/recall of v1/v2 regexes per stratum."""
import json, os, collections
d = "results/qwen3-0.6b/"
lab = json.load(open(d + "audit_labels.json"))["labels"]
key = json.load(open(d + "audit_key.json"))["key"]
out = ["# Audit join (qwen3-0.6b)", "", "Labels committed before this join (see git history).", ""]
strata = collections.defaultdict(list)
for k in key:
    strata[k["stratum"]].append((lab[k["audit_id"]] == "Y", k["v1_think"], k["v2_think"], k["answer"], k["shown_letter"]))
out.append("| stratum | n | human Y | v1 fires | v2 fires |"); out.append("|---|---|---|---|---|")
for s, r in sorted(strata.items()):
    out.append(f"| {s} | {len(r)} | {sum(x[0] for x in r)} | {sum(bool(x[1]) for x in r)} | {sum(bool(x[2]) for x in r)} |")
for name, idx in (("v1", 1), ("v2", 2)):
    allr = [x for r in strata.values() for x in r]
    tp = sum(x[0] and x[idx] for x in allr); fp = sum((not x[0]) and x[idx] for x in allr); fn = sum(x[0] and not x[idx] for x in allr)
    out.append(f"\n{name}: TP={tp} FP={fp} FN={fn} precision={tp/max(tp+fp,1):.2f} recall={tp/max(tp+fn,1):.2f} (n=50, all strata)")
    cue = [x for s, r in strata.items() if s.startswith("cue") for x in r]
    tp = sum(x[0] and x[idx] for x in cue); fp = sum((not x[0]) and x[idx] for x in cue); fn = sum(x[0] and not x[idx] for x in cue)
    out.append(f"{name} cue strata only: TP={tp} FP={fp} FN={fn} precision={tp/max(tp+fp,1):.2f} recall={tp/max(tp+fn,1):.2f}")
cue = [x for s, r in strata.items() if s.startswith("cue") for x in r]
out.append(f"\nHuman-Y rate in cue strata: {sum(x[0] for x in cue)}/{len(cue)}; in neutral strata: {sum(x[0] for s,r in strata.items() if s.startswith('neutral') for x in r)}/10")
for s in ("cue_user", "cue_tool"):
    r = strata[s]; out.append(f"{s}: human-Y among traces whose answer==cue letter: {sum(x[0] for x in r if x[3]==x[4])}/{sum(x[3]==x[4] for x in r)}")
open(d + "audit_join.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
