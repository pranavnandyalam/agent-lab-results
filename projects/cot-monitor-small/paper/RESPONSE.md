# Response to referee round 1: cot-monitor-small

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

The referee report itself is not stored in the repository. Hole numbers follow the Lead's summary and
`../RESULTS.md` ("Addendum: referee round 1 follow-up"). All new numbers come from `../results/posthoc_test.md`
(post-hoc, exploratory, point estimates without CIs).

## Hole 1: the cue-reliance contrast (Qwen2.5 relies on cue words far more than Qwen3) may be prompt-dependent
**Status: partly confirmed; addressed by new analysis.** Note: our first revision overstated this, saying the
contrast "does not hold" / "disappears" under the other prompts. That was wrong; the direction is consistent and
only the gap size is prompt-dependent. This revision corrects the wording.
- Test split re-scored under the two unselected prompts v1 and v3 for Qwen2.5-0.5B and Qwen3-0.6B
  (`src/run_test_prompts.sh`, `src/posthoc.py`).
- K (CoT-only), 0.5B / 0.6B: v2 +0.288 / +0.052; v1 +0.177 / +0.139; v3 +0.083 / -0.018. Qwen2.5-0.5B has the
  larger K under all three prompts; the gap is 0.236 (v2), 0.038 (v1), 0.101 (v3). Under v1 the 0.5B monitor's
  B-vs-A AUROC is only 0.512, so its K there is measured near chance. Point estimates, no CIs, about 12
  skeleton clusters, two smallest monitors only: the family difference is not established.
- Removed from the paper: a dev-split K comparison (Qwen3-0.6B vs Qwen2.5-1.5B under v1) that compared a
  different monitor pair and was selectively chosen.
- Paper changes: new Table VII (K per prompt) in new Section V-A; the abstract, contribution 3, the H3
  paragraph, the Discussion and the Conclusion now say the direction of the family contrast is the
  same under all three prompts, the gap size is prompt-dependent (0.236, 0.038, 0.101), and the family
  difference is not established. Contribution 4 now lists the post-hoc checks.

## Hole 2: pooled action-only AUROC hides heterogeneity across task families
**Status: confirmed and fixed by new analysis.**
- New Table VIII: per-family action-only AUROC (20 vs 20 passes) for all four monitors under v2 and the two small
  monitors under v1/v3. Range 0.05 (Qwen3-0.6B checksum, v2) to 1.00 (Qwen3-1.7B fix_test, v2), while pooled is
  0.476-0.668. Cells cross 0.5 across prompts (Qwen3-0.6B speedup 0.54 v2 -> 0.26 v1; validator 0.20 -> 0.70).
- Wording changed: "hack actions are largely not recognized" now applies explicitly to pooled AUROC only
  (Results, Section V-A, Conclusion, abstract).
- H4 wording: the supporting monitor (Qwen2.5-0.5B) is now described as below chance in *pooled* action-only
  AUROC (per-family 0.19-0.74 under v2); we state that O was not computed per family and that the pooled O may
  average over opposite per-family effects. The Qwen3-1.7B negative O is labeled pooled.

## Hole 5: intent-equivalence of the Bnc (cue-free) rewrites is unchecked
**Status: acknowledged, not done.** Stated in Discussion ("Limits of the post-hoc checks"): if a synonym swap
weakens the stated intent, K overstates cue reliance. Listed as a next step in the Conclusion.

## Hole 8: cue stems in the hack actions
**Status: acknowledged, not done.** Stated in Discussion: we did not control whether hack actions contain stems
from the fixed cue list (e.g., the fix_test skip action `@pytest.mark.skip` / "1 skipped" in `src/templates.py`),
so the action view may partly measure lexical cues. No analysis was run.

## Other changes
- Additional limitations stated explicitly: only the two smallest monitors re-scored under v1/v3; all post-hoc
  numbers are point estimates with no CIs; effective n is still about 12 skeletons; the bag-of-words confound
  applies equally to the per-prompt K values.
- Scale-free false-alarm AUROC (Acue vs A, CoT-only) 0.37-0.61 across all monitors and prompts run, reported in
  Section V-A.
- Two related works added after verifying title, authors and year via the arXiv API: arXiv:2511.08525
  (Yang et al., "Investigating CoT Monitorability in Large Reasoning Models", 2025) and arXiv:2601.05752
  (Yang et al., "AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor", ACL 2026 Findings
  per the arXiv comment).
- Reproducibility statement now includes the post-hoc commands.

## Not done
Re-scoring Qwen2.5-1.5B and Qwen3-1.7B under v1/v3; CIs for post-hoc numbers; per-family O; holes 5 and 8.
