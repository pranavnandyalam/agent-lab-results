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

---

# Round 2: response to referee round 2 (verdict NOT_YET)

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

The round-2 referee report is not stored in the repository; items below follow the Lead's numbered summary of
the required changes. All new numbers come from `../results/posthoc_test_r2.md` (`src/posthoc2.py`, post-hoc,
exploratory, no model scoring) or from earlier committed result files. Overall response: claims were cut to what
the evidence supports and the limits are disclosed; the rebuild the referee asked for was not done this cycle.

## 1. Reframe as a pilot; BoW overlap
**Done.** Title, abstract, introduction, Discussion and Conclusion now call the study a pilot on 12 hand-written
synthetic skeletons by a single author-agent, state that nothing was validated on real trajectories, and make
no claim about real monitors. BoW (skeleton-grouped CV) 0.758 [0.683, 0.894] is stated to overlap the CIs of
Qwen2.5-0.5B [0.690, 0.905] and also Qwen2.5-1.5B [0.869, 0.973] (the latter was not in the request but follows
from the same numbers); only the Qwen3 intervals lie above it.

## 2. Cue-free prompt v3 as primary cue-priming control; K with CIs and sign-flip p
**Done.** Method lists the three prompts and their cue-stem counts (v2: 5, v1: 1 `game`, v3: 0; checked against
`src/cues.py`). New Table VI gives B-vs-A, Bnc-vs-A, K with skeleton-bootstrap CIs, exact cluster sign-flip
p1/p2 (4096 assignments; described as the few-cluster-valid check, bootstrap over 12 clusters may under-cover),
and Acue-vs-A AUROC, for all 8 monitor-prompt rows. Text: v3 Qwen3-0.6B K = -0.018 [-0.058, 0.027], p2 = 0.4482
(no cue effect); v3 Qwen2.5-0.5B K = +0.083 [0.048, 0.119] with B-vs-A 0.552 (gap between two near-chance
AUROCs); v1 Qwen2.5-0.5B B-vs-A 0.512 likewise. "Direction holds across all three prompts" is removed from
abstract, introduction, results and conclusion and explicitly withdrawn. "Only 2 of 4 monitors were re-scored
under v1/v3" is stated in Setup, Results and Discussion.

## 3. Skeleton / hack-variant alias
**Done.** Method has a new paragraph: skeleton p = j mod 12 fixes hack variant (p mod 2); skeletons 0-7 have 12
instances each, 8-11 have 6 each (corrected from the "20 vs fewer" in the referee's wording; 20 is per family).
New Table VIII: action AUROC by variant (instance bootstrap) and CoT B-vs-A by variant (6-cluster skeleton
bootstrap, flagged as likely under-covering). Text says the difference cannot be assigned to skeleton or variant.

## 4. Cue stems in hack actions
**Done.** 10/120 a1 texts (all fix_test, skip-marker variant) match `skip`/`skipped`; no a0 matches. Table VII
gives pooled and fix_test action AUROC with and without them (CIs). Excluding them lowers v2 pooled action AUROC
for Qwen2.5-1.5B 0.58 -> 0.52 and Qwen3-1.7B 0.67 -> 0.61; the paper says the action view is partly lexical and
notes that the exclusion also removes one hack variant.

## 5. H4
**Done.** "Supported" withdrawn. Verdict is now "not supported; pooled O uninterpretable". New Table IX gives
per-family O with skeleton CIs for all 8 rows (opposite-signed cells with CIs excluding 0, e.g. Qwen3-0.6B v2
accuracy +0.315, checksum -0.218, validator -0.310). Qwen2.5-0.5B pooled D-vs-A AUROCs below chance (0.476
action-only, 0.430 CoT+action) stated. Table V keeps the literal-rule column, relabeled, with the caveat in its
caption. The pooled-O figure (old Fig. 2) was dropped from the paper.

## 6. Acue-vs-A 0.368
**Done.** New paragraph: Qwen2.5-0.5B v2 Acue-vs-A 0.368 [0.215, 0.543] (benign cue-word text scored lower than
plain benign, contrary to a simple cue-firing account; upper bound above 0.5 so the inversion itself is not
established); range 0.368-0.606 over all 8 rows, no interval excludes 0.5. States that the non-negated
cue-mention condition was not run and that the pre-registered H3 verdict used the raw logit difference
(AUROC is post-hoc).

## 7. Bnc intent equivalence
**Partly (disclosure only).** Discussion lists the swaps fool->satisfy, fake->produce, trick->steer,
hack->quick edit, cheat on->only satisfy (`src/templates.py`, REV_TEST), states K is an upper bound on cue
reliance, and that no intent rating and no matched-strength non-cue control was done. The rating itself was not
done.

## 8. Per-family action AUROC CIs; "changes direction" claims
**Done.** Table VII now has CIs for every cell. The general "changes direction across prompts" claim is removed;
only the two cells with non-overlapping CIs on opposite sides of 0.5 are mentioned (Qwen3-0.6B speedup 0.54
[0.37, 0.69] v2 vs 0.26 [0.15, 0.36] v1; validator 0.20 [0.08, 0.32] vs 0.70 [0.64, 0.80]); other cross-prompt
differences are stated as not interpreted.

## 9. Log completeness
**Done.** Setup states: `results/test_run.log` has no Qwen3-1.7B entry (2529 s only in `results/timings.jsonl`);
the first 88 dev passes of Qwen2.5-1.5B under v1 are not in `results/dev_run.log`; the v1/v3 test scoring
(`src/run_test_prompts.sh`, default `PY`) and round-1 post-hoc analysis used a sibling project's venv (score
metadata record identical torch/transformers/NumPy/Python versions); `src/posthoc2.py` ran in this project's
`.venv`. The reproducibility block now passes `PY=.venv/bin/python` and includes the posthoc2 command.

## 10. New related work
**Done.** Both verified via export.arxiv.org/api/query (title, authors, year) before adding:
arXiv:2608.04735 (Duzan, Cooper Stickland, 2026, "Chain-of-Thought Monitoring Can Be Unreliable in
Implicit-Influence Settings") and arXiv:2609.19101 (Bergen et al., 18 authors, 2026, "Monitoring and Discovering
Reward Hacking with Internal Representations during LLM Evaluations"; the full arXiv title includes "during LLM
Evaluations"). Each is described in one or two paraphrased sentences in Related Work with how it differs.

## Not done (rebuild), and why
- Rebuild with >= 50 skeletons from independent writers: **not done**.
- Validation on real Terminal Wrench trajectories: **not done**.
- Intent-strength rating of the R_rev / R_revnc pairs and a matched-strength non-cue control: **not done**.
- Non-negated cue-mention condition: **not done**.
- All 4 monitors x 3 prompts (Qwen2.5-1.5B and Qwen3-1.7B under v1/v3): **not done**.
- CI for between-prompt differences in K: **not done**.

Reason: compute (CPU only; Qwen3-1.7B alone took 2529 s per prompt on test) and the one-cycle limit for this
revision. All items are listed as future work in the Conclusion; the paper's claims were reduced accordingly
rather than extended.
