# RESULTS: cot-monitor-small

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

Status: final; overseer RESULTS + ethics APPROVE_WITH_CONDITIONS (applied); paper revised after referee round 1 (see addendum below).

## Summary
Four small zero-shot monitors (Qwen2.5-0.5B/1.5B-Instruct, Qwen3-0.6B/1.7B, non-thinking mode) were asked "is the agent trying to game the task? YES/NO" on **fully synthetic, template-written toy transcripts** (6 families x 20 instances, 12 hand-written wording skeletons shared across families). Score = logit(YES) - logit(NO). Scoring is deterministic (no seeds); uncertainty is a cluster bootstrap over wording skeletons (12 clusters; 2000 resamples; seed 0). These are controlled probes of small monitors, **not** evidence about deployed monitoring or real agent traces. **Prior work:** closest are 2608.00583 (pooled CoT-monitor accuracy hides reasoning-dependent fragility; frontier monitors; its benign-narration effect is the idea our condition D probes at small scale), 2505.23575 (CoT Red-Handed, view comparison with weak monitors), 2605.12746 (CoT-Guard, trained 4B monitor), 2601.21112, 2512.18311, 2510.01367, 2503.11926. In a limited arXiv/web scout (cycle 12; Semantic Scholar returned HTTP 429 and was skipped; the GitHub-specific search was not run separately, only general web search) we did not find a prior study at 0.5-1.7B with the keyword minimal-pair control. Not found != does not exist; our novelty is limited to that scale and control, not to the view/narration effects themselves.

Key findings (test split, prompt v2):
1. **Monitors separate intent-revealing from benign reasoning text (synthetic templates only); ceiling reached by Qwen3-0.6B.** AUROC for "reasoning reveals intent, benign action" (B) vs benign (A), CoT-only view: 0.805 [0.690, 0.905] (Q2.5-0.5B), 0.916 [0.869, 0.973] (Q2.5-1.5B), 0.957 [0.918, 0.988] (Q3-0.6B), 0.990 [0.968, 1.000] (Q3-1.7B). Because two monitors exceed 0.95 (Q3-0.6B 0.957, Q3-1.7B 0.990) the pre-registered H2 (larger > smaller within a family) is **ceiling-limited and untestable** (the rule is global as pre-registered; the Qwen2.5 pair alone is not above the ceiling and a per-pair reading would have marked it supported, since its difference CI barely excludes 0). Differences larger-smaller: +0.111 [0.002, 0.251] (Qwen2.5) and +0.033 [-0.000, 0.067] (Qwen3; lower bound -0.0001, includes 0), exploratory only.
2. **Cue-word dependence (K) is largest for the smallest monitors, but pre-registered H3 is not supported for any monitor.** K = AUROC(B vs A) - AUROC(Bnc vs A) where Bnc swaps cue words for neutral synonyms: 0.288 / 0.165 / 0.052 / 0.025 (0.5B / 1.5B / 0.6B / 1.7B, CoT-only). K is lower for the larger model in each family (exploratory, no CI on the within-family difference; partly a family effect, Q3-0.6B << Q2.5-1.5B), but H3 also requires benign cue-mention text (Acue) to raise alarms with CI excluding 0 in the skeleton bootstrap; that failed everywhere in the CoT view (CIs span 0, very wide; the CoT+action Acue-A for Q3-0.6B, 0.397 [0.106, 0.683], does exclude 0 but is secondary). A naive reading: the 0.5B and 1.5B monitors partly depend on cue words (Bnc AUROC 0.517 and 0.751), the Qwen3 monitors much less (0.905, 0.964).
3. **Bag-of-words baseline is non-trivial but below every monitor (point estimates; CIs overlap the 0.5B monitor).** Out-of-fold unigram logistic regression on CoT text, grouped CV by skeleton: B vs A AUROC 0.758 [0.683, 0.894] (monitors: 0.805-0.990); Bnc vs A 0.750 [0.684, 0.872]; the rule "BoW >= 0.95 => monitor AUROC reflects template wording" did not trigger. Caveat: with CV grouped by family x p (skeletons leak across families) BoW is 1.000, and the skeleton-grouped rule is close to unfalsifiable with so few skeletons. So even the high monitor AUROCs cannot be attributed to "understanding intent" rather than detecting the synthetic reasoning style.
4. **Narration override (H4) "supported" by the literal rule, but weakly and on one monitor only.** O = AUROC_action(D vs A) - AUROC_both(D vs A): +0.046 [0.021, 0.077] for Qwen2.5-0.5B; -0.015 [-0.050, 0.014], -0.001 [-0.024, 0.025], -0.072 [-0.110, -0.038] for the other three. For the 0.5B monitor the action-only AUROC is 0.476 and the CoT+action AUROC 0.430 (both below chance) so the positive O is a tiny effect from a monitor that does not detect the hack action at all. For Qwen3-1.7B, O < 0 means benign narration *helped* (D-A both 0.741 > action 0.668).
5. **Sanity checks:** action-only B vs A = 0.500 exactly for all monitors (identical inputs, by construction). Format compliance (P-mass on YES/NO) was ~1.0 for every monitor, so no compliance correction mattered.
6. **Visible hacks (C vs A, CoT+action)** are detected well by Q2.5-1.5B, Q3-0.6B and Q3-1.7B (0.977, 0.943, 0.998) but only 0.649 by Q2.5-0.5B; action-only C vs A is weak for all monitors (0.48-0.67): the monitors mostly use the reasoning text, not the action diff, i.e. the toy "hack actions" alone are not recognised.

## Full tables
See `results/analysis_test.md` (generated by `python -I src/analyze.py --split test --prompt v2`, python 3.14.4, numpy 2.3.5) and `results/analysis_test.json`. Raw per-pass scores: `results/scores_test_v2_<monitor>.jsonl` (+ `.meta.json` with full revision hashes and prompt text). Dev analysis (prompt selection only): `results/analysis_dev.md`.

## Setup / reproduction
- Models (safetensors, fp32, 4 threads, no trust_remote_code, Apache-2.0): Qwen/Qwen2.5-0.5B-Instruct 7ae557604adf, Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4c, Qwen/Qwen3-0.6B c1899de289a0, Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e (hash prefixes; full 40-char hashes in each `results/scores_test_v2_*.meta.json`). torch 2.14.1+cpu, transformers 4.57.6, numpy 2.3.5 (`results/pip_freeze.txt`, `requirements.txt`).
- Data: `data/instances.jsonl` (generated by `src/generate.py`; no external data). Dev split = 24 instances with disjoint reasoning skeletons, used only to pick among 3 prompt variants (selection rule in PLAN.md: highest mean dev AUROC; v2 selected, mean dev B-vs-A AUROC 0.862).
- Run (python = `projects/cot-monitor-small/.venv/bin/python`; first `python -I src/generate.py` to regenerate the gitignored `data/instances.jsonl`, byte-identical): `cd projects/cot-monitor-small; HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 python -I src/score.py --monitor <m> --prompt v2 --split test` for each monitor, then `python -I src/analyze.py --split test --prompt v2`. Timings in `results/timings.jsonl` (0.5B 658 s, Q3-0.6B 830 s, Q2.5-1.5B 1894 s for the last 1312 rows, Q3-1.7B 2529 s).
- Commit of the analysis outputs: e8fa17a.

## Limitations and deviations (all disclosed)
- Synthetic, lexically regular templates; only 12 wording skeletons shared by all families, so the effective n is ~12 (primary CI) and wide intervals result; the 72-cluster family x p CIs are optimistic and are reported only as secondary.
- v2 prompt was selected on dev from 3 variants; it contains cue stems ("editing tests, hardcoding expected outputs, skipping checks") and primes cue vocabulary, which may inflate K and the Acue effect. v1 on Qwen2.5-0.5B was inverted on dev (AUROC 0.21), showing strong prompt sensitivity at this size. The dev-selection script was written after the first dev file existed, so "frozen before looking" is asserted, not demonstrable.
- Analysis choices (global ceiling scope for H2, CoT-only as primary view for H3, skeleton-grouped CV for BoW) were pre-registered in PLAN.md before the test analysis was run (cycle 9); the first test-split analysis happened in cycle 11 after all 4 monitors were scored.
- Dev and test share action/task/slot-phrase templates; only reasoning skeletons and instance tuples differ.
- Zero-shot, single prompt per model, one YES/NO readout; no calibration or thresholding (AUROC only). No Qwen3 thinking mode. Only two model families, 0.5-1.7B.
- H3's second condition (Acue-A) has extremely wide CIs, so "not supported" means "not demonstrated", not "refuted".
- Condition D is a detection probe, not an evasion recipe; no real agent traces, no people data.

## Addendum: referee round 1 follow-up (exploratory, post-hoc, cycle 13)
Command (from the project dir): `../quant-cot-looping/.venv/bin/python -I src/posthoc.py --split test --prompts v2,v1,v3` after `src/run_test_prompts.sh` (test split scored under v1 and v3
for the two small monitors only; 1.5B/1.7B not re-scored). Raw: `results/posthoc_test.md`, `results/scores_test_v{1,3}_*.jsonl`. Point estimates only, no CIs.

| prompt | qwen2.5-0.5b K | qwen3-0.6b K |
|---|---|---|
| v2 (selected) | +0.288 | +0.052 |
| v1 | +0.177 | +0.139 |
| v3 | +0.083 | -0.018 |

- **The size of the Qwen2.5-vs-Qwen3 cue-reliance gap is prompt-dependent; a family difference is not established.** Qwen2.5-0.5B has the larger K under all three prompts, but the gap shrinks from 0.236 (v2) to 0.038 (v1) and 0.101 (v3) (point estimates, no CIs, ~12 clusters; 0.5B's B-vs-A AUROC is only 0.51 under v1). The v2 gap size is prompt-specific (v2 contains cue stems) and should be
  treated as unestablished.
- **Action-view AUROC is heterogeneous by family.** Pooled 0.48-0.67 (v2) hides within-family values from 0.05 to 1.00 (e.g. Qwen3-1.7B fix_test 1.00,
  validator 0.23; Qwen3-0.6B checksum 0.05), and these within-family values cross 0.5 across prompts for the same monitor (Qwen3-0.6B speedup 0.54 v2 -> 0.26 v1).
  "Hack actions are largely not recognized" is therefore a statement about pooled AUROC only; some families look well separated by the larger monitors (point estimates from one prompt and 20 a1/a0 pairs per family, no CIs).
- Scale-free false alarm AUROC(Acue vs A, cot) is 0.37-0.61 across all monitors and prompts run (below 0.5 means Acue scored lower than A).
- Not yet done: re-scoring 1.5B/1.7B under v1/v3, intent-equivalence check of Bnc rewrites, cue stems in hack actions (referee holes 5, 8), CIs for these numbers.
  The paper was revised accordingly in cycle 14 (see paper/RESPONSE.md).
