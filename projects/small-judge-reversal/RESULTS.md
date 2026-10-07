# RESULTS: small-judge-reversal

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

**Status: revised after external referee round 2 (see 'Referee round 2 analyses'); previously revised after REFEREE round 1 (calibrated section added; ethics APPROVE_WITH_CONDITIONS on the revision, conditions applied; overseer RESULTS fixes applied).** Plan: PLAN.md (rev 3c). Full tables: `results/analysis.md` / `analysis.json`.

## Setup (as pre-registered)
- 4 judges, logit A-vs-B readout, no thinking (Qwen3 `enable_thinking=False`), fp32 CPU: Qwen2.5-0.5B/1.5B-Instruct, Qwen3-0.6B/1.7B (revisions + licenses: `results/models.md`).
- 300 RewardBench pairs (rev 168d848c..., ODC-By), 3 disjoint resamples of 100, responses <=150 tokens. Deterministic inference, so resamples are data splits, not seeds. Primary prompts: "better" and W1 "Which response is worse?", both response orders (4 judgments/pair).
- Env: Python 3.14.4, torch 2.14.1, transformers 4.57.6 (`results/pip_freeze.txt`). Commands: `results/commands.md`. Raw outputs: `results/raw_<model>_r<k>.json`. Raw outputs complete at commit 30caa64; analysis: `src/analyze.py` (committed together with this file). The Qwen2.5-1.5B r2 run exceeded its shell timeout window (shared CPU) but completed.

## Headline (honest negative result)
**On this sample, none of the four sub-2B Qwen judges shows correct reversal under the argmax A/B readout (thinking off).** Out of 300 pairs per model, a correct reversal (chooses chosen under "better" and rejected under "worse", in both orders) happens 0, 0, 0 and 1 times (Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B, Qwen3-1.7B). **CORRECTION (referee round 1, see calibrated section below): argmax position-locking is real output behaviour, but it is explained by a letter offset, and the earlier inference "not criterion-blindness" was wrong. In a post-hoc, exploratory calibrated readout (letter offset removed) content preference is mostly shared between "better" and "worse" (corr 0.63-0.95), i.e. largely criterion-blind. Calibrated accuracy is near chance for several judges (0.50-0.61 for three of four), so d is partly noise. The argmax count (0/0/0/1) and the calibrated shares below measure different things.**

| model | n_cond | pos-consistent acc ("better") | cond. flip rate | uncond. flip | correct-reversal | position-locked | criterion-blind | reversed-consistent | other | A/B mass |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0 | 0.000 | undefined | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| Qwen2.5-1.5B | 31 | 0.103 | 0.000 | 0.013 | 0.000 | 0.710 | 0.003 | 0.000 | 0.287 | 0.996 |
| Qwen3-0.6B | 15 | 0.050 | 0.000 | 0.027 | 0.000 | 0.890 | 0.023 | 0.000 | 0.087 | 0.975 |
| Qwen3-1.7B | 129 | 0.430 | 0.008 | 0.040 | 0.003 | 0.430 | 0.113 | 0.000 | 0.453 | 0.745 |
| random judge (analytic) | - | 0.250 | 0.250 | 0.250 | 0.062 | 0.125 | 0.125 | 0.062 | 0.625 | - |
| always-A (analytic) | 0 | 0 | undefined | 0 | 0 | 1 | 0 | 0 | 0 | - |

n=300 pairs per model, single deterministic pass (no seeds apply). Shares are of all 300 pairs.

## Hypotheses
- **H1 (flip rate rises with size): INCONCLUSIVE by the pre-registered rule** (conditioning subset n<30 for Qwen2.5-0.5B (0) and Qwen3-0.6B (15)). Descriptive: Qwen3 0.000 -> 0.008, diff 97.5% CI [0.000, 0.030]; Qwen2.5 undefined -> 0.000. Chance-normalised unconditional flip diff CIs: Qwen2.5 [0.00, 0.04], Qwen3 [-0.02, 0.05]. Under the argmax readout all flip rates are near zero at both sizes. **Round-2 note:** the post-hoc calibrated readout does show a within-family size difference (see 'Referee round 2' below); the two readouts disagree on this point.
- **H2 (position-locked share of non-correct-reversal pairs > 0.5): SUPPORTED** for both pre-specified judges: Qwen2.5-0.5B 1.000 (CI [1.00, 1.00]; the model answers "A" on every one of its 1600 passes), Qwen3-0.6B 0.890 (95% CI [0.857, 0.927]). Exploratory: Qwen2.5-1.5B 0.710 [0.657, 0.763]; Qwen3-1.7B 0.431 [0.378, 0.487], i.e. argmax position-locking fades with size but is replaced by "other" inconsistent behaviour, not by correct reversal. **Caveat:** H2 holds by the pre-registered rule but is mostly explained by letter marginals (see letter-null below); on this sample the argmax readout shows no criterion sensitivity (whether this is criterion-blindness or a limit of the test is unresolved; see Round 2 analyses).

## Calibrated re-analysis (added after REFEREE_REPORT round 1; `src/calibrated.py`, `results/calibrated.md`)
Per pair and criterion, d = (score_cf - score_rf)/2 (A-minus-B logit; letter offset cancels; >0 = prefers chosen). Same 300 pairs, no new model runs (`.venv/bin/python -I src/calibrated.py`, bootstrap over pairs, seed 0, 2000 resamples).

| model | corr(d_better, d_W1) [95% CI] | same sign | calibrated acc ("better") | calibrated correct reversal |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.95 [0.92, 0.97] | 0.92 | 0.61 | 0.06 |
| Qwen2.5-1.5B | 0.63 [0.51, 0.73] | 0.71 | 0.50 | 0.22 |
| Qwen3-0.6B | 0.90 [0.87, 0.93] | 0.87 | 0.57 | 0.08 |
| Qwen3-1.7B | 0.63 [0.55, 0.71] | 0.76 | 0.81 | 0.18 |

- **Conclusion changes (post-hoc):** content preference is strongly *positively* correlated across "better" and "worse" for all four judges, so the failure is criterion-blindness (the question wording has little effect on which response is preferred; mostly it shifts the letter). Letter-prior shifts (Qwen3-1.7B: P(A) 0.34 better vs 0.12 W1) exist but do not reflect content reversal. The earlier "position-locking" shares are real argmax outputs but are explained by the letter offset. The calibrated "reversal" shares (6-22%) are just the tail where noise in d changes sign (the same-sign share is 71-92%); they are not evidence of reversal and a null for the calibrated readout was not defined.
- **Letter-biased null (referee fix 2):** an independent-letter judge with each model's observed per-criterion A-rates would give all-four-passes-same-letter in 1.00/0.63/0.79/0.34 of pairs (observed 1.00/0.71/0.89/0.43). So H2 as stated adds little beyond letter marginals; under the uniform null it is nearly tautological. Position-consistent "better" accuracy: Qwen3-0.6B 0.050 vs letter-only prediction 0.053; Qwen2.5-1.5B 0.103 vs 0.159 (below the letter-only level). Observed same-letter shares exceed the null by 0.08-0.10 for three of four models, so the null explains most but not all of H2; no CIs on the null.
- **Sample composition (fix 3):** 217/300 pairs are HumanEvalPack code pairs (correct vs buggy, often near-identical text), 83 non-hep. The correlation finding holds in both subsets (corr 0.57-0.96); calibrated accuracy differs (e.g. Qwen2.5-0.5B 0.43 non-hep vs 0.68 hep; Qwen3-1.7B 0.61 vs 0.88) (`results/calibrated.md`).
- **Readout reliability (fix 4):** restricting to pairs with A/B mass > 0.9 on all four passes (n=300/288/286/140) gives the same picture (corr 0.60-0.95). A generation-based check was NOT run.
- Not done: 3-4B judge, thinking-enabled condition (optional fix 7).

## Referee round 2 analyses (post hoc; `src/referee2.py`, `results/referee2.md/.json`; no new model runs)
- **Calibrated size trend (paired bootstrap, larger minus smaller, same 300 pairs):** r(d_better,d_W1) and same-sign share FALL with size in both families. Qwen2.5 0.5B->1.5B: dr -0.32 [-0.43,-0.22], dsame-sign -0.21 [-0.27,-0.15], dacc_cal -0.12 [-0.19,-0.04]. Qwen3 0.6B->1.7B: dr -0.27 [-0.37,-0.18], dsame-sign -0.10 [-0.16,-0.05], dacc_cal +0.23 [0.17,0.29]. Post hoc, two families x two sizes, confounded by letter-bias differences; H1 (argmax) and the calibrated readout differ.
- **Magnitude:** OLS slope of d_W1 on d_better 0.82/0.39/0.67/0.26 (Qwen2.5-0.5B/1.5B, Qwen3-0.6B/1.7B); median |d_W1|/|d_better| 0.90/0.82/0.89/0.35. Order-averaged P_C(better)+P_C(W1) = 1.002/0.997/1.014/1.220; Delta = P_C(better)-P_C(W1) = -0.000/+0.013/+0.007/+0.135. For Qwen3-1.7B the pair-dependent shift shrinks ~3x under "worse" (weak partial reversal); the earlier phrase "barely changes which response is favored" is withdrawn for that judge. S~1 does not discriminate criterion following (a letter-following judge also gives S=1).
- **Length as partial positive control (resample 0, 72 non-tied pairs):** r(d_length, char-length diff) = 0.12 [-0.20,0.40] / 0.68 [0.46,0.81] / 0.60 [0.33,0.76] / 0.46 [0.20,0.64]; calibrated sign accuracy 0.65/0.61/0.64/0.58. But r(d_length, d_better) (n=100) = 0.96/0.89/0.91/0.74, and r(d_better, length diff) = 0.06/0.50/0.50/0.34, so not a clean control.
- **HEP edit distance:** token Levenshtein between C and R, quantiles (0/10/25/50/75/90/100%) 1/1/1/1/2/3/24; 131/217 at distance 1, 50 at 2, 36 at >=3; 204/217 below 5 tokens (non-HEP median 45). r by bin (ed=1 / ed=2 / ed>=3): Qwen2.5-0.5B 0.96/0.95/0.97; Qwen2.5-1.5B 0.70/0.26/0.63; Qwen3-0.6B 0.88/0.90/0.95; Qwen3-1.7B 0.70/0.68/0.67. HEP results are about one-word bug edits.
- **Selection / nulls:** mass>0.9 filter keeps 140/300 pairs for Qwen3-1.7B with calibrated acc 0.72 vs 0.81 overall (`calibrated.md`). Letter-null excess (observed vs expected same-letter: 0.71 vs 0.63, 0.89 vs 0.79, 0.43 vs 0.34) suggests pair-level letter offsets. All calibrated/null analyses are post hoc, without multiplicity control; W2 and the both-orders length shares are argmax-based.
- **Revised headline wording:** on this sample, for these small judges, the logit readout is dominated by a pair-dependent shift that the negated question does not invert. "Criterion-blind" is used only in this operational sense.
- **NOT DONE (next cycle):** generation-based readout validation (greedy decode on >=100 pairs); positive control from a larger judge (Qwen3-4B / Qwen2.5-7B). Without it, "criterion-blind" cannot be separated from "the test cannot detect criterion following".

## Secondary and controls (resample 0 only, 100 pairs; exploratory)
- W2 wording ("which should be rejected?"): correct-reversal share 0 for all four judges; same conclusion as W1.
- Length control ("which is longer?", truth = longer by characters, n=72 non-tied pairs, correct in both orders): 0.000, 0.000, 0.014, 0.250. Even this trivially checkable criterion is not followed by 3 of 4 judges, consistent with letter-prior dominance at argmax rather than a "worse"-specific failure. Qwen3-1.7B's 0.250 equals the random-judge level (0.25), so no judge shows length sensitivity at argmax (the calibrated readout does show length signal for three judges: r = 0.68, 0.60, 0.46, see Round 2 analyses). Letter use: Qwen2.5-0.5B answers A on 1600/1600 passes (its H2 'support' is a pure format failure, identical to the always-A reference); Qwen2.5-1.5B is mostly B (1467 B vs 133 A); Qwen3-0.6B mostly A (1527/1600).
- Per-section tables are in `results/analysis.md`. The chat section is tiny (n=12 of 300, because most chat responses exceed 150 tokens), and reasoning dominates (n=223); per-section differences should not be over-read.

## Limitations
- The calibrated analysis is post hoc (not pre-registered), assumes the letter offset is additive and identical in both orders (content-by-slot interactions, e.g. length, would land in d), has no null for its reversal shares, and Qwen2.5-0.5B (always 'A') has tiny d relative to its letter offset (median |d_better| 0.047 vs mean A-minus-B score 2.80; Qwen3-1.7B median 5.4; `paper/figures/paper_stats.json`), so its 0.95 correlation is not evidence of a coherent preference. Calibrated code: `src/calibrated.py` (this commit).
- Only four Qwen models (<=1.7B), one benchmark, A/B logit readout without reasoning, responses <=150 tokens and full prompt <=400 tokens (of 2985 RewardBench items: 740 safety-subset and 1 safety-keyword excluded, 1021 response >150 tokens, 213 prompt >400 tokens, leaving a pool of 1010: chat 46, chat-hard 275, reasoning 689; this biases to short/code/math items), no 3B+ judge (extensions not run). Larger or reasoning-enabled judges may behave differently; we make no claim about them.
- The low "better" accuracy partly reflects the strict requirement of being correct in both orders; the models are weak, not necessarily harmed by the prompt. Qwen3-1.7B has mean A/B mass 0.745 and 25.2% of its better/W1 passes have mass <0.5 (Qwen3-0.6B: 1.7%; others 0), so its A-vs-B logit comparison is less clean.
- One pass per prompt, no prompt-wording search beyond W1/W2; conclusions are about these prompts. Novelty claim rests on a limited arXiv/web scan (Semantic Scholar was rate-limited); closest work is listed in PLAN.md (2609.02942, 2503.06139, 2504.01282, 2609.32407).
- Not preregistered: the length control scoring by characters, and the exploratory H2 values for the larger judges.

## Reproduce
See README.md and `results/commands.md`.
