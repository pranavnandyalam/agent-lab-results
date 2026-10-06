# RESULTS: small-judge-reversal

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

**Status: final (overseer REJECT fixed; ethics APPROVE_WITH_CONDITIONS, conditions met).** Plan: PLAN.md (rev 3c). Full tables: `results/analysis.md` / `analysis.json`.

## Setup (as pre-registered)
- 4 judges, logit A-vs-B readout, no thinking (Qwen3 `enable_thinking=False`), fp32 CPU: Qwen2.5-0.5B/1.5B-Instruct, Qwen3-0.6B/1.7B (revisions + licenses: `results/models.md`).
- 300 RewardBench pairs (rev 168d848c..., ODC-By), 3 disjoint resamples of 100, responses <=150 tokens. Deterministic inference, so resamples are data splits, not seeds. Primary prompts: "better" and W1 "Which response is worse?", both response orders (4 judgments/pair).
- Env: Python 3.14.4, torch 2.14.1, transformers 4.57.6 (`results/pip_freeze.txt`). Commands: `results/commands.md`. Raw outputs: `results/raw_<model>_r<k>.json`. Raw outputs complete at commit 30caa64; analysis: `src/analyze.py` (committed together with this file). The Qwen2.5-1.5B r2 run exceeded its shell timeout window (shared CPU) but completed.

## Headline (honest negative result)
**None of the four sub-2B judges can follow the "pick the worse response" instruction.** Out of 300 pairs per model, a correct reversal (chooses chosen under "better" and rejected under "worse", in both orders) happens 0, 0, 0 and 1 times (Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B, Qwen3-1.7B). The dominant failure in the smallest judges is position-locking, not criterion-blindness.

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
- **H1 (flip rate rises with size): INCONCLUSIVE by the pre-registered rule** (conditioning subset n<30 for Qwen2.5-0.5B (0) and Qwen3-0.6B (15)). Descriptive: Qwen3 0.000 -> 0.008, diff 97.5% CI [0.000, 0.030]; Qwen2.5 undefined -> 0.000. Chance-normalised unconditional flip diff CIs: Qwen2.5 [0.00, 0.04], Qwen3 [-0.02, 0.05]. Everything is near zero, so there is no evidence of any size trend in this range.
- **H2 (position-locked share of non-correct-reversal pairs > 0.5): SUPPORTED** for both pre-specified judges: Qwen2.5-0.5B 1.000 (CI [1.00, 1.00]; the model answers "A" on every one of its 1600 passes), Qwen3-0.6B 0.890 (95% CI [0.857, 0.927]). Exploratory: Qwen2.5-1.5B 0.710 [0.657, 0.763]; Qwen3-1.7B 0.431 [0.378, 0.487], i.e. position-locking fades with size but is replaced by "other" inconsistent behaviour, not by correct reversal.

## Secondary and controls (resample 0 only, 100 pairs; exploratory)
- W2 wording ("which should be rejected?"): correct-reversal share 0 for all four judges; same conclusion as W1.
- Length control ("which is longer?", truth = longer by characters, n=72 non-tied pairs, correct in both orders): 0.000, 0.000, 0.014, 0.250. Even this trivially checkable criterion is not followed by 3 of 4 judges, consistent with position-locking rather than a "worse"-specific failure. Qwen3-1.7B's 0.250 equals the random-judge level (0.25), so no judge shows length sensitivity. Letter use: Qwen2.5-0.5B answers A on 1600/1600 passes (its H2 'support' is a pure format failure, identical to the always-A reference); Qwen2.5-1.5B is mostly B (1467 B vs 133 A); Qwen3-0.6B mostly A (1527/1600).
- Per-section tables are in `results/analysis.md`. The chat section is tiny (n=12 of 300, because most chat responses exceed 150 tokens), and reasoning dominates (n=223); per-section differences should not be over-read.

## Limitations
- Only four Qwen models (<=1.7B), one benchmark, A/B logit readout without reasoning, responses <=150 tokens and full prompt <=400 tokens (of 2985 RewardBench items: 740 safety-subset and 1 safety-keyword excluded, 1021 response >150 tokens, 213 prompt >400 tokens, leaving a pool of 1010: chat 46, chat-hard 275, reasoning 689; this biases to short/code/math items), no 3B+ judge (extensions not run). Larger or reasoning-enabled judges may behave differently; we make no claim about them.
- The low "better" accuracy partly reflects the strict requirement of being correct in both orders; the models are weak, not necessarily harmed by the prompt. Qwen3-1.7B has mean A/B mass 0.745 and 25.2% of its better/W1 passes have mass <0.5 (Qwen3-0.6B: 1.7%; others 0), so its A-vs-B logit comparison is less clean.
- One pass per prompt, no prompt-wording search beyond W1/W2; conclusions are about these prompts. Novelty claim rests on a limited arXiv/web scan (Semantic Scholar was rate-limited); closest work is listed in PLAN.md (2609.02942, 2503.06139, 2504.01282, 2609.32407).
- Not preregistered: the length control scoring by characters, and the exploratory H2 values for the larger judges.

## Reproduce
See README.md and `results/commands.md`.
