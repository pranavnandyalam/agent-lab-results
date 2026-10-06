# PLAN: small-judge-reversal (rev 3c, 2026-10-06; rev1+rev2 REJECTed by overseer, fixes in 'Rev fixes' at the end)

## What's new here
Pairwise goal-reversal ("pick the WORSE response") is known for closed judges (GRP, 2503.06139) and criterion
reversal stickiness for one 7B pointwise judge (2609.02942). Not found: how this behaves in small open judges
(0.5-4B) as size shrinks, and whether failures are position-locking or criterion-blindness, via a pair-level
taxonomy over all 4 judgments. Novelty claim rests on an arXiv/web scan (Semantic Scholar 429; retry, GitHub
search by builder); 2609.02942 full text checked by overseer: single judge, no size ablation.

## Literature check (arXiv/web only; scout 2026-10-06, Semantic Scholar not used)
- 2609.02942 (Bagaria et al.): pointwise, one judge Qwen2.5-7B, HealthBench/ResearchRubrics; response and criterion reversal, 37.7% desired flips on response reversal. No sub-7B or size scaling reported (overseer read the HTML: no size ablation).
- Order-swap bias in small judges is covered: 2505.08498 (incl. Llama-3.2-3B), 2604.16790 (Qwen3-4B), 2606.19544 (small models most position-biased), 2505.10320 (J1). None does criterion reversal.
- 2504.01282 (PRIN, Ahn & Yin, COLM 2025): 'which are correct' vs 'which are incorrect' multiple-choice (MATH, MathQA, EquationInference); checked by overseer: models GPT-4/4o, Llama-3-8B, Llama-3.3-70B, Falcon-40B, Qwen2.5-72B, Mixtral-8x22B (smallest 8B); its size question is about number of options. Delta: no sub-8B judges, no model-size sweep, no pairwise order-swap, no split of position-locking vs criterion-blindness.
- 2503.06139 (GRP): pairwise 'worse' prompt under both orders, closed judges, JudgeBench. Delta: small open judges, size sweep, position-vs-content split.
- Difference: we do pairwise judging with criterion negation across sizes; and we separate position bias from criterion-insensitivity, which a plain swap test cannot.

## Hypotheses (pre-registered; primary prompt W1, bootstrap unit = pooled 300 pairs, 2000 resamples, 95% CI)
H1: conditional flip rate (below) rises with size. Tests: Qwen2.5 0.5B->1.5B and Qwen3 0.6B->1.7B (paired bootstrap of the difference; 2 tests, Bonferroni: 97.5% CI). Supported if the CI excludes 0 in the positive direction in both families; mixed if one; refuted otherwise. Because the conditioning subset (correct under "better") grows with size, report its n per model and also the unconditional chance-normalised flip rate.
H2: position-locked share among non-"correct reversal" pairs > 0.5, bootstrap 95% CI lower bound > 0.5; evaluated for the smallest judge of each family (Qwen2.5-0.5B, Qwen3-0.6B) separately.

## Method (inference only: next-token logits of A vs B)
Data: RewardBench revision 168d848cdbbea9764fae4a544dc9ca1e6cca4931 (ODC-By, not gated). Subsets chat, chat-hard, reasoning (exclude all safety subsets, and filter remaining items whose prompt is safety-style by keyword, report count). Keep pairs with both responses <=150 tokens; confirm >=350 remain before sampling (if not, raise to 200 and recompute budget). Eval: 3 disjoint 100-pair data resamples (seeds 0,1,2; inference is deterministic, so these are resamples, not seeds), 300 pairs pooled; 30 disjoint dev pairs used ONLY to check output format/token ids. Results reported per subset (chat, chat-hard, reasoning) and pooled.
Per pair, 4 judgments: {better, worse} x {chosen first, chosen second}. "Worse" asked with W1 "Which response is worse?" is PRIMARY for all tests. W2 "Which response should be rejected?" is a secondary robustness check on resample 0 only (100 pairs).
Readout: model's chat template; Qwen3 enable_thinking=False; verify token ids of "A"/" A"/"B"/" B" per tokenizer; score = logit(A)-logit(B). Report P-mass on {A,B} as format compliance; judges with mass <0.5 on dev are flagged and excluded from H tests (kept in tables).
Pair-level taxonomy over the 4 W1/better judgments (computed on W1 only; exact rules, mutually exclusive in this order):
 1. correct reversal: picks chosen in both orders under "better" and rejected in both orders under "worse".
 2. position-locked: same letter in all 4 judgments.
 3. criterion-blind: same response (content) chosen under both criteria in both orders.
 4. reversed-consistent: picks rejected under both orders of 'better' and chosen under 'worse' (judge systematically backwards).
 5. other/inconsistent.
Metrics: share of each category; pos-consistent accuracy under "better"; conditional flip rate = fraction of pairs correct under "better" in both orders for which the W1 choice is the rejected response in both orders; chance-normalised flip rate with analytic reference rows (random judge, always-A judge) in the table; length-criterion sensitivity control ("which is longer?", ground truth = length) labelled as such.
Models (safetensors, bf16/fp32, no trust_remote_code, revisions pinned in RESULTS): CORE: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3-0.6B, Qwen3-1.7B. EXTENSIONS only if the timed trial projects the core finishing in <=1.5 h: Qwen2.5-3B-Instruct (Qwen Research License, non-commercial research use: fine here), then Llama-3.2-1B (gated; drop if access fails, no workaround). No 7B. Revision SHA and license for each model recorded by fetch_data.sh in results/models.md BEFORE the first run; deps (torch, transformers, datasets) pinned in requirements.txt.
## Compute (re-derived, rev 3c)
Per model: 300 pairs x 4 passes + 100 x 2 (W2) + 100 x 2 (length control) = 1600 passes. Core params ~4.3B total (embeddings included). Prompt length: MEASURE mean tokens on the filtered set (prompts uncapped; also drop pairs whose full prompt >400 tokens). With ~350 tokens: 2*4.3e9*350*1600 = 4.8e15 FLOPs; at 250-500 GFLOPS (fp32, 8 threads, measured peak 700) = 2.7-5.4 h. The 20-pair timed trial decides. One ordered cut rule (replaces all others): projected >4 h -> use resamples 0,1 only (200 pairs, flagged); still >4 h -> drop W2 and length control; still >4 h after that -> stop and ask Pranav (questions.md); never drop a model family (H1 needs both). >2 h compute so Pranav is asked (Q-20261006-1, default proceed). Resumable per (model, resample) JSON; checkpoints in ~/scratch (allowed scratch dir).
## Stop criteria
Finish after one full sweep; cuts follow only the ordered cut rule in Compute (no model family is dropped). Extensions are expected NOT to run (core projects 2.7-5.4 h). H1 is reported 'inconclusive' if the conditional-flip subset for any model has n<30. If a judge shows chance accuracy everywhere, report it, do not hide it. Null result is reported as is.
## Ethics
Public benchmark of open local models, no jailbreaking, no personal data. RewardBench includes some safety prompts: exclude the safety subsets.

Ethics conditions: AI-authorship note; license recorded; limitations list (small models only, logit A/B protocol, <=150 token filter, one benchmark, novelty scan limited).

## Rev fixes
Rev1 (overseer): cited GRP 2503.06139; degenerate position share replaced by 4-judgment taxonomy; chance/competence-corrected conditional flip; falsifiable H1; logit readout spec (templates, enable_thinking=False, token ids, A/B mass); 2 negation wordings; length control relabelled; per-subset reporting; RewardBench pinned; compute redone.
Rev2 (overseer): pass count corrected (4 per pair core); compute budget and cuts stated; H2 exact statistic + named judges; H1 Qwen3 comparison, bootstrap unit, primary prompt, multiplicity, subset n; literature contradiction fixed (2609.02942 HTML read by overseer: Qwen2.5-7B-Instruct only judge, no size ablation); model licenses/revisions table + pinned deps required before first run. Taxonomy computed on W1 only. Safety-style keyword list (fixed now): kill, weapon, bomb, hack, suicide, drug, explosive, illegal, abuse, porn (whole-word, case-insens.); items matching are dropped, count reported. GitHub search + Semantic Scholar retry: the builder does one before the first run; novelty wording stays "not found in a limited scan".

- rev 3c: removed conflicting Qwen2.5-only cut rule; recorded PRIN model list/tasks; overseer APPROVED (2026-10-06).
