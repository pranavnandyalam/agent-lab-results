team: beta
# PLAN: quant-cot-looping (rev 3b, 2026-10-06)
## What's new here
We split the token inflation that quantization causes in a sub-1B reasoning model into degenerate loop tokens and distinct new-reasoning tokens, per bit width, and test whether loop-token share predicts failure better than raw length. Prior work reports loop/exhaustion rates per generation or semantic step repetition, never a token-level split in a bit sweep on a sub-1B model.

Closest prior (abstract pages verified; numbers via full-text summaries, cycle 2):
- 2606.25519: Qwen3-4B/30B, Phi-4-R (>=4B), INT4/INT3; step-count and embedding-similarity step repetition (semantic, per step); no loop/truncation split, no <4B.
- 2606.02011 (Table 1, App. B): Qwen3-8B/32B, W4/W2; loop = 20-gram x4 in 1024 tokens; reports (per our read of a full-text summary, unverified) a strong cross-setting negative correlation of loop rate with accuracy; no token decomposition. We use per-trace AUROC and token decomposition, a different analysis.
- 2609.26708: Qwen3-0.6B/1.7B/4B, RTN mixed 2.79/1.88-bit with QAD-trained models, MATH-500; binary per-generation flags (summary-read, unverified: most outputs of the low-bit QAD model end in repeated 8-grams and exhaust the budget, far above BF16); two settings, no split.
- 2606.00206: R1-Distill 1.5B+, "wait/but" markers; 2607.20129 mitigation controller.
Novelty is moderate: loop rates under quantization are partly known; the token split and the sub-1B bit sweep are new.

## Hypotheses (pre-registered)
Definitions. For quantized level q, problem set P, seeds S: T_q = mean generated tokens per generation (truncated ones count their generated length, cap 2048). L_q = mean loop tokens per generation (detector below). Added tokens A_q = T_q - T_fp32; added loop tokens AL_q = L_q - L_fp32.
- H1: share_q = AL_q / A_q (defined only if A_q > 0). Prediction: share rises as bits drop and the point estimate exceeds 0.5 at the lowest non-collapsed level. 95% CI by bootstrap over problems (10k resamples, resampling problems, all seeds of a problem kept together). Because of the 2048 cap, A_q is a lower bound; stated in results.
- H2: restricted to (problem, seed) pairs where BOTH fp32 and q finish (non-truncated), median paired relative change in distinct tokens is within +-30%. Decision rule: evaluated only if >=15 such pairs exist at that level; otherwise "not testable" (reported, not claimed).
- H3 (exploratory, no confirm/refute claim): per level, AUROC for incorrectness of loop fraction vs total length vs truncation indicator, on non-truncated traces and on all traces; bootstrap CIs over problems; shuffled-label null.

## Method
- Model: Qwen/Qwen3-0.6B rev c1899de289a04d12100db370d81485cdf75e47ca (Apache-2.0). Data: openai/gsm8k rev 740312add88f781978c0658806c59bc2815b9866 (MIT). 16 test problems (random, seed 0) for the study; 8 train problems for detector development only. Revisions are hardcoded in fetch_data.sh.
- Quantization: simulated weight-only RTN asymmetric group-wise on all Linear except lm_head (no C compiler, so no llama.cpp; the wheel cannot be built). Sanity (results/trial.json): w8g64 KL 0.0007, w4g64 KL 0.078, w3g64 KL 0.75 (drops think block), w2g64 gibberish. Levels: fp32, w5g64, w4g64, w3g32. Confound: w3 uses g32, w4/w5 use g64, so effective bits (incl. 16-bit scale+zero per group) are: w5g64 5.5, w4g64 4.5, w3g32 4.0; we plot against effective bits and say the w3-vs-w4 comparison mixes bits and group size. A level with accuracy <=2/16 on every seed is "collapsed" and excluded from H1/H2. w2 appears only as a collapsed-level note.
- Decoding: sampled only (Qwen advises against greedy in thinking mode): temp 0.6, top_p 0.95, top_k 20, 3 seeds (0,1,2), max_new_tokens 2048, fp32, batch 24 with compaction of finished rows, 4 threads.
- Answer extraction: if the think block is closed, take the text after </think>: last \boxed{}, else last number. If there is no </think> (truncated or dropped), score incorrect for truncated; for traces that ended (EOS) without a think block (observed at w3), extract from the whole text. Report accuracy overall and among finished traces.
- Loop detector, FROZEN before any test run (commit hash cited in RESULTS): a token span is a loop if a 20-token n-gram recurs >=4 times within 1024 tokens (2606.02011 definition) OR an n-gram of >=8 tokens repeats >=3 times consecutively. Loop tokens = tokens in repeats after the first occurrence; distinct = total - loop. Dev: 8 train problems (trial indices in results/timed_trial_w4.json) x 3 seeds at w4 (existing trial, 24 traces) and fp32 (existing seed-0 trial, 8 traces, plus one new fp32 batch for seeds 1-2, 16 traces); hand-label up to 40 flagged and 40 unflagged spans (if fewer flagged spans exist, label all and say so) (labeller = Lead, documented), report precision/recall; thresholds may change ONLY on dev, then frozen. After the test run: precision audit of 20 random flagged test traces, no retuning. Sensitivity: 25%-coverage variant and 8-gram-tail variant.
- Baseline: fp32. Null: shuffled-label AUROC.

## Compute (measured, results/timed_trial_w4.json)
Trial, w4g64, 24 generations (8 train problems x 3 seeds), one batch of 24 with compaction, 2048 cap, other team's job running: 902 s wall, 40,233 useful tokens, mean 1676 tokens, 14/24 truncated, 44.6 tok/s aggregate. Per-step time grows with context (0.35 s/step early to 0.52 s late) and does not fall with fewer active rows, so a batch costs about 850-900 s whenever any row reaches 2048, regardless of its mean length. Earlier 61-78 tok/s numbers were for 256-token runs and do not apply.
Costing by batch wall time (not tokens/s): batch = 24 generations = 8 problems x 3 seeds. Core: 16 test problems x 4 levels (fp32, w5g64, w4g64, w3g32) x 3 seeds = 192 generations = 8 batches x ~900 s = ~2.0 h (floor by token rate ~1.7 h; ceiling if contended ~2.4 h). Dev: reuse the w4 trial (24 traces) and the fp32 seed-0 trial (results/timed_trial_fp32.json: 8 gens, 388 s, 0/8 truncated, mean 1260 tokens), add one fp32 batch of 16 (seeds 1-2) = ~0.15 h. Total ~2.15 h (range 1.9-2.6 h). Approved by Pranav (Q-20261006-2 ANSWERED 2026-10-06, standing CPU approval; scope recorded there). Cut rule is by wall time: if the first core batch (run order: w4 first, then fp32, w3g32, w5g64) takes >1100 s, drop w5g64 (3 levels, ~1.5 h). Resumable per batch JSON (one batch = one chunk, <40 min). Never concurrent with another team's heavy job.

## Deviations from rev 3b (cycle 3, recorded before any test trace was analysed)
- Detector rule (A) additionally ignores 20-grams copied from the problem text (dev finding, DEV.md); all occurrences after the first are marked.
- Sensitivity variants: 16-gram x4 and 12-gram x4 (rule A only) replace the "25%-coverage" and "8-gram-tail" variants (the 25%-coverage variant was never defined; 8-gram variants flag too much, DEV.md).
- Dev hand-labelling was smaller than planned (7 flagged + 7 unflagged truncated traces, one rater, no recall figure). The post-run precision audit of 20 flagged test traces still stands.
- The core run runs concurrently with the other team's 4-thread job (owner's scope allows sharing at 4 threads each), so batch times are contended. The cut rule (first batch >1100 s -> drop w5g64) is applied by hand. A batch killed by the 2400 s timeout leaves no output and restarts on resume.
- Disclosure: the 16 test problems were also used for the earlier throughput/KL sanity trial (results/trial.json) that picked the quantization levels. The `boxed` field in raw JSONs is not the scoring rule; scoring follows the Method section and is done in analysis.
- Core JSON `argv` contains absolute paths; scrubbed before commit.

## Dependencies
requirements.txt (torch 2.14.1 CPU, transformers 4.57.6, pinned; results/pip_freeze.txt). Python 3.14.

## Risks/misuse
Low: defensive efficiency analysis on public small model and math data; no personal data. Caveats: RTN fake-quant is not GGUF k-quant; one 0.6B model; 16 problems gives wide intervals; the 2048 cap truncates >50% of w4 traces so lengths are censored; the detector may flag legitimate re-checking.

## Stop criteria
All levels except w5 collapsed or no truncation-uncensored data -> write up as is; 3 cycles without a new result -> archive.

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*
