team: beta
# PLAN: quant-cot-looping (rev 2, 2026-10-06)

## What's new here
Quantization is known to lengthen CoTs of small reasoners (2606.25519 qualitative "more steps and more repetition"; 2606.00206 blames "wait/but" overthinking markers on R1-Distill-1.5B; 2504.04823 longer outputs). Nobody (found) *quantifies the split*: how much extra length is degenerate looping (repeated n-gram cycles, often ending in max-token truncation) vs. distinct new reasoning, and which part carries the accuracy loss. We give a per-quantization-level decomposition of token count = distinct-content tokens + loop tokens, and test whether a loop-token fraction predicts failure better than raw length. Closest prior: 2606.25519 (qualitative, INT4/3, bigger models), 2606.00206 (marker penalty, no loop split), 2607.20129 (mitigation controller). Full-text lit pass (cycle 1): 2606.25519 measures step-level semantic repetition (embedding cosine) at INT4/INT3 on Qwen3-4B..30B, no loop/truncation rate by bit width, nothing <4B. 2606.02011 (Qwen3-8B/32B, W4/W2) defines a loop as a 20-gram x4 in 1024 tokens and reports loop rate, but no token decomposition. 2609.26708 reports (snippet only, unverified in full text) Qwen3-0.6B at ~2.8 bit: 95% budget exhaustion, 70% repeated 8-grams vs BF16 12%, no bit sweep. 2606.00206 (R1-distill 1.5B..32B) blames overthinking markers; 2607.20129 mitigation at one setting. Gap: a per-bit sweep INT6..INT3 on a sub-1B model that splits added tokens into loop vs distinct, with truncation-corrected accuracy and a loop-detector robustness check. Novelty is moderate (loop rates under quantization are partly known); the decomposition and the sub-1B bit sweep are the new parts.

## Hypotheses
- H1: loop-token share of the added tokens (vs fp32) rises as bits drop and exceeds 50% at the lowest non-collapsed level.
- H2: on problems where BOTH fp32 and the quantized level finish (same problem, same seed index), the median paired relative change in distinct-content tokens is within +-30%.
- H3 (exploratory, no confirm/refute claim): within each level, AUROC for incorrectness of loop fraction vs total length vs truncation indicator, on non-truncated traces only and on all traces; bootstrap 95% CIs.

## Method
- Model: Qwen/Qwen3-0.6B rev c1899de289a04d12100db370d81485cdf75e47ca (Apache-2.0). Data: openai/gsm8k rev 740312add88f781978c0658806c59bc2815b9866 (MIT). 30 test problems (random, seed 0) for the study; 40 train problems for detector development only.
- Quantization: simulated weight-only RTN asymmetric group-wise on all Linear except lm_head (no C compiler -> no llama.cpp). Measured sanity (results/trial.json): w8g64 KL 0.0007, w4g64 KL 0.078, w3g64 KL 0.75 (drops the think block in the sample), w2g64 gibberish. So levels: fp32 baseline, w6g64, w5g64, w4g64, w3g32 (finer groups to keep 3-bit alive). 8-bit skipped (indistinguishable); w2 reported only as a collapsed-level note from the 12-prompt sanity. A level whose accuracy is <=3/30 on every seed is reported as "collapsed" and excluded from H1/H2.
- Decoding: sampled only (Qwen advises against greedy in thinking mode), temp 0.6 top_p 0.95, 3 seeds (0,1,2) = the >=3 seeds; no greedy. max_new_tokens 2048. fp32 (measured faster than bf16: 61 vs 55 tok/s, batch 12, 4 threads).
- Answer extraction: parse final number after </think> (last "\boxed{}" or last number); if the think block is unterminated at truncation, score incorrect (truncated = incorrect, counted separately). Report accuracy as overall and accuracy among finished traces.
- Loop detector, FROZEN before any test run (committed with hash cited in RESULTS): token span is a loop if an n-gram with n=20 tokens recurs >=4 times within 1024 tokens (the definition used by 2606.02011) OR a >=8-token n-gram repeats >=3 times consecutively. Loop tokens = tokens in repeats after the first occurrence; distinct tokens = total - loop tokens. Dev use: 40 train problems at fp32 and w4, hand-inspect ~40 flagged and ~40 unflagged spans (labeller = Lead, documented), report precision/recall; thresholds may be changed ONLY on dev, then frozen. After test run: precision audit of 20 random flagged test traces, no retuning. Sensitivity: also report with the 25% n-gram coverage variant and the 8-gram-tail variant.
- Baseline: fp32. Null: shuffled-label AUROC.

## Compute (measured 61 tok/s aggregate at batch 12, fp32, 4 threads)
Full design: 30 problems x 5 levels x 3 seeds = 450 generations x ~1300 mean tokens (truncated ones cost 2048) ~ 585k tokens ~ 2.7 h; dev run adds ~0.3 h. Required throughput for 3 h: 585k/10800 = 54 tok/s (met). Cut order if the timed 3-problem trial of w4 shows mean >1700 tokens: (1) drop w6g64 (4 levels, ~2.2 h); (2) 24 problems (~1.8 h). Needs Pranav's OK since >2 h (Q-20261006-2). Run in resumable per (level, seed) JSONs in chunks <=40 min. Do not run concurrently with another team's heavy job (shared 9 vCPU).

## Dependencies
requirements.txt in project (torch 2.14.1, transformers 4.57.6, pinned; see results/pip_freeze.txt). Python 3.14.

## Risks/misuse
Low: defensive efficiency analysis on public small model and math data; no personal data. Caveats: RTN fake-quant is not GGUF k-quant; one 0.6B model, 30 problems -> wide intervals; loop detector may flag legitimate re-checking.

## Stop criteria
Collapse of all levels except w6/w5 (nothing to sweep) -> write up as is; 3 cycles w/o new result -> archive. Data/licenses as above; fetch_data.sh pins revisions.
