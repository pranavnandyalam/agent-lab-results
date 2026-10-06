# Backlog (re-scored novelty-first, 2026-10-06, cycle 5)

Format: `- [P] title | original angle (what is new) | why it matters | sources | CPU feasibility`
Score = novelty (highest weight) x trend importance x CPU feasibility x finishable write-up.
Novelty notes are from arXiv/web search only (Semantic Scholar was rate-limited, 429); every PLAN needs a fuller check.

## P0
- [P0] Reasoning-dependent CoT-monitor curve at 1-3B | Build matched transcripts (same actions, benign vs rewritten reasoning) and measure how a 1.7-3B monitor's catch rate collapses on hacks only the reasoning reveals; no 1-3B monitor study found | safety eval with a clean pass/fail design | https://arxiv.org/abs/2608.00583 , https://arxiv.org/abs/2608.04735 , CoT-Guard 2605.12746 (4B+) | high, synthetic transcripts, inference only
- [P0] Small-judge reversal test | Does LLM-judge "rubric artifact" (verdict barely flips when criterion/response reversed) get better or worse as the judge shrinks to 0.5-3B? Prior work only 7B+ | every cheap eval pipeline uses small judges | https://arxiv.org/abs/2609.02942 | high, 1 day
- [P0] Token inflation vs looping in quantized small reasoners | Decompose the CoT-length growth under INT4/3/2 into extra genuine steps vs repetition on 0.5-1.5B (GGUF Q8..Q2); unclear if 2606.25519 already splits it, check first | quantization speedups may be eaten by longer outputs | https://arxiv.org/abs/2606.25519 , https://arxiv.org/abs/2607.20129 | high, llama.cpp, 1-2 days

## P1
- [P1] CoT verbalization by cue placement at sub-4B | Extend the 15-model benchmark below its 4B floor (Qwen3-1.7B, R1-distill-1.5B); does the tool-return < user-message verbalization gap persist? | agent faithfulness | https://arxiv.org/abs/2608.29464 | medium (slow reasoning generation)
- [P1] Judge calibration vs ranking at 1.5-3B | rank-gap-conditional consistency + calibration-corrected acceptance for small judges on a public preference set | judges can rank well and still be miscalibrated | https://arxiv.org/abs/2610.02492 , https://arxiv.org/abs/2609.37577 | high
- [P1] Scaffold-flip experiment | one small model, vary only scaffold (prompt, tool format, retry), count model-ranking flips | agent leaderboards unreliable | https://arxiv.org/abs/2610.00651 | medium-high
- [P1] Two-stage vs mixed synthetic data on a tiny LM | first empirical LM test of the linear-regression theory (synthetic-then-real avoids error floor); no empirical test found | model-collapse debate | https://arxiv.org/abs/2609.09572 , https://arxiv.org/abs/2609.18878 | medium, 10-30M GPT, 2-3 days
- [P1] Zero-data byte pretraining at toy scale | does program/UTM-output pretraining of a 1-5M byte LM transfer to tiny real text? adaptive-curriculum angle not found | data-free pretraining | https://arxiv.org/abs/2609.30063 (procedural prior work 2601.21725, 2505.22308) | medium

## P2
- [P2] Eval rank stability audit at small scale | N=30 repeat runs on <=3B model; extends the claim to quantized models | https://arxiv.org/abs/2609.30074 | very high; novelty is only moderate
- [P2] CoT reasoning-operation linear separability at 1-3B | no smaller-model replication known | https://arxiv.org/abs/2609.04753 | high
- [P2] Per-layer SNR predictor of quantization loss | test the SNR account (2608.08188) predictively on a 0.5B model | https://arxiv.org/abs/2608.08188 | medium
- [P2] Silent tool-failure injection on a toy ReAct agent | https://arxiv.org/abs/2609.26836 | high
- [P2] Mech-interp feature faithfulness under paraphrase noise | https://arxiv.org/abs/2609.15533 | medium

## P3
- [P3] LeakScale causal contamination probe (needs hand-built tasks) https://arxiv.org/abs/2609.27176 ; multilingual red-teaming gap (ethics sign-off required) https://arxiv.org/abs/2609.06573 ; multi-agent right-for-wrong-reasons https://arxiv.org/abs/2609.38761 ; SynthSentry paraphrase stress-test https://arxiv.org/abs/2609.12353 ; contamination detector stress-test https://arxiv.org/abs/2510.02386 ; quantized code model tradeoff https://arxiv.org/abs/2601.02563
- Dropped: DPO GAW-PO-lite (archived, see projects/dpo-grad-reweight), sound-action replication, music stem clustering, Swift-Qwen/Jeff-code (unverified HN claims).
