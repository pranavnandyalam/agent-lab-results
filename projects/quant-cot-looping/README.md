# quant-cot-looping

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

## Abstract
When a small reasoning model is weight-quantized, its chain of thought gets longer and accuracy drops. How much of the extra length is degenerate looping versus more ordinary reasoning? We sample Qwen3-0.6B on 16 GSM8K test problems x 3 seeds at fp32 and three simulated RTN weight-quantization levels (5.5, 4.5, 4.0 effective bits), with a frozen n-gram loop detector. Results (preliminary, 16 problems, one model): at 5.5 and 4.5 bits, loop tokens are only ~10-14% of the added tokens (CI up to 0.51 / 0.23; hypothesis ">50%" refuted for the primary detector), most added length is not detected as exact-repeat loops (soft repetition not ruled out; detector recall unmeasured); at 4.0 bits (g32) the model collapses completely (0/48 correct, all traces hit the 2048 cap, 46/48 heavily looping). Everything is censored by the 2048-token cap, and truncation, not wrong answers, is the dominant failure even at fp32 (17/48 truncated).

Full numbers, caveats and audit: [RESULTS.md](RESULTS.md). Pre-registered plan: [PLAN.md](PLAN.md). Detector development: [DEV.md](DEV.md).

## Method (short)
Fake-quant weight-only RTN (group-wise, all Linear except lm_head), sampled decoding (T 0.6, top-p 0.95, top-k 20), 3 seeds, max 2048 new tokens, 4 CPU threads. Loop = 20-gram recurring >=4x in 1024 tokens (excluding problem-text n-grams) or >=8-gram repeated >=3x consecutively. Bootstrap over problems for CIs.

## Reproduce (CPU only, ~2.5 h of generation in 8 batches of 11-20 min)
```
cd projects/quant-cot-looping
bash fetch_data.sh                      # pinned model + dataset revisions
uv venv .venv && uv pip install -r requirements.txt   # python 3.14, CPU torch
cd ../.. && bash projects/quant-cot-looping/src/run_core.sh   # resumable, writes results/core/*.json
projects/quant-cot-looping/.venv/bin/python -I projects/quant-cot-looping/src/analyze_core.py --audit  # regenerates analysis.md, audit_sample.md
```
Raw generations are committed under `results/core/`. Sampling on CPU is not bit-reproducible across machines; expect the same qualitative picture, not identical tokens.

## Limitations
See RESULTS.md: one 0.6B model, simulated RTN (not GGUF), 16 problems, 2048-token cap, single-rater audit, group-size confound at the lowest level.
