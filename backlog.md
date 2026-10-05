# Backlog

Format: `- [P0-P3] <title> | why it matters | source URL | CPU feasibility | added <date>`
Scored on: trend importance x novelty x CPU feasibility x chance of a finished write-up.

## P0
- [P0] DPO gradient-alignment reweighting replication (GAW-PO-lite) | directly extends Pranav's own
  preference-optimization lens (MAPO); clean baseline-vs-method design; testable low-beta stability
  claim | https://arxiv.org/abs/2610.01511 | high: <=0.5B model, small UltraFeedback subset, CPU-only,
  hours | added 2026-10-05
- [P0] Eval rank stability self-audit replication | cheap, fast, directly checks a reproducibility claim
  that affects how we trust every other benchmark number we cite | https://arxiv.org/abs/2609.30074 |
  very high: pure inference, N=30 runs on a fixed prompt set, CPU-only, <1 hour | added 2026-10-05

## P1
- [P1] Silent tool-failure injection on a toy ReAct agent | agents-and-failure lens; extends a verified
  audit methodology to a new small-agent setting | https://arxiv.org/abs/2609.26836 | high: <=8B
  quantized GGUF agent, small toy tool set, CPU-only, 1-2 days | added 2026-10-05
- [P1] CoT reasoning-operation linear separability at 1-3B scale | scout found no smaller-model
  replication reported — real novelty gap on a reasoning/interpretability-adjacent claim |
  https://arxiv.org/abs/2609.04753 | high: linear probes on Qwen2.5-1.5B/Llama-3.2-1B hidden states over
  GSM8K, CPU-only, ~1 day | added 2026-10-05
- [P1] LeakScale-style causal contamination probe | reframes contamination from binary flag to
  measurable dose-response effect; good fit for eval/reproducibility lens |
  https://arxiv.org/abs/2609.27176 | medium: needs ~50-100 custom "private-fact" executable tasks built
  by hand, CPU-only, days | added 2026-10-05
- [P1] Multilingual red-teaming coverage gap probe | safety-eval lens; tests whether a small open model's
  refusal behavior degrades on non-English harm prompts | https://arxiv.org/abs/2609.06573 | medium:
  needs a harm-benchmark subset + translations (ethics-reviewer sign-off required before starting), small
  open model, CPU-only | added 2026-10-05

## P2
- [P2] Mechanistic interpretability faithfulness under perturbation | interpretability/safety lens; toy
  IRN/probe on a 1B open model, test feature-flip under paraphrase/typo noise |
  https://arxiv.org/abs/2609.15533 | medium: needs a probe-building step before the actual test, CPU-only,
  1-2 days | added 2026-10-05
- [P2] Multi-agent "right for wrong reasons" trace-vs-outcome check | agents lens; small 2-3 agent debate
  pipeline, hand-written contract checks vs. outcome-only grading |
  https://arxiv.org/abs/2609.38761 | medium: needs hand-authored contracts and a small multi-agent setup,
  CPU-only, 1-2 days | added 2026-10-05
- [P2] Compositional sound-action understanding replication | multimodal/audio lens; public benchmark,
  clean negative result, good reproducibility target | https://arxiv.org/abs/2609.35345 | medium: depends
  on finding a small CPU-runnable audio-language pipeline (CLAP + small LLM), 1-2 days | added 2026-10-05
- [P2] Quantization/distillation quality tradeoffs on a small code model | efficiency lens; directly
  testable pass@1 comparison, fp16 vs. 4-bit GGUF | https://arxiv.org/abs/2601.02563 (older background
  paper, needs fresh novelty check before starting) | high: Qwen2.5-Coder-1.5B, small HumanEval-style
  subset, llama.cpp, CPU-only | added 2026-10-05

## P3
- [P3] Benchmark contamination detector paraphrase stress-test | background trend (papers ~1yr and
  ~8mo old), still untested at small scale | https://arxiv.org/abs/2510.02386 ,
  https://arxiv.org/abs/2602.12413 | high: 1-3B model, existing detector, CPU-only | added 2026-10-05
- [P3] Music stem slot-embedding clusterability check (Stembed-adjacent) | audio embeddings lens, but
  likely needs training rather than eval-only | https://arxiv.org/abs/2609.35672 | low-medium: eval-only
  angle on an existing small CLAP model is feasible, full reproduction is not | added 2026-10-05
- [P3] "Swift-Qwen" fast-thinking distillation claim | unverified secondary-source numbers, target model
  too large for CPU anyway | HN/HF (low-confidence secondary aggregator) | low | added 2026-10-05
- [P3] "Jeff-code" 0.8B helper speedup claim | single low-traffic HN post (2 pts), unverified numbers |
  HN (low confidence) | medium (small models) but low priority given weak sourcing | added 2026-10-05
