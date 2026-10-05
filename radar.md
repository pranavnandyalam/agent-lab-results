# Radar — top AI trends right now

> Refreshed by the agent at least daily. Each entry: sources, date seen, why it matters, CPU-feasible
> angle. Stale items get dropped. Last refresh: 2026-10-05 (cycle 1, loop main).

1. **Eval rankings are less reproducible than assumed.** "How Reproducible Are Evaluation Conclusions?"
   (arXiv:2609.30074, Sept 24 2026). Identical prompts yield unstable inferred structure (Jaccard
   0.39-0.96); bootstrap shows only bottom-ranked models are reliably ranked. Source: arXiv abstract
   fetch (primary, verified). Why it matters: many published leaderboard claims may overstate confidence
   from single runs. CPU angle: run a <=3B/8B-quantized model N=30x on a fixed prompt set, compute
   Jaccard/bootstrap rank stability directly — cheap, CPU-only, hours.

2. **Contamination detectors may only catch verbatim leakage, not paraphrase.** "The Fragility of
   Benchmark Contamination Detection in Reasoning Models" (arXiv:2510.02386) and "Soft Contamination
   Means Benchmarks Test Shallow Generalization" (arXiv:2602.12413). Source: arXiv + HN discussion
   (2 sources, but dates are older than our 2-3wk window — flagged as background, not breaking news).
   Why it matters: standard contamination checks may be giving false confidence. CPU angle: run an
   existing contamination detector against paraphrased vs. verbatim benchmark items on a 1-3B model,
   measure false-negative rate.

3. **Benchmark exposure has a measurable causal effect, not just a binary contamination flag.**
   "LeakScale" (arXiv:2609.27176, Sept 23-24 2026). Exposure boosts accuracy +7-27pp across 2,048 task
   families using fresh executable tasks. Source: arXiv abstract (verified). Why it matters: reframes
   contamination from yes/no to a quantifiable dose-response effect. CPU angle: build ~50-100 small
   "private-fact" executable tasks, prompt-inject a subset into a <=3B model's context (not full
   fine-tune), measure accuracy delta vs. held-out. Days, CPU-only.

4. **Interpretability's faithfulness is shakier than assumed.** "The Misery of Mechanistic
   Interpretability: A Formal Perspective" (arXiv:2609.15533, Sept 14 2026). Interpretable replacement
   networks flip dominant features under minor input perturbations across 5 open-weight families
   (Gemma 3 1B, Llama 3.2 1B, etc.). Source: arXiv abstract (verified). Why it matters: undermines trust
   in common interpretability tools if faithfulness is only checked empirically. CPU angle: small open
   1B model, build a toy probe/IRN, test feature stability under paraphrase/typo perturbations.

5. **Automated red-teaming has a language coverage gap.** "A Translational Note on AI Safety Evaluation"
   (arXiv:2609.06573, Sept 6 2026). Non-English prompts expose harms that English-only automated
   red-teaming misses. Source: arXiv abstract (verified). Why it matters: argues deployment-context /
   multilingual evaluation remains necessary. CPU angle: run a harm-benchmark subset in English vs.
   2-3 non-English translations against a small open-weight model (1-3B), compare refusal rates.

6. **Agent tool failures are often silent, not crashes.** "Silent Failures in Agent-Tool Interaction"
   (arXiv:2609.26836, Sept 21 2026). Audited 15 scientific tools in ToolUniverse, found 91 failures
   (mostly missing/incomplete data with no error signal). Source: arXiv abstract (verified). Why it
   matters: agents may silently misreport confidence when tools degrade quietly. CPU angle: wire a
   <=8B quantized GGUF agent to a small toy tool set via a ReAct loop, inject malformed/truncated tool
   outputs, measure how often the agent reports success anyway.

7. **Multi-agent systems can be "right for the wrong reasons."** "Where Do Multi-Agent Systems Fail?"
   (arXiv:2609.38761, Sept 30 2026). Correct final answers can coexist with internal violations of
   collective-reasoning contracts; an LLM diagnoser catches more violations in traces than outcomes
   alone reveal, but transfers poorly across workflows. Source: arXiv abstract (verified). CPU angle:
   build a tiny 2-3 agent debate/pipeline task with a small open model, log traces, hand-write contract
   checks, compare outcome-only vs. trace-level failure detection.

8. **Chain-of-thought reasoning steps may be linearly decodable from hidden states.** "Beneath the
   Surface of Chains-of-Thought" (arXiv:2609.04753, Sept 4 2026). Reasoning operations (decomposition,
   deduction) are linearly separable in hidden states, peaking in middle layers, independent of surface
   wording. Source: arXiv abstract (verified). Scout found no smaller-model replication reported — real
   novelty gap. CPU angle: linear probes (logistic regression) on hidden states of Qwen2.5-1.5B or
   Llama-3.2-1B over GSM8K CoT traces; check if the claim holds at 1-3B scale.

9. **Gradient-alignment reweighting may fix DPO's low-beta instability.** "GAW-PO" (arXiv:2610.01511,
   Oct 1 2026). Reweights DPO's token-level loss by gradient alignment with the preferred response;
   +0.97 over vanilla DPO across 11 benchmarks, stable at low beta where vanilla DPO degrades. Source:
   arXiv abstract (verified), no public code repo found. Why it matters: directly relevant to Pranav's
   own preference-optimization work (MAPO, arXiv:2410.19499). CPU angle: tiny DPO vs. GAW-PO-lite
   replication on a <=0.5B model with a small preference subset, few epochs, CPU-only.

10. **Audio-language models fail at compositional sound understanding.** "Probing Large Audio-Language
    Models for Compositional Understanding of Sounding Actions" (arXiv:2609.35345, Sept 28 2026). New
    public benchmark shows models fail to compositionally infer human activities from atomic sound
    events. Source: arXiv abstract (verified), data/tools public. Why it matters: clean negative result
    with a reusable public benchmark. CPU angle: run a small open audio-language pipeline (CLAP + small
    LLM) on their public benchmark, check if the negative result replicates at small scale.

## Lower-confidence items seen but not promoted to top 10
- "Swift-Qwen3.8-27b" fast-thinking distillation claim (HF/HN, ~16-33 pts) — numbers only sourced via
  secondary aggregator, not independently confirmed on primary HF page; too large to run on CPU anyway.
- "Jeff-code" 0.8B helper model claiming 47% coding speedup for a 27B model (HN, 2 pts, single post) —
  single low-traffic source, unverified.
- "capbencher" leakage/gaming alarm tool (GitHub Show HN, 1 pt) — new, unreviewed, license unchecked.
These stay out of the radar top 10 pending better sourcing; logged here so we don't rediscover them from
scratch, but they should not be treated as confirmed trends yet.
