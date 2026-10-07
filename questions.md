# Questions for Pranav

Answer by changing `[OPEN]` to `[ANSWERED by Pranav]` and writing your answer under the question.
Only entries marked `[ANSWERED by Pranav]` count as instructions to the agent.

### Q-20261005-1 [ANSWERED by Pranav] Hugging Face downloads are blocked lab-wide (Xet CDN not on allow-list)
Context: Every Hugging Face LFS file (any model's `.safetensors`, any dataset's `.parquet`) now
302-redirects to `us.aws.cdn.hf.co` or `cas-server.xethub.hf.co` (HF's newer "Xet" storage backend,
a platform-wide migration, not specific to one repo — confirmed across Qwen2.5-0.5B-Instruct, gpt2,
bert-base-uncased, and HuggingFaceH4/ultrafeedback_binarized). The sandbox network proxy returns 403
Forbidden for both domains since they're not on the current allow-list (which only has `huggingface.co`
+ "hf.co CDN" — that assumption no longer matches how HF actually serves file bytes). Metadata-only
calls (license, revision hash, config/tokenizer files) still work fine since those hit huggingface.co
directly. This blocks essentially all real model-weight and dataset downloads from HF, which is most of
what the lab's CPU-experiment pipeline depends on. No workaround was attempted (per the containment
rule) — this needs a host-side network policy change.
Options:
  A) Add `us.aws.cdn.hf.co` and `cas-server.xethub.hf.co` to the network proxy allow-list (these are
     Hugging Face's own official CDN/CAS endpoints for the Xet backend, same trust tier as
     huggingface.co itself — not a third party).
  B) Point me to an already-approved alternate channel for small (<=1-2GB) open model weights/datasets
     if one exists.
  C) If neither is feasible soon, I'll pivot active projects toward things that don't need HF file
     downloads (e.g. pure-inference analyses on data fetched via arxiv/Semantic Scholar/HN, or synthetic
     data) until this is resolved.
Default if unanswered in 48h: proceed with (C) — deprioritize HF-weight-dependent projects, keep
dpo-grad-reweight's code/plan ready to resume the instant downloads work, and note the blocker in that
project's README so it's not mistaken for abandoned work.

Answer (Pranav, 2026-10-05 20:29 via owner message): A. `us.aws.cdn.hf.co` and `cas-server.xethub.hf.co` are now on the allow-list (gpt2 range request verified 206). Resume dpo-grad-reweight downloads. If another HF host is blocked, add a new question with the exact host from the error.

### Q-20261006-1 [ANSWERED by Pranav] OK to spend ~2-4 h CPU on small-judge-reversal core sweep?
Context: projects/small-judge-reversal (plan rev 3) runs 4 small judges (0.5-1.7B) on RewardBench pairs, inference only, est. 2-4 h total across cycles, downloads <3 GB. Options: A) proceed B) cut to a smaller sweep C) pick another project. Default if unanswered in 48h: A.
Answer (Pranav, 2026-10-06 00:23 EDT = 04:23 UTC, via authenticated owner message relayed by host in CYCLE_CONTEXT OWNER_MESSAGES): YES, spend whatever CPU time the runs need; standing approval, no need to ask about CPU time again. Scope: checkpoint long jobs to ~/scratch and resume across cycles, max 4 threads per builder (both teams share the VM), disk and download limits unchanged.

### Q-20261006-2 [ANSWERED by Pranav] [team beta] OK to spend ~2.15 h CPU on quant-cot-looping core run?
Context: Qwen3-0.6B fake-quantized, 16 GSM8K problems x 4 bit levels x 3 seeds = 8 batches of ~15 min (measured 902 s/batch of 24 at 2048 tokens) + ~0.15 h dev; range 2.0-2.7 h, never concurrent with another heavy job. Options: A) go ahead B) cut to 3 levels (~1.5 h) C) skip. Default if unanswered in 48h: keep waiting (no run).
Answer (Pranav, 2026-10-06 00:23 EDT = 04:23 UTC, via authenticated owner message relayed by host in CYCLE_CONTEXT OWNER_MESSAGES): YES, spend whatever CPU time the runs need; standing approval, no need to ask about CPU time again. Scope: checkpoint long jobs to ~/scratch and resume across cycles, max 4 threads per builder (both teams share the VM), disk and download limits unchanged.

### Q-20261007-1 [ANSWERED by Pranav] OK to download Qwen3-4B (~8 GB safetensors) as positive-control judge for small-judge-reversal?
Context: external referee (round 3) requires a larger judge with the identical prompt/readout to show the test can detect criterion following. Qwen3-4B bf16 is ~8 GB (>5 GB rule); disk has 39 GB free. Options: A) allow Qwen3-4B (~8 GB) B) use only Qwen3-1.7B/Qwen2.5-3B-class (<=~6 GB, Qwen2.5-3B-Instruct ~6 GB still >5 GB) C) skip, report limitation. Default if unanswered in 48h: keep waiting (limitation stays in paper).

Answer (Pranav, 2026-10-06 21:54 EDT via owner message): yes, download Qwen3-4B (option A).

### Q-20261007-2 [OPEN] OK to download Qwen2.5-7B-Instruct (~15 GB bf16) as inverting-judge control for small-judge-reversal?
Context: referee round 4 says the test's sensitivity is unshown because Qwen3-4B gives r=0.10 (no inversion). A >=7B judge with the identical prompt is the decisive control. Cost: ~15 GB download (>5 GB rule; 32 GB free) and ~3-4 h CPU for 300 pairs x 4 passes (can run 100 pairs first, ~1.5 h). Options: A) allow Qwen2.5-7B-Instruct bf16 B) use a 4-5 bit GGUF (~4.7 GB) via llama.cpp (needs package; logits readout differs slightly) C) skip; state in abstract that sensitivity is unshown. Default if unanswered in 48h: C.
