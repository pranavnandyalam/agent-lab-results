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
