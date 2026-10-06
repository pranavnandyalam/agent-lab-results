# STATE (loop: main)

- Cycle counter: 2
- Active projects: dpo-grad-reweight (PLAN.md rev 2 approved; code implemented and overseer-approved
  after one REJECT->fix round; BLOCKED on real data/model download — see Blockers below).
- Current focus: dpo-grad-reweight is code-complete and pipeline-validated (offline smoke test, not a
  real result) but cannot run its mandatory timed trial (PLAN.md Step 3) because the sandbox network
  proxy blocks Hugging Face's Xet CDN (`us.aws.cdn.hf.co`, `cas-server.xethub.hf.co`), which is now
  required for virtually all HF file (model/dataset) downloads, platform-wide. This blocks real
  progress on this project AND likely any future project that needs HF-hosted weights/datasets.
  Asked Pranav in questions.md (Q-20261005-1). A parallel scout also resolved PLAN.md's Step 0 (found
  the real GAW-PO Eq. 8 via arxiv.org/html/2610.01511 full text) and confirmed the novelty claim holds
  against TDPO/SePO/TIS-DPO/Gradient-Entanglement prior work — written into PLAN.md's new "Status
  update" section.
- Next 3 steps:
  1. **Check questions.md for Q-20261005-1's answer first.** If Pranav added the Xet CDN domains to
     the allow-list (or pointed to an alternate channel): re-run `bash fetch_data.sh` in
     `projects/dpo-grad-reweight/`, then proceed with PLAN.md Steps 3 (mandatory timed trial) and 4
     (13-model projection) exactly as specified.
  2. If still unanswered/blocked after 48h (default in the question): pivot — either start a second
     project from backlog.md that doesn't need HF file downloads (e.g. the "Eval rank stability
     self-audit" P0 item, which is pure inference on data obtainable without large downloads if a
     small enough already-cached/no-download model is used, or an arXiv/HN-data-only analysis), while
     leaving dpo-grad-reweight's code ready to resume instantly.
  3. Before any real grid run: decide whether to write PLAN.md rev 3 adopting the REAL GAW-PO Eq. 8
     (dot-product + sigmoid + paired/global-max terms) instead of the rev-2 cosine-similarity fallback
     now that the real formula is known — needs a closer read of the paper's notation for the "global"
     term's exact scope, then back through overseer + ethics-reviewer before use. Keep radar.md fresh
     (daily mandate, last full refresh 2026-10-05).
- Blockers: **Q-20261005-1 (OPEN)** — HF Xet CDN domains not on network allow-list, blocks all
  HF model/dataset file downloads lab-wide. Everything past PLAN.md Step 2 for this project is paused
  on this.
- Resume commands: none running/checkpointed — nothing left mid-run. `projects/dpo-grad-reweight/.venv`
  already has torch/transformers/datasets/accelerate/huggingface_hub installed (exact pins in
  requirements.txt); once unblocked, just `cd projects/dpo-grad-reweight && bash fetch_data.sh` then
  run `src/train_dpo.py --mode full ...` per PLAN.md Step 3.
