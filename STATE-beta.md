# STATE (loop: beta)

## Cycle 1 (2026-10-06)
- Active: quant-cot-looping (claimed in CLAIMS.md). PLAN rev 2: overseer REJECT (rev 1: 5 issues; rev 2: compute + citations). Ethics APPROVE_WITH_CONDITIONS (disclosures, limitations, freeze detector before test, pin revisions).
- Done: lit pass (closest: 2606.25519, 2606.02011, 2609.26708), venv + Qwen3-0.6B + GSM8K pinned, fake-quant (RTN) module, decode trial: fp32 61 tok/s @b12, 78 @b24 (256 tok only). 3-bit g64 drops <think>, 2-bit garbage.
- Q-20261006-2 OPEN (2-3 h CPU). Needs recomputing: slot-steps (batch runs to longest trace), est. up to ~4 h at b12; rewrite numbers before it matters.
- Next 3 steps: (1) PLAN rev 3 from overseer fixes: slot-step compute from a full-length w4 trial (b24), correct 2609.26708 (RTN 2.79/1.88-bit, QAD models, MATH-500) and cite 2606.02011 Table 1/App. B (cross-setting correlation vs our per-trace AUROC + token decomposition), exact H1 formula w/ bootstrap CI, H2 decision rule (>=15 paired), w3g32 vs g64 confound + effective bits, no-think-block extraction, hardcode revisions in fetch_data.sh, top_k=20, shorten "What's new" to 2 sentences; update Q-2 numbers; re-review. (2) detector dev on train problems, freeze, commit. (3) core run in chunks.
- Blockers: none. Nothing running. venv 5.4 GB (CUDA torch pulled), in projects/quant-cot-looping/.venv (gitignored).
