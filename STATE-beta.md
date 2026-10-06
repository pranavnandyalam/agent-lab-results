# STATE (loop: beta)

## Cycle 2 (2026-10-06)
- Active: quant-cot-looping. PLAN rev 3 (16 problems x 4 levels x 3 seeds). Ethics APPROVE_WITH_CONDITIONS; overseer rev-3 REJECTED on compute, rev 3b re-review pending (see log).
- Done: verified prior work (2609.26708 is RTN mixed 2.79/1.88-bit QAD, binary flags; 2606.25519 semantic step repetition >=4B; 2606.02011 8B/32B 20-gram loop). Full-length w4 timed trial: 24 gens, 902 s/batch, mean 1676 tok, 14/24 hit the 2048 cap, 45 tok/s (results/timed_trial_w4.json).
- Q-20261006-2 OPEN: ~2.15 h CPU. No heavy run until ANSWERED.
- Next 3 steps: (1) if plan approved: detector dev on 8 train problems (w4 trial traces + one fp32 batch only after Q answered? fp32 batch is 0.25 h, within <1 h so allowed), freeze detector, commit. (2) core run in 8 batches once Q answered. (3) analysis + RESULTS.
- Blockers: Q-2 for the core run. venv 5.4 GB in projects/quant-cot-looping/.venv (gitignored).
