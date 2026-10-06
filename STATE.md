# STATE (loop: main)

- Cycle counter: 4
- Active projects: dpo-grad-reweight (UNBLOCKED; PLAN rev 4b = trimmed baseline + original extension, overseer APPROVED; grid blocked on LR instability)
- Current focus: HF downloads work (Pranav allow-listed the Xet CDN). Step 3 timed trial done: 789 s/run at 100/80/200 sizes
  -> grid projects ~135 min, so trim to 50 train / 100 ARC (~83 min). Trial loss looked unstable (mean 1.41 > ln2).
  Builder is adding/running the pre-registered dev LR check (`--mode dev_lr`, 3 LRs on the 50 dev pairs).
  North-star now demands novelty: extension = what GAW-PO token weights track + does a cheap proxy match a true gradient weight.
- Dev LR check DONE (cycle 4): by the pre-registered rule NO lr of {1e-6,5e-6,1e-5} qualifies (mean online dev loss 0.80/1.18/2.36 vs ln2=0.69), so the grid is NOT run as designed. Likely cause: RMSprop at batch size 1.
- Next 3 steps:
  1. Write PLAN rev 5 (changes design, needs overseer review): gradient accumulation (batch 8-16) and/or lower lr, new pre-registered LR rule, chosen on the 50 dev pairs only (~10 min dev run). Then the trimmed 12-run grid (~83 min, resumable chunks).
  2. Run the approved rev-4b extension analysis (base model, dev+train pairs only, 3-pair timed trial first) via a builder: this does not depend on the LR problem and is the novelty part. Confirm NTHR's arXiv ID first (2505.18830 is a GRPO paper; unverified).
  3. Re-score backlog.md novelty-first with an original angle per item (north-star ask); refresh radar.md (last full refresh 2026-10-05).
- Blockers: none. (Q-20261005-1 answered A.)
- Resume commands: nothing running at cycle end. Grid not started. Dev check: `.venv/bin/python src/train_dpo.py --mode dev_lr` (~14 min). Once LR is fixed:
  `cd projects/dpo-grad-reweight && OMP_NUM_THREADS=4 timeout 1500 .venv/bin/python src/train_dpo.py --mode full --method <vanilla|gawpolite> --beta <0.1|0.01> --seed <0|1|2> --lr <chosen> --n_train 50 --n_arc 100 --out_json results/grid_<method>_b<beta>_s<seed>.json`
