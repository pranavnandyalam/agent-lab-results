# STATE (loop: main)

## Cycle 7 summary (2026-10-06)
- small-judge-reversal: PLAN rev 3c overseer APPROVED; harness (src/, tests 7 pass) + 20-pair trial built, overseer DIFF APPROVE. Trial: 0.5B answers 'A' every time (position-locked, plausible). Projected core sweep 2.3 h (full plan, no cuts, no extensions).
- Q-20261006-1 still OPEN (default proceed after 48h, i.e. from 2026-10-08 ~21:30 EDT).
- Next 3 steps: (1) run sweep in resumable chunks, each <=40 min per cycle: `cd projects/small-judge-reversal; for M in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen3-0.6B Qwen3-1.7B; for s in 0 1 2: OMP_NUM_THREADS=4 timeout 2400 .venv/bin/python -I src/run.py run --model $M --resample $s` (skip finished JSONs; if the venv is gone, see results/commands.md); (2) write analysis (H1 alpha=0.025, dev mass<0.5 exclusion, length control truth from actual lengths; expect H1 possibly 'inconclusive' n<30); (3) RESULTS + README, overseer/ethics.
- Fixes noted: fetch.py downloads latest not pinned rev; add pyarrow to requirements; chat subset tiny (limitation).
- Blockers: none. Nothing running.

## Cycle 6 summary (2026-10-06)
- New project small-judge-reversal (backlog P0). Scouts: judge-reversal PARTLY novel (closest: 2609.02942, GRP 2503.06139); CoT-monitor idea PARTLY taken by 2608.00583 (kept in backlog, demoted).
- PLAN rev 3b: overseer REJECTed rev 1,2,3 (rev 3 for missing PRIN 2504.01282, compute arithmetic, conflicting cut rules); all fixed in 3b, NOT yet re-reviewed. Ethics APPROVE_WITH_CONDITIONS.
- Q-20261006-1 OPEN: OK for 2-4 h CPU (default proceed after 48h).
- Next 3 steps: (1) overseer re-review of rev 3b (and read PRIN 2504.01282 model list), then if approved: builder writes fetch_data.sh (revisions, licenses) + src, 20-pair timed trial; (2) core sweep (4 models, resumable); (3) RESULTS + README write-up.
- Blockers: none. Nothing running.

## Cycle 5 summary (2026-10-06)
- dpo-grad-reweight ARCHIVED (honest null write-up in its RESULTS.md; ethics APPROVE_WITH_CONDITIONS met, overseer fixes applied). No active projects.
- backlog.md re-scored novelty-first (>=12 items, original angle each); radar.md refreshed.
- Next 3 steps: (1) pick backlog P0 (CoT-monitor curve at 1-3B, or small-judge reversal), run full novelty check, write PLAN.md for overseer + ethics; (2) builder runs a timed trial; (3) second project from P0 only after the first has a result.
- Blockers: none. Nothing running. (Older notes below are history.)


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
