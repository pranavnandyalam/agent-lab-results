# STATE (loop: main)

## Cycle 13 summary (2026-10-06)
- small-judge-reversal analysis DONE: 0/0/0/1 of 300 pairs correctly reversed (4 judges); H2 supported, H1 inconclusive (n_cond<30). RESULTS.md + README written, src/analyze.py, results/analysis.md. Overseer APPROVE, ethics APPROVE_WITH_CONDITIONS (met). Project DONE.
- Next: apply review fixes, mark DONE; then pick next P0 from backlog (novelty check first). No active project after this. Nothing running.

## Cycle 12 summary (2026-10-06)
- SWEEP COMPLETE: all 4 models x 3 resamples raw JSONs exist (Qwen2.5-1.5B r2 finished after the shell timeout notice; overseer APPROVE).
- Then: analysis script (H1/H2, see PLAN.md and overseer notes below), RESULTS, README, overseer/ethics.


## Cycle 11 summary (2026-10-06)
- small-judge-reversal sweep: Qwen3-1.7B r2 and Qwen2.5-1.5B r0 DONE. Remaining: Qwen2.5-1.5B r1, r2 (~25-30 min each; same command as cycle 7 step 1, skip finished JSONs, timeout 1700).
- Then: analysis, RESULTS, README, overseer/ethics. Nothing running.

## Cycle 10 summary (2026-10-06)
- Q-20261006-1/-2 already marked ANSWERED (standing CPU approval).
- Sweep: Qwen3-1.7B r0,r1 DONE (~25 min each). Remaining: Qwen3-1.7B r2, Qwen2.5-1.5B r0-r2 (same command as cycle 7 step 1; skip finished JSONs; use timeout 1500 per run, ~25 min each).
- Then: analysis, RESULTS, README, overseer/ethics.


## Cycle 9 summary (2026-10-06)
- Sweep: Qwen3-0.6B r0,r1,r2 DONE (~4-11 min each). Remaining: Qwen3-1.7B, Qwen2.5-1.5B x 3 resamples (each ~25 min?, one process at a time, skip finished JSONs; same command as cycle 7 step 1).
- fetch.py now pins revisions, pyarrow added (overseer APPROVE). Novelty re-check done (scout): intact but narrower (2609.32407 closest; see PLAN.md). 0.5B dev A/B mass 0.9998 >= 0.5, so it stays in H1/H2 (not excluded).
- Q-20261006-1 still OPEN (default proceed 2026-10-08 evening).
- Next: finish sweep (1.7B, 1.5B), analysis, RESULTS, README, overseer/ethics. Write README early.

## Cycle 8 summary (2026-10-06)
- Owner request done: `team: main` added to small-judge-reversal PLAN.md; CLAIMS.md created and pushed.
- Sweep progress: Qwen2.5-0.5B-Instruct resamples 0,1,2 DONE (results/raw_*.json, ~9 min each). Remaining: Qwen2.5-1.5B, Qwen3-0.6B, Qwen3-1.7B x 3 resamples (same command as cycle 7 step 1; skip finished JSONs).
- Q-20261006-1 still OPEN (default proceed 2026-10-08 evening).
- Then: analysis, RESULTS, README, overseer/ethics.


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
- Overseer notes for analysis: apply mass_AB threshold/report format compliance (Qwen3-1.7B r2: 102/400 mass_AB<0.5); Qwen2.5-1.5B strongly B-biased; update results/commands.md; note torch cu130 env on 0.5B r0.
