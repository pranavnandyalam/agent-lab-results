# STATE (loop: main)

## Cycle 26 summary (2026-10-07)
- synth-two-stage-tinylm: paper revised per referee r1 (title softened, no 'grows with N', per-phase-LR control, 3 cites, RESPONSE.md); overseer PAPER APPROVE. Awaiting referee r2.
- NEXT: optional heavier-repetition (real-only 4/8 ep) + stronger G0 + N=2M experiments; small-judge-reversal non-code runs (see cycle 24 NEXT). Nothing running.

## Cycle 25 summary (2026-10-07, referee r1 on synth-two-stage-tinylm)
- Done: per-phase-LR S→R control (ordering survives, 15.96 vs mixed 17.39), LR1e-3 seed0 hint, plan-script fixes; RESULTS addendum.
- NEXT: (1) paper-writer: revise paper per addendum (drop "grows with N", retitle/soften ordering claim, "same-architecture generator", add 3 cites, say H.2 unread, venue) + paper/RESPONSE.md, overseer PAPER; (2) optional experiments: heavier repetition (real-only 4/8 ep vs S-R) and stronger G0, N=2M; (3) small-judge-reversal non-code runs (see cycle 24 NEXT below).

## Cycle 24 summary (2026-10-07, referee r4 on small-judge-reversal)
- small-judge-reversal: DONE this cycle (code/results only, paper NOT yet updated): independence null for calibrated reversal (4 small judges well BELOW null; Qwen3-4B 0.31 vs null 0.30 => no partial reversal, referee hole 2 right), single shared bootstrap (CIs shifted 0.01-0.03; paper CIs need refresh from results/calibrated.md), 4B params 4.02B recorded, fetch/analyze handle 4B, 200 non-code pairs built (results/splits_noncode.json), Qwen3-0.6B nc run done (reverse 0.055).
- NEXT: run non-code passes: `cd projects/small-judge-reversal; for M in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen3-1.7B: OMP_NUM_THREADS=4 timeout 2400 .venv/bin/python -I src/run.py run --pairs noncode --model $M --resample 1` (~9/32/34 min; 4B ~76 min, timeout 5400, resumable), then `src/calibrated.py`, then paper-writer update (drop size-trend sentences in abstract/conclusion, null, CIs, non-code, 4B params) + overseer PAPER. Also gen check Qwen2.5-1.5B.
- Q-20261007-2 OPEN: download Qwen2.5-7B (15 GB) for inverting-judge control (default skip; then abstract must state sensitivity unshown).
- synth-two-stage-tinylm: overseer RESULTS APPROVE after hash fix; paper written (5 pp), overseer PAPER review pending/see log.

## Cycle 23 summary (2026-10-07)
- small-judge-reversal: Qwen3-4B positive control DONE (n=100 pairs, 1 resample): corr(better,W1)=0.10, calibrated reversal 0.31 vs ~0 for 4 small judges; paper/RESULTS/README updated (8 pp), overseer PAPER+DIFF APPROVE. Awaiting referee on changed paper. Still open: non-code pairs, Qwen2.5-1.5B gen check, 4B r0/r2.
- synth-two-stage-tinylm: next = overseer RESULTS re-review, then paper-writer (unchanged). Nothing running.

## Cycle 22 summary (2026-10-07)
- Q-20261007-1 ANSWERED (Pranav OK'd Qwen3-4B). Downloaded Qwen/Qwen3-4B rev 1cfa9a72 (apache-2.0), added to src/sjr/config.py. Positive-control run started: `cd projects/small-judge-reversal && OMP_NUM_THREADS=4 timeout 2400 .venv/bin/python -I src/run.py run --model Qwen3-4B --resample 1` (resumable checkpoint in ~/scratch/small-judge-reversal/run_Qwen3-4B_r1.jsonl, 400 items, ~8 items/min). Cycle ended with it partial; RESUME NEXT CYCLE (same command; run in background tool, NOT nohup). Then calibrated.py with 4B, add to paper as positive control (overseer PAPER review). Still open: non-code pairs, Qwen2.5-1.5B gen check.
- synth-two-stage-tinylm: next = overseer RESULTS re-review, then paper-writer (unchanged).

## Cycle 21 summary (2026-10-07)
- small-judge-reversal referee r3: done = Qwen3-1.7B gen check (90% agree w/ logit; gen reversal 4/100), cite 2503.03064, abstract/typo fixes, RESPONSE.md. NOT done: (1) positive-control 4B judge (Q-20261007-1 OPEN, 8 GB download), (2) 200+ non-code pairs (reward-bench chat/chat-hard cached; run 4 judges, ~1 h), (3) gen check for Qwen2.5-1.5B (`.venv/bin/python -I src/gen_check.py gen --model Qwen2.5-1.5B-Instruct --n 100`, ~30 min, then `report`).
- synth-two-stage-tinylm: next = overseer RESULTS re-review, then paper-writer (unchanged).

## Cycle 20 summary (2026-10-07)
- synth-two-stage-tinylm: grid COMPLETE (3 seeds, N=1M,4M). RESULTS.md/README written (overseer REJECT on over-claiming finding 3 -> softened; ethics APPROVE w/ conditions, applied). 4M: S-R beats mixed by 1.27 ppl (H1 rule met), but 2N fresh real tokens beat S-R by 0.89. 1M: mixed/S-R == real-only-2ep (repeat real). **Family B 4M DONE: real-only-2ep 15.18 beats S-R by 0.94 ppl. Next: overseer RESULTS re-review of final RESULTS.md (numbers from results/analysis.md, edited post-approval), add commit hash, then paper-writer.**
- small-judge-reversal: referee round 2 addressed (size trend, magnitude, edit distance, length control, 3 new cites) and pushed; overseer PAPER APPROVE. Still open: generation-based readout check (`cd projects/small-judge-reversal && .venv/bin/python -I src/gen_check.py gen --model Qwen3-1.7B --n 100`, resumable, ~11/100 pairs done; src/gen_check.py + partial json untracked/uncommitted-safe) and larger positive-control judge (Qwen3-4B / Qwen2.5-7B bf16).

## Cycle 19 summary (2026-10-07)
- small-judge-reversal: referee round 1 addressed. Calibrated re-analysis shows criterion-blindness (r 0.63-0.95; 0.5B's 0.95 not meaningful); paper rewritten + RESPONSE.md, overseer PAPER APPROVE. Open (optional): generation-based readout check, 3-4B judge, thinking-on condition. Awaiting referee round 2.
- synth-two-stage-tinylm: N=4M seed2 partial (mixed-realtail_N4M_s2 missing; check results/grid). Next: finish seed 2, rerun analyze.py with overseer fixes, RESULTS/README. Nothing running.

## Cycle 18 summary (2026-10-06)
- synth-two-stage-tinylm: grid N=4M seed1 done (seed0,1 complete; seed2 remaining, ~25 min). src/analyze.py + results/analysis.md written (prelim). N=1M (n=3): mixed 44.6, S->R 44.4, R->S 45.6, real-only 77.6, mixed-realtail 32.6 (it sees 2N real tokens). S-R vs mixed gap tiny/seed-inconsistent at 1M; 4M seeds 0,1 favour S-R by ~1.3 ppl.
- Next: (1) `cd projects/synth-two-stage-tinylm && BUDGET_S=1800 bash src/run_grid.sh` (4M seed2); (2) rerun analyze.py, write RESULTS/README (note real-only is undertrained in Family A, N-scaling confound); overseer + ethics; (3) read 2609.09572 LM section; redo novelty with S2.
- Overseer to-fix in analyze.py: hierarchical bootstrap (seeds then docs), H1 contrast g(4M)-g(1M), n=2 rows labelled. Nothing running.

## Cycle 17 summary (2026-10-06)
- synth-two-stage-tinylm: S1 filtered, Family A plans + resumable grid runner (src/run_grid.sh). Done: N=1M all 5 arms x 3 seeds; N=4M seed0: real-only, mixed, S-R, R-S. No analysis yet.
- Next: (1) resume `cd projects/synth-two-stage-tinylm && BUDGET_S=1800 bash src/run_grid.sh` (skips finished runs; ~65 min left: 4M seeds 0-2 remaining, mixed-realtail slowest ~450 s); (2) write analysis script (per-arm R_val ppl mean±std, paired vs real-only, slices) + RESULTS/README; overseer + ethics.
- Nothing running at cycle end.

## Cycle 16 summary (2026-10-06)
- synth-two-stage-tinylm: sampler (KV cache, selftest pass), LR check on R_dev (3e-3 best of 1e-3/3e-3/6e-3; 1M-token runs undertrained, may not transfer), G0 trained 8.4M tok (R_val ppl 15.43, report only), S1 corpus 4.26M tok sampled (21k tok/s). 0.75% empty docs in S1 (decide: filter before grid).
- Next: (1) filter empty docs in S1, write arm-plan generator + run Family A grid in resumable chunks (N=1M,4M first; 3.4 h est total, cut N=2M); (2) check whether G0 should be fed LR retune at 8M; (3) analysis + RESULTS.
- Data in data/ (gitignored); S1.bin regenerable via results/commands_G0_S1.txt. Nothing running.


## Cycle 15 summary (2026-10-06)
- synth-two-stage-tinylm: pipeline built (fetch, BPE, GPT 0.79M non-emb, resumable trainer); timed trial 27.5k tok/s, full grid est 3.4 h (2.8 h without N=2M). Novelty scout: PARTLY TAKEN (Li&Zou ran a WikiText-103 LM test themselves); reframed as replication+extension (TinyStories, recency controls, N-scaling).
- Next: (1) read 2609.09572 LM section in full; (2) write sampling (KV cache) + G0 training + synthetic corpus; LR tune on R_dev; (3) run grid in resumable chunks. Data in data/ (gitignored; rerun ./fetch_data.sh, src/tok.py if gone). Nothing running.

## Cycle 14 summary (2026-10-06)
- Active: synth-two-stage-tinylm (PLAN rev 3 approved w/ conditions; claimed). Idea: test Li&Zou 2609.09572 (two-stage synthetic->real avoids error floor) on a tiny GPT. Scout: NOVEL (medium conf). Overseer REJECT x2 (design fixes applied in rev 3), ethics APPROVE_WITH_CONDITIONS.
- Next: (1) Semantic Scholar/GitHub novelty check; (2) [done: overseer APPROVE w/ conditions, applied]; (3) if approved, builder: fetch_data.sh + timed 1M-token trial. No code written yet. Nothing running.
- Backlog alternative: CoT-verbalization sub-4B (scout: NOVEL but ~days of CPU).

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
