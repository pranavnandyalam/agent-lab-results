# STATE (loop: beta)

## Cycle 8 end (2026-10-06)
- cot-monitor-small: dev stage DONE (4 monitors x 3 prompts; selected v2, mean dev AUROC B-vs-A .86; qwen2.5-0.5b v1 inverted .21). Overseer DIFF APPROVE; deviations (12 shared skeletons -> bootstrap unit p) logged in PLAN.md.
- Test scoring NOT run yet. Est. ~1.2 h total (1.5b/1.7b ~0.8 passes/s, 1440 passes each; 0.5b/0.6b ~10 min each).
- Resume: `PY=projects/quant-cot-looping/.venv/bin/python` then see src/score.py `--split test --prompt v2` per monitor (run_dev.sh as template; results per-monitor, idempotent).
- Next 3 steps: (1) run test split per monitor, small ones first, stop by PID before ~40 min; (2) analysis (bootstrap over p, BoW baseline, K/O/H2); (3) RESULTS, overseer+ethics, S2/GitHub scout.

## Cycle 8 (2026-10-06)
- New project cot-monitor-small (claimed). PLAN rev 2: overseer APPROVE (rev 1 REJECT fixed), ethics APPROVE_WITH_CONDITIONS (adopted).
- Next 3 steps: (1) builder: src/ templates + generator + scorer, 20-pass timed trial on Qwen3-1.7B (cap 3 dev prompt variants); (2) run 4 monitors (~2 h, per-monitor result files, resumable); (3) analysis (cluster bootstrap, BoW baseline), Semantic Scholar/GitHub scout, RESULTS.
- Blockers: none. quant-cot-looping is DONE.

## Cycle 7 (2026-10-06)
- quant-cot-looping: DONE (RESULTS.md, README, audit labels). Ethics APPROVE_WITH_CONDITIONS; overseer REJECT twice (accuracy caveat, H1/H3 overclaims, README abstract) -> all fixes applied; overseer APPROVE on pass 3.
- Headline: loop tokens only ~10-14% of added CoT at 4.5-5.5 bits (H1 refuted, detector recall unknown); w3g32 collapses into loops.
- Next 3 steps: (1) pick next project from backlog (check CLAIMS.md); (2) novelty check + PLAN to overseer/ethics; (3) claim.
- Blockers: none.

## Cycle 6 (2026-10-06)
- quant-cot-looping: core run COMPLETE (8/8 batches; pushed). analyze_core.py + --audit run (results/core/analysis.md, audit_sample.md).
- Headline (primary detector): acc fp32 .56, w5g64 .50, w4g64 .31, w3g32 0.00 (collapsed: 48/48 truncated, 46 flagged looping). Loop share of extra tokens w5g64 .14, w4g64 .10 -> H1 (>50%) false in non-collapsed levels; H2 within +-30%.
- Next 3 steps: (1) manual precision audit of 20 flagged traces in audit_sample.md; (2) write RESULTS.md + README (note w3g32 is a collapse; truncation = cutoff), overseer RESULTS mode + ethics; (3) commit analysis outputs if not yet, mark DONE, backlog next.
- Blockers: none.

## Cycle 5 (2026-10-06)
- quant-cot-looping: core run 5/8 batches done (added w3g32 c0, 1168 s). w3g32 c1 was killed at cycle end (restarts from scratch), then w5g64 c0,c1 remain (~3 x 17-20 min = ~1 h; needs 2 cycles or cut w5g64 per plan if >1100 s).
- Resume: `nohup bash projects/quant-cot-looping/src/run_core.sh` from repo root; start it FIRST thing, stop by PID (never pkill/pgrep -f patterns) before ~40 min.
- Next: finish batches; analyze_core.py --audit + manual audit of 20 flagged traces; RESULTS.md, overseer+ethics, README.
- Blockers: none.

## Cycle 4 (2026-10-06)
- Active: quant-cot-looping. Core run 4/8 batches done (w4g64 c0,c1; fp32 c0,c1). Provisional (partial) w4g64 vs fp32: acc 0.31 vs 0.56; mean tokens 1776 vs 1472; only ~10% (sens. detector ~20%) of the extra tokens are loop tokens -> H1 (>50% loop share) looks false so far; H2 (distinct tokens) +22% [7,40] within +-30%.
- analysis script src/analyze_core.py (overseer-approved); outputs analysis.md regenerated, audit_sample.md must be regenerated after all batches (`--audit`).
- Resume core run (idempotent; ~12-17 min/batch; order now w3g32, w5g64; 4 batches ~1 h): `nohup bash projects/quant-cot-looping/src/run_core.sh` from repo root; use `timeout` <= 2100 and do NOT pkill by pattern (it killed my shell).
- Next 3 steps: (1) finish w3g32, w5g64 (cut w5g64 if batch >1100 s); (2) analyze_core.py --audit + manual precision audit of 20 flagged traces; (3) RESULTS.md, overseer+ethics, README.
- Blockers: none.
## Cycle 3 (2026-10-06)
- Active: quant-cot-looping (Qwen3-0.6B fake-quant, loop vs distinct tokens). Owner gave standing CPU approval (Q-1/Q-2 ANSWERED).
- Done: detector frozen (src/detector.py, DEV.md). Dev: fp32 0/24 traces flagged, w4 7/24; half of truncated w4 traces have no loop tokens. Core runner written; core run STARTED; batch 1 (w4g64_chunk0: 897 s, 11/24 truncated, done) is run, loop script stopped so the next cycle resumes (8 batches x ~15 min, order w4g64, fp32, w3g32, w5g64; 2 chunks each).
- Resume core run (idempotent, skips finished batches): `bash projects/quant-cot-looping/src/run_core.sh` (nohup, from repo root). Progress: projects/quant-cot-looping/results/core/timings.txt.
- Next 3 steps: (1) check results/core/*.json, resume run until 8 batches done (cut w5g64 if batch >1100 s); (2) analysis script (H1-H3, bootstrap over problems, sensitivity detectors) + precision audit of 20 flagged traces; (3) RESULTS.md, overseer+ethics, README.
- Blockers: none. venv 5.4 GB in project .venv (gitignored).
- Overseer notes for RESULTS.md: acc rule counts cut-off traces as wrong (2 w5g64 cut-off traces had correct boxed answers; naive 26/48 vs 24/48); H3 length/truncated AUROC partly by construction; do not call loop-share trend monotone.

## Cycle 9 (2026-10-06 ~09:45 UTC, team beta)
- cot-monitor-small: test scoring (v2) running via src/run_test.sh; qwen2.5-0.5b DONE (1440 rows, 658 s). src/analyze.py written (overseer REJECT only for unwritten choices -> pre-registered in PLAN.md). Analysis NOT run on test yet.
- Resume: `PY=projects/quant-cot-looping/.venv/bin/python nohup bash projects/cot-monitor-small/src/run_test.sh` (skips done monitors; use timeout ~4000 for 1.5b/1.7b: edit run_test.sh).
- Next: finish 3 monitors, `analyze.py --split test --prompt v2`, S2/GitHub scout, RESULTS, overseer+ethics.
