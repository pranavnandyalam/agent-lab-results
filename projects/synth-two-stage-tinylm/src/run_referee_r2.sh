#!/usr/bin/env bash
# Referee round-2 runs (same conventions as run_referee_r1.sh, but outputs in results/referee_r2/<run>/ and
# ckpt/referee_r2/<run>; skip if result.json exists; resumable from last.pt; result.json copied last).
# Priority order (one job at a time; stops launching when elapsed + EST > BUDGET_S):
#  A1 constant LR (5% warmup then flat 3e-3, --lr_const), N=4M: S-R-constLR / mixed-constLR, seeds interleaved 0,1,2
#  B  LR sensitivity N=4M, global schedule: lr 6e-3 seed 0 for mixed, S-R, real-only-2ep; then lr 1e-3 seeds 1,2
#     (lr 1e-3 seed 0 already in results/grid/*-lr1e-3_N4M_s0)
#  C  N=1M per-phase S-R (H2-primary), seeds 0,1,2
# Usage: BUDGET_S=1950 bash src/run_referee_r2.sh   (re-run the same command to resume)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
BUDGET_S="${BUDGET_S:-1950}"
OUT=results/referee_r2; CK=ckpt/referee_r2; LOG="$OUT/runner_log.txt"; mkdir -p "$OUT" "$CK"
start=$(date +%s); echo "# referee_r2 start $(date -Is) budget ${BUDGET_S}s" >> "$LOG"
# run:arm:N:seed:lr:sched:const:est_seconds
RUNS="S-R-constLR_N4M_s0:S-R:4M:0:3e-3:global:1:300 mixed-constLR_N4M_s0:mixed:4M:0:3e-3:global:1:300
S-R-constLR_N4M_s1:S-R:4M:1:3e-3:global:1:300 mixed-constLR_N4M_s1:mixed:4M:1:3e-3:global:1:300
S-R-constLR_N4M_s2:S-R:4M:2:3e-3:global:1:300 mixed-constLR_N4M_s2:mixed:4M:2:3e-3:global:1:300
mixed-lr6e-3_N4M_s0:mixed:4M:0:6e-3:global:0:300 S-R-lr6e-3_N4M_s0:S-R:4M:0:6e-3:global:0:300
real-only-2ep-lr6e-3_N4M_s0:real-only-2ep:4M:0:6e-3:global:0:300
mixed-lr1e-3_N4M_s1:mixed:4M:1:1e-3:global:0:300 S-R-lr1e-3_N4M_s1:S-R:4M:1:1e-3:global:0:300
real-only-2ep-lr1e-3_N4M_s1:real-only-2ep:4M:1:1e-3:global:0:300
mixed-lr1e-3_N4M_s2:mixed:4M:2:1e-3:global:0:300 S-R-lr1e-3_N4M_s2:S-R:4M:2:1e-3:global:0:300
real-only-2ep-lr1e-3_N4M_s2:real-only-2ep:4M:2:1e-3:global:0:300
S-R-perphase_N1M_s0:S-R:1M:0:3e-3:per_phase:0:90 S-R-perphase_N1M_s1:S-R:1M:1:3e-3:per_phase:0:90
S-R-perphase_N1M_s2:S-R:1M:2:3e-3:per_phase:0:90"
for spec in $RUNS; do
  IFS=: read -r run arm N seed lr sched const est <<< "$spec"; out="$OUT/$run"
  [ -f "$out/result.json" ] && continue
  el=$(( $(date +%s) - start ))
  if [ $(( el + est )) -gt "$BUDGET_S" ]; then echo "budget stop before $run (elapsed ${el}s, est ${est}s)" | tee -a "$LOG"; exit 0; fi
  extra=""; [ "$const" = 1 ] && extra="--lr_const"
  t0=$(date +%s)
  timeout 1200 .venv/bin/python -I src/train.py --plan "plans/A_${arm}_N${N}.json" --ckpt "$CK/$run" \
    --seed "$seed" --lr "$lr" --lr_schedule "$sched" $extra --log_every 32 --eval R_val=data/R_val.bin R_dev=data/R_dev.bin \
    > "$CK/$run.stdout" 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && [ -f "$CK/$run/result.json" ]; then
    mkdir -p "$out"; cp "$CK/$run/train_log.jsonl" "$CK/$run"/eval_*_perdoc.npz "$out/"
    cp "$CK/$run/result.json" "$out/result.json"; echo "$run ok ${dt}s" | tee -a "$LOG"
  else echo "$run FAILED rc=$rc ${dt}s" | tee -a "$LOG"; tail -5 "$CK/$run.stdout"; exit 1; fi
done
echo "referee_r2 complete" | tee -a "$LOG"
