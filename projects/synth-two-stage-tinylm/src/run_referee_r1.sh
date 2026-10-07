#!/usr/bin/env bash
# Referee round-1 ablation at N=4M (same conventions as run_grid.sh: skip if results/grid/<run>/result.json exists,
# resumable ckpt in ckpt/grid/<run>, result.json copied last). Runs in order:
#   S-R-perphase_N4M_s{0,1,2}: plans/A_S-R_N4M.json, lr 3e-3, --lr_schedule per_phase
#   {mixed,S-R,real-only-2ep}-lr1e-3_N4M_s0: lr 1e-3, global schedule
# Stops launching when elapsed + est run time (~330 s) > BUDGET_S. Usage: BUDGET_S=1950 bash src/run_referee_r1.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
BUDGET_S="${BUDGET_S:-1950}"; EST=330
LOG="results/grid/runner_log.txt"; mkdir -p results/grid ckpt/grid
start=$(date +%s); echo "# referee_r1 start $(date -Is) budget ${BUDGET_S}s" >> "$LOG"
RUNS="S-R-perphase_N4M_s0:S-R:0:3e-3:per_phase S-R-perphase_N4M_s1:S-R:1:3e-3:per_phase S-R-perphase_N4M_s2:S-R:2:3e-3:per_phase
mixed-lr1e-3_N4M_s0:mixed:0:1e-3:global S-R-lr1e-3_N4M_s0:S-R:0:1e-3:global real-only-2ep-lr1e-3_N4M_s0:real-only-2ep:0:1e-3:global"
for spec in $RUNS; do
  IFS=: read -r run arm seed lr sched <<< "$spec"; out="results/grid/$run"
  [ -f "$out/result.json" ] && continue
  el=$(( $(date +%s) - start ))
  if [ $(( el + EST )) -gt "$BUDGET_S" ]; then echo "budget stop before $run (elapsed ${el}s)" | tee -a "$LOG"; exit 0; fi
  t0=$(date +%s)
  timeout 1200 .venv/bin/python -I src/train.py --plan "plans/A_${arm}_N4M.json" --ckpt "ckpt/grid/$run" \
    --seed "$seed" --lr "$lr" --lr_schedule "$sched" --log_every 32 --eval R_val=data/R_val.bin R_dev=data/R_dev.bin \
    > "ckpt/grid/$run.stdout" 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && [ -f "ckpt/grid/$run/result.json" ]; then
    mkdir -p "$out"; cp "ckpt/grid/$run/train_log.jsonl" "ckpt/grid/$run"/eval_*_perdoc.npz "$out/"
    cp "ckpt/grid/$run/result.json" "$out/result.json"; echo "$run ok ${dt}s" | tee -a "$LOG"
  else echo "$run FAILED rc=$rc ${dt}s" | tee -a "$LOG"; tail -5 "ckpt/grid/$run.stdout"; exit 1; fi
done
echo "referee_r1 complete" | tee -a "$LOG"
