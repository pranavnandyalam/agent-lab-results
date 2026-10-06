#!/usr/bin/env bash
# Family A grid: order N=1M then N=4M; within N, seed 0,1,2; within seed, arms in fixed order. LR 3e-3 (R_dev, results/lr_dev.json).
# Skips a run if results/grid/<arm>_N<N>_s<seed>/result.json exists. Checkpoints in ckpt/grid/<run> (gitignored; train.py
# resumes from last.pt). Stops launching new runs when elapsed + estimated run time > BUDGET_S (default 2100 s), so
# it stops cleanly between runs. Usage (from anywhere): BUDGET_S=2100 bash src/run_grid.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
BUDGET_S="${BUDGET_S:-2100}"
TOKS="${TOKS_PER_S:-30000}"   # throughput estimate for the launch check only
LR=3e-3
ARMS="real-only mixed S-R R-S mixed-realtail"
declare -A NTOK=([1M]=1048576 [4M]=4194304)
declare -A MULT=([real-only]=1 [mixed]=2 [S-R]=2 [R-S]=2 [mixed-realtail]=3)
LOG="results/grid/runner_log.txt"
mkdir -p results/grid ckpt/grid
start=$(date +%s)
echo "# start $(date -Is) budget ${BUDGET_S}s" >> "$LOG"
for N in 1M 4M; do for seed in 0 1 2; do for arm in $ARMS; do
  run="${arm}_N${N}_s${seed}"; out="results/grid/$run"
  [ -f "$out/result.json" ] && continue
  est=$(( NTOK[$N] * MULT[$arm] / TOKS + 30 ))
  el=$(( $(date +%s) - start ))
  if [ $(( el + est )) -gt "$BUDGET_S" ]; then
    echo "budget stop before $run (elapsed ${el}s, est ${est}s)" | tee -a "$LOG"; exit 0
  fi
  t0=$(date +%s)
  timeout 1800 .venv/bin/python -I src/train.py --plan "plans/A_${arm}_N${N}.json" --ckpt "ckpt/grid/$run" \
    --seed "$seed" --lr "$LR" --log_every 32 --eval R_val=data/R_val.bin R_dev=data/R_dev.bin \
    > "ckpt/grid/$run.stdout" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && [ -f "ckpt/grid/$run/result.json" ]; then
    mkdir -p "$out"
    cp "ckpt/grid/$run/train_log.jsonl" "ckpt/grid/$run"/eval_*_perdoc.npz "$out/"
    cp "ckpt/grid/$run/result.json" "$out/result.json"   # copied last: its presence marks completion
    echo "$run ok ${dt}s" | tee -a "$LOG"
  else
    echo "$run FAILED rc=$rc ${dt}s (see ckpt/grid/$run.stdout)" | tee -a "$LOG"; tail -5 "ckpt/grid/$run.stdout"; exit 1
  fi
done; done; done
echo "grid complete" | tee -a "$LOG"
