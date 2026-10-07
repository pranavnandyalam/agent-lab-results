#!/usr/bin/env bash
# Family B matched-total-token real-only arm (real-only-2ep: exactly 2N tokens = the same N real tokens x 2 epochs, no cap; plans from src/make_plans.py). Same conventions as run_grid.sh.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=results/grid/runner_log.txt; arm=real-only-2ep
for N in 1M 4M; do for seed in 0 1 2; do
  run="${arm}_N${N}_s${seed}"; out="results/grid/$run"; [ -f "$out/result.json" ] && continue
  timeout 1800 .venv/bin/python -I src/train.py --plan "plans/A_${arm}_N${N}.json" --ckpt "ckpt/grid/$run" --seed "$seed" --lr 3e-3 --log_every 32 --eval R_val=data/R_val.bin R_dev=data/R_dev.bin > "ckpt/grid/$run.stdout" 2>&1
  if [ -f "ckpt/grid/$run/result.json" ]; then mkdir -p "$out"; cp ckpt/grid/$run/train_log.jsonl ckpt/grid/$run/eval_*_perdoc.npz "$out/"; cp ckpt/grid/$run/result.json "$out/result.json"; echo "$run ok" | tee -a "$LOG"; else echo "$run FAILED" | tee -a "$LOG"; exit 1; fi
done; done
