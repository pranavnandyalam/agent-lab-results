#!/usr/bin/env bash
# Core run: 4 levels x 2 problem chunks = 8 batches of 24 generations, run order per PLAN (w4g64, fp32, w3g32, w5g64).
# Resumable: core_run.py skips a batch whose JSON exists. Wall time per batch appended to results/core/timings.txt.
# Usage (from anywhere): bash projects/quant-cot-looping/src/run_core.sh
set -u
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$P/results/core"
LOG="$P/results/core/timings.txt"
for spec in "w4g64 4 64" "fp32 16 64" "w3g32 3 32" "w5g64 5 64"; do
  read -r tag bits group <<< "$spec"
  for k in 0 1; do
    if [ -e "$P/results/core/${tag}_chunk${k}.json" ]; then echo "skip ${tag}_chunk${k}"; continue; fi
    s=$(date +%s)
    timeout 2400 "$P/.venv/bin/python" -I "$P/src/core_run.py" --bits "$bits" --group "$group" --tag "$tag" --problems-chunk "$k"
    rc=$?
    e=$(date +%s)
    echo "$(date -u +%FT%TZ) ${tag}_chunk${k} wall_s=$((e - s)) rc=$rc" >> "$LOG"
  done
done
