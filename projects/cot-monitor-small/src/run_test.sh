#!/bin/bash
# Test-split scoring with the dev-selected prompt (v2), all views; small monitors first. Idempotent per monitor.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PY:-.venv/bin/python}
for m in qwen2.5-0.5b qwen3-0.6b qwen2.5-1.5b qwen3-1.7b; do
  timeout 2400 $PY -I src/score.py --monitor $m --prompt v2 --split test 2>&1 | grep -v -i warn
done
