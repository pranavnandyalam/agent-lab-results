#!/bin/bash
# Referee round 1: test-split scoring of the small monitors under the non-selected dev prompts (v1, v3). Idempotent.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PY:-../quant-cot-looping/.venv/bin/python}
for p in v1 v3; do for m in qwen2.5-0.5b qwen3-0.6b; do
  timeout 900 $PY -I src/score.py --monitor $m --prompt $p --split test 2>&1 | grep -v -i warn
done; done
