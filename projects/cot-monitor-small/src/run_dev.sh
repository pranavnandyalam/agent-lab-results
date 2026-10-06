#!/bin/bash
# Dev prompt selection: CoT-only dev passes, 3 prompt variants x 4 monitors.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PY:-.venv/bin/python}  # quant-cot-looping venv was reused via PY=... (same pinned torch/transformers)
for m in qwen2.5-0.5b qwen3-0.6b qwen2.5-1.5b qwen3-1.7b; do for p in v1 v2 v3; do
  timeout 900 $PY -I src/score.py --monitor $m --prompt $p --split dev --views cot 2>&1 | grep -v -i warn
done; done
