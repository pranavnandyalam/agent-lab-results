# Commands (run from the project directory /home/agent/agent-lab/projects/small-judge-reversal; all paths are relative to it)
uv venv .venv --python 3.14 && uv pip install --python .venv/bin/python -r requirements.txt
timeout 3700 ./fetch_data.sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1200 .venv/bin/python -I src/run.py prepare
.venv/bin/python -m pytest -q -p no:cacheprovider tests
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1200 .venv/bin/python -I src/run.py trial --model Qwen2.5-0.5B-Instruct --n 20
OMP_NUM_THREADS=4 timeout 600 .venv/bin/python -I src/project.py
# Full sweep, run in resumable chunks over cycles 8-12 (finished JSONs skipped; timeouts 1500-2400 s per run;
# Qwen2.5-1.5B r2 passed the shell timeout notice but its output file completed):
#   for M in 4 core models, s in 0 1 2: OMP_NUM_THREADS=4 timeout 2400 .venv/bin/python -I src/run.py run --model $M --resample $s
OMP_NUM_THREADS=2 .venv/bin/python -I src/analyze.py   # -> results/analysis.{json,md}

## Calibrated re-analysis (cycle 19)
`.venv/bin/python -I src/calibrated.py` -> results/calibrated.{json,md} (no model runs; reads raw_*.json).

## Cycle 23: independence null, param counts, non-code pair set
.venv/bin/python -I src/param_count.py --write          # param counts from cached safetensors headers -> results/models.{md,json}
OMP_NUM_THREADS=4 timeout 900 .venv/bin/python -I src/analyze.py      # now also writes optional_models (Qwen3-4B) if raw present
OMP_NUM_THREADS=4 timeout 900 .venv/bin/python -I src/calibrated.py   # + independence null; -> calibrated.{json,md}, calibrated_null.json
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1200 .venv/bin/python -I src/run.py prepare_noncode          # -> results/splits_noncode.json
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1200 .venv/bin/python -I src/run.py trial --pairs noncode --model Qwen3-0.6B --n 10   # -> results/trial_nc_Qwen3-0.6B.json
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1500 .venv/bin/python -I src/run.py run --pairs noncode --model Qwen3-0.6B --resample 1   # -> results/raw_nc_Qwen3-0.6B_r1.json
# remaining non-code passes (resumable; rerun same command after a timeout): M in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen3-1.7B [Qwen3-4B]
#   OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 2400 .venv/bin/python -I src/run.py run --pairs noncode --model $M --resample 1
