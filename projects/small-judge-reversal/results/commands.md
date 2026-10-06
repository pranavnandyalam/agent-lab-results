# Commands (run from /tmp; project = /home/agent/agent-lab/projects/small-judge-reversal)
uv venv .venv --python 3.14 && uv pip install --python .venv/bin/python -r requirements.txt
timeout 3700 ./fetch_data.sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1200 .venv/bin/python -I src/run.py prepare
.venv/bin/python -m pytest -q -p no:cacheprovider tests
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1200 .venv/bin/python -I src/run.py trial --model Qwen2.5-0.5B-Instruct --n 20
OMP_NUM_THREADS=4 timeout 600 .venv/bin/python -I src/project.py
# Full sweep (NOT run yet): for M in 4 core models, s in 0 1 2:
#   OMP_NUM_THREADS=4 timeout 7200 .venv/bin/python -I src/run.py run --model $M --resample $s
