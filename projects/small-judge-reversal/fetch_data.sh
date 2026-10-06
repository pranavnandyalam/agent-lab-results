#!/usr/bin/env bash
# Downloads the 4 core models + RewardBench (pinned) into ~/models/hf_cache; writes results/models.md
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="${HF_HOME:-$HOME/models/hf_cache}"
df -h /
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 25 ]; then echo "Need >=25GB free (keeps >=10GB after ~11GB download); have ${AVAIL_GB}GB"; exit 1; fi
timeout 3600 "$HERE/.venv/bin/python" -I "$HERE/src/fetch.py" "$HERE/results/models.md"
df -h /
