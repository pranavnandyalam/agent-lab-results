#!/usr/bin/env bash
# Downloads the 4 core models + RewardBench (pinned) into ~/models/hf_cache; writes results/models.md/.json
# Optional: FETCH_4B=1 ./fetch_data.sh also fetches the Qwen3-4B positive-control judge (pinned rev 1cfa9a72..., ~8 GB).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="${HF_HOME:-$HOME/models/hf_cache}"
df -h /
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
NEED=25; [ "${FETCH_4B:-0}" = "1" ] && NEED=33   # +~8 GB for Qwen3-4B, still keeps >=10 GB free
if [ "$AVAIL_GB" -lt "$NEED" ]; then echo "Need >=${NEED}GB free (keeps >=10GB after download); have ${AVAIL_GB}GB"; exit 1; fi
timeout 3600 "$HERE/.venv/bin/python" -I "$HERE/src/fetch.py" "$HERE/results/models.md"
timeout 120 "$HERE/.venv/bin/python" -I "$HERE/src/param_count.py" --write > /dev/null   # param counts of cached models
df -h /
