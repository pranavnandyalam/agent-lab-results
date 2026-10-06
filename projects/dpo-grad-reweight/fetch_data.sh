#!/usr/bin/env bash
# fetch_data.sh — fetch model + dataset subset for dpo-grad-reweight, record
# license + revision hash in results/fetch_notes.md.
#
# Network: huggingface.co only. Cache lives outside the repo at ~/models/hf_cache
# (never commit weights/dataset files). Run from anywhere; paths are absolute.
#
# Usage: bash fetch_data.sh
set -euo pipefail

export HF_HOME="$HOME/models/hf_cache"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# The xet CAS backend is NOT on the network allow-list: both us.aws.cdn.hf.co
# and cas-server.xethub.hf.co are blocked by the sandbox network proxy (403
# Forbidden -- see results/fetch_notes.md). This disables the hf-xet client's
# own direct CAS calls, but LFS `resolve` URLs on huggingface.co still
# server-side 302-redirect to the blocked us.aws.cdn.hf.co host regardless, so
# this does NOT unblock real file downloads (see fetch_notes.md for details).
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

echo "df -h before download:"
df -h ~

"$VENV_PY" "$PROJECT_DIR/src/fetch_and_verify.py"
