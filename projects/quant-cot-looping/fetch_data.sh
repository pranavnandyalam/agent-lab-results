#!/usr/bin/env bash
# Downloads Qwen/Qwen3-0.6B (safetensors only) and openai/gsm8k (main, test parquet) into HF_HOME=~/models/hf.
# Prints and records resolved revision hashes in results/revisions.txt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="$HOME/models/hf"
df -h ~
AVAIL_GB=$(df -BG --output=avail ~ | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 13 ]; then echo "Need >=13GB free (keep 10GB after ~1.5GB download); have ${AVAIL_GB}GB"; exit 1; fi
timeout 1800 "$HERE/.venv/bin/python" -I - "$HERE/results/revisions.txt" <<'PY'
import sys
from huggingface_hub import snapshot_download, HfApi
out = sys.argv[1]
api = HfApi()
m_rev = "c1899de289a04d12100db370d81485cdf75e47ca"
p = snapshot_download("Qwen/Qwen3-0.6B", revision=m_rev,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges.txt", "vocab.json"])
d_rev = "740312add88f781978c0658806c59bc2815b9866"
dp = snapshot_download("openai/gsm8k", repo_type="dataset", revision=d_rev, allow_patterns=["main/*.parquet"])
lines = [f"Qwen/Qwen3-0.6B {m_rev} {p}", f"openai/gsm8k {d_rev} {dp}"]
print("\n".join(lines))
open(out, "w").write("\n".join(lines) + "\n")
PY
df -h ~
