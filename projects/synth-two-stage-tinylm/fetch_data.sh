#!/usr/bin/env bash
# Fetch a TinyStories train shard (roneneldan/TinyStories, license CDLA-Sharing-1.0;
# Eldan & Li 2023, arXiv 2305.07759) at a pinned commit, verify sha256, split into
# disjoint doc-level R_gen/R_train/R_dev/R_val under data/ (gitignored).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REV=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
FILE=data/train-00000-of-00004-2d5a1467fff1081b.parquet
SHA256=77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b
RAW="${RAW_DIR:-$HERE/data/raw}"
mkdir -p "$RAW" "$HERE/data"
OUT="$RAW/train-00000.parquet"
if [ ! -f "$OUT" ] || ! echo "$SHA256  $OUT" | sha256sum -c --quiet; then
  timeout 900 curl -fL --retry 3 -o "$OUT.part" \
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/$REV/$FILE"
  echo "$SHA256  $OUT.part" | sha256sum -c
  mv "$OUT.part" "$OUT"
fi
cd /tmp && OMP_NUM_THREADS=4 timeout 900 "$HERE/.venv/bin/python" -I "$HERE/src/fetch.py" \
  --parquet "$OUT" --out "$HERE/data" --seed 0 \
  --gen_chars 32000000 --train_chars 30000000 --dev_chars 1300000 --val_chars 1300000
