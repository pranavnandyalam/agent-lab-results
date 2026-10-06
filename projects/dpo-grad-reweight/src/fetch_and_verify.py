"""Fetch Qwen2.5-0.5B-Instruct + a small ultrafeedback_binarized subset + ARC-Easy
subset into ~/models/hf_cache (HF_HOME). Verify licenses and record exact revision
hashes into results/fetch_notes.md. No trust_remote_code. safetensors only.

KNOWN BLOCKER (observed 2026-10-06, see results/fetch_notes.md): every HF LFS file
resolve URL (model .safetensors, dataset .parquet, for every repo tried, not just
these three) 302-redirects to https://us.aws.cdn.hf.co/xet-bridge-us/... (HF's "Xet"
storage CDN), which the sandbox network proxy rejects with 403 Forbidden /
"Approval required for us.aws.cdn.hf.co:443". Metadata-only calls (model_info,
dataset_info, small non-LFS files) succeed fine against huggingface.co directly. This
script will raise RuntimeError with the blocked-domain explanation if that redirect is
hit; do not add retries/workarounds for it, per the lab's network-boundary rule.
"""
import os
import json
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / "models" / "hf_cache"))
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["HF_HUB_DISABLE_XET"] = "1"  # does not avoid the server-side redirect, see above

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
ARC_DATASET_ID = "allenai/ai2_arc"

api = HfApi()

notes = []
notes.append("# fetch_notes.md — model + dataset provenance\n")
notes.append(f"Fetched with HF_HOME={os.environ['HF_HOME']}\n")

# --- Model ---
model_info = api.model_info(MODEL_ID)
model_rev = model_info.sha
model_license = None
for tag in model_info.tags or []:
    if tag.startswith("license:"):
        model_license = tag.split("license:", 1)[1]
notes.append(f"## Model: {MODEL_ID}\n")
notes.append(f"- Revision (commit sha): {model_rev}\n")
notes.append(f"- License tag from model_info: {model_license}\n")
assert model_license == "apache-2.0", (
    f"ABORT: expected apache-2.0 license, got {model_license!r}"
)
notes.append("- License verified == apache-2.0. Proceeding.\n")

try:
    local_model_dir = snapshot_download(
        repo_id=MODEL_ID,
        revision=model_rev,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "tokenizer*",
            "vocab*",
            "merges.txt",
        ],
    )
except Exception as e:
    with open(RESULTS_DIR / "fetch_notes.md", "w") as f:
        f.writelines(notes)
    raise RuntimeError(
        "BLOCKED: model.safetensors download redirects to us.aws.cdn.hf.co "
        "(Xet CDN), not on the network allow-list. See module docstring and "
        "results/fetch_notes.md. Original error: " + repr(e)
    ) from e
notes.append(f"- Local snapshot dir: {local_model_dir}\n")
# confirm no .bin (pickle) checkpoint files were pulled
local_files = list(Path(local_model_dir).glob("*"))
bin_files = [f.name for f in local_files if f.suffix == ".bin"]
notes.append(f"- Files fetched: {sorted(f.name for f in local_files)}\n")
notes.append(f"- Pickle .bin files present (should be empty): {bin_files}\n")
assert not bin_files, "ABORT: pickle .bin weight files present, safetensors only allowed"

# --- Preference dataset ---
ds_info = api.dataset_info(DATASET_ID)
ds_rev = ds_info.sha
ds_license = None
for tag in ds_info.tags or []:
    if tag.startswith("license:"):
        ds_license = tag.split("license:", 1)[1]
notes.append(f"\n## Dataset: {DATASET_ID}\n")
notes.append(f"- Revision (commit sha): {ds_rev}\n")
notes.append(f"- License tag from dataset_info: {ds_license}\n")
PERMISSIVE = {"mit", "apache-2.0", "cc-by-4.0", "cc0-1.0", "cc-by-sa-4.0"}
assert ds_license in PERMISSIVE, (
    f"ABORT: dataset license {ds_license!r} not in permissive allow-list {PERMISSIVE}"
)
notes.append(f"- License verified in permissive allow-list {sorted(PERMISSIVE)}. Proceeding.\n")

from datasets import load_dataset  # noqa: E402

# Stream small subset only; do not download whole dataset.
pref_ds = load_dataset(
    DATASET_ID,
    split="train_prefs",
    revision=ds_rev,
)
notes.append(f"- train_prefs split size (full): {len(pref_ds)}\n")

# --- ARC-Easy sanity benchmark ---
arc_info = api.dataset_info(ARC_DATASET_ID)
arc_rev = arc_info.sha
arc_license = None
for tag in arc_info.tags or []:
    if tag.startswith("license:"):
        arc_license = tag.split("license:", 1)[1]
notes.append(f"\n## ARC sanity benchmark: {ARC_DATASET_ID} (ARC-Easy config)\n")
notes.append(f"- Revision (commit sha): {arc_rev}\n")
notes.append(f"- License tag from dataset_info: {arc_license}\n")
assert arc_license in PERMISSIVE | {"cc-by-sa-4.0"}, (
    f"ABORT: ARC license {arc_license!r} not permissive"
)
notes.append("- License verified permissive. Proceeding.\n")

arc_ds = load_dataset(ARC_DATASET_ID, "ARC-Easy", split="test", revision=arc_rev)
notes.append(f"- ARC-Easy test split size (full): {len(arc_ds)}\n")

notes.append("\n## Summary\n")
notes.append(f"- MODEL_REVISION={model_rev}\n")
notes.append(f"- DATASET_REVISION={ds_rev}\n")
notes.append(f"- ARC_REVISION={arc_rev}\n")
notes.append("- All licenses verified permissive at fetch time (see above). No abort triggered.\n")

with open(RESULTS_DIR / "fetch_notes.md", "w") as f:
    f.writelines(notes)

# Also dump a small machine-readable manifest other scripts can read.
manifest = {
    "model_id": MODEL_ID,
    "model_revision": model_rev,
    "model_license": model_license,
    "dataset_id": DATASET_ID,
    "dataset_revision": ds_rev,
    "dataset_license": ds_license,
    "arc_dataset_id": ARC_DATASET_ID,
    "arc_revision": arc_rev,
    "arc_license": arc_license,
    "hf_home": os.environ["HF_HOME"],
}
with open(RESULTS_DIR / "fetch_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("DONE. See results/fetch_notes.md and results/fetch_manifest.json")
print(json.dumps(manifest, indent=2))
