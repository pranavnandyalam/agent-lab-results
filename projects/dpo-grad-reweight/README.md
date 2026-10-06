# dpo-grad-reweight

See `PLAN.md` for the full approved method, metrics, and compute-budget/stop-criteria.
This is a best-effort conceptual reconstruction ("GAW-PO-lite"), not a verified
replication of GAW-PO (arXiv:2610.01511). AI-authored, not peer-reviewed.

## Step 0 — paper text

Superseded by a Lead-relayed scout finding (not independently re-verified by this
builder): the real GAW-PO Eq. 8 differs from our pre-registered GAW-PO-lite fallback
(dot-product + sigmoid link + max over a paired and a "global" chosen-direction term,
vs. our cosine-similarity + clipped-linear link, single term). PLAN.md rev 3 will
incorporate this once re-reviewed. The run below still uses the approved rev-2
GAW-PO-lite fallback formula, unchanged, as instructed.

## Setup

```
python3 -m venv --without-pip .venv   # ensurepip unavailable in this image, no sudo
# bootstrap pip into the venv from a pypi-downloaded wheel (see shell history), then:
.venv/bin/pip install -r requirements.txt
```

Full transitive `pip freeze` recorded in `results/pip_freeze.txt`.

## Fetch (model + dataset)

```
bash fetch_data.sh
```

**Currently BLOCKED** — see `results/fetch_notes.md`. Every Hugging Face LFS file
(any model's `.safetensors`, this dataset's `.parquet` files) 302-redirects to
`us.aws.cdn.hf.co` / `cas-server.xethub.hf.co` (HF's Xet storage CDN), which the
sandbox network proxy rejects (403 Forbidden / approval required). Licenses and
revision hashes were still confirmed via the metadata-only API (no file download
needed):

- Model `Qwen/Qwen2.5-0.5B-Instruct` rev `7ae557604adf67be50417f59c2c2f167def9a775`, license apache-2.0.
- Dataset `HuggingFaceH4/ultrafeedback_binarized` rev `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`, license MIT.
- `allenai/ai2_arc` (ARC-Easy) rev `210d026faf9955653af8916fad021475a3f00453`, license CC-BY-SA-4.0.

## Code

- `src/common.py` — model/data loading, tokenization, vanilla DPO + GAW-PO-lite loss
  (per PLAN.md Step 0 fallback formula), closed-form gradient-alignment weights.
- `src/train_dpo.py` — training loop (1 epoch), the 5 PLAN.md metrics, ARC-Easy
  log-likelihood eval. `--mode full` (real model/data, currently blocked) or
  `--mode smoke_test` (offline: real Qwen tokenizer + chat template, from-config
  random fp32 weights, hand-written synthetic pairs — validates the pipeline only,
  not a result). In smoke-test mode the eval pairs are a SUBSET of the smoke-test
  train pairs (`SMOKE_PAIRS[2:]` out of `SMOKE_PAIRS`, 4 pairs total) — fine for a
  pipeline/wiring smoke test but not a valid train/eval split, so smoke-test output
  JSONs are explicitly marked `PIPELINE_VALIDATION_ONLY_NOT_A_RESULT: true` and
  filenames are prefixed `PIPELINE_ONLY_`.
- `src/fetch_and_verify.py` — license/revision verification + fetch (see blocker above).

Example commands actually run (seed 0):
```
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 300 .venv/bin/python src/train_dpo.py \
    --mode smoke_test --method vanilla --beta 0.1 --seed 0 \
    --out_json results/PIPELINE_ONLY_smoke_test_vanilla_seed0.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 300 .venv/bin/python src/train_dpo.py \
    --mode smoke_test --method gawpolite --beta 0.1 --seed 0 \
    --out_json results/PIPELINE_ONLY_smoke_test_gawpolite_seed0.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 60 .venv/bin/python src/train_dpo.py \
    --mode full --method vanilla --beta 0.1 --seed 0 \
    --n_dev 50 --n_train 5 --n_eval 5 --n_arc 5   # fails at blocked CDN, see
    # results/step3_real_run_attempt.log
```

Steps 3 (mandatory timed trial) and 4 (13-model projection) could not be executed —
see final report to the Lead for details.
