# fetch_notes.md — model + dataset provenance
Fetched with HF_HOME=/home/agent/models/hf_cache
## Model: Qwen/Qwen2.5-0.5B-Instruct
- Revision (commit sha): 7ae557604adf67be50417f59c2c2f167def9a775
- License tag from model_info: apache-2.0
- License verified == apache-2.0. Proceeding.
- Local snapshot dir: /home/agent/models/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
- Files fetched: ['config.json', 'generation_config.json', 'merges.txt', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']
- Pickle .bin files present (should be empty): []

## Dataset: HuggingFaceH4/ultrafeedback_binarized
- Revision (commit sha): 3949bf5f8c17c394422ccfab0c31ea9c20bdeb85
- License tag from dataset_info: mit
- License verified in permissive allow-list ['apache-2.0', 'cc-by-4.0', 'cc-by-sa-4.0', 'cc0-1.0', 'mit']. Proceeding.
- train_prefs split size (full): 61135

## ARC sanity benchmark: allenai/ai2_arc (ARC-Easy config)
- Revision (commit sha): 210d026faf9955653af8916fad021475a3f00453
- License tag from dataset_info: cc-by-sa-4.0
- License verified permissive. Proceeding.
- ARC-Easy test split size (full): 2376

## Summary
- MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
- DATASET_REVISION=3949bf5f8c17c394422ccfab0c31ea9c20bdeb85
- ARC_REVISION=210d026faf9955653af8916fad021475a3f00453
- All licenses verified permissive at fetch time (see above). No abort triggered.
