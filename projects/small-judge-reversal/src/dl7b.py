import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/models/hf_cache"))
from huggingface_hub import snapshot_download, HfApi
repo="Qwen/Qwen2.5-7B-Instruct"
info=HfApi().model_info(repo)
print("SHA", info.sha, "LICENSE", (info.card_data or {}).get("license"))
p=snapshot_download(repo, revision=info.sha, allow_patterns=["*.safetensors","*.json","tokenizer*","vocab.json","merges.txt"])
print("PATH", p)
