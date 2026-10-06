"""Download core models (safetensors only) + RewardBench at pinned revision; record SHAs/licenses."""
import os, sys, json, datetime
from huggingface_hub import HfApi, snapshot_download

CORE = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
PINNED = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
RB = "allenai/reward-bench"
RB_REV = "168d848cdbbea9764fae4a544dc9ca1e6cca4931"
ALLOW = ["*.safetensors", "*.json", "tokenizer*", "merges.txt", "vocab.json", "LICENSE*"]

def main(out_md):
    api = HfApi()
    rows = []
    for m in CORE:
        info = api.model_info(m, revision=PINNED[m])
        sha = info.sha
        lic = (info.card_data or {}).get("license") if info.card_data else None
        files = [s.rfilename for s in info.siblings]
        assert any(f.endswith(".safetensors") for f in files), f"{m}: no safetensors"
        path = snapshot_download(m, revision=sha, allow_patterns=ALLOW)
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
        rows.append((m, sha, lic, size / 1e9))
        print(m, sha, lic, f"{size/1e9:.2f}GB", flush=True)
    dinfo = api.dataset_info(RB, revision=RB_REV)
    dlic = (dinfo.card_data or {}).get("license") if dinfo.card_data else None
    snapshot_download(RB, repo_type="dataset", revision=RB_REV)
    print(RB, RB_REV, dlic, flush=True)
    with open(out_md, "w") as f:
        f.write(f"# Models and data (recorded by fetch_data.sh, {datetime.date.today()})\n\n")
        f.write("| repo | revision SHA | license (HF card) | size GB |\n|---|---|---|---|\n")
        for m, sha, lic, sz in rows:
            f.write(f"| {m} | {sha} | {lic} | {sz:.2f} |\n")
        f.write(f"| {RB} (dataset) | {RB_REV} | {dlic} | - |\n")
        f.write("\nWeights: safetensors only; loaded with trust_remote_code=False.\n")
    with open(out_md.replace(".md", ".json"), "w") as f:
        json.dump({"models": {m: sha for m, sha, _, _ in rows}, "rewardbench": RB_REV}, f, indent=1)

if __name__ == "__main__":
    main(sys.argv[1])
