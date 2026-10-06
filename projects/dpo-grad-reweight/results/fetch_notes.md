# fetch_notes.md — model + dataset provenance (BLOCKED — see below)

HF_HOME=/home/agent/models/hf_cache (outside the git-tracked project folder, as required).

## Licenses + revisions (confirmed via huggingface_hub metadata API — these calls succeed;
they hit huggingface.co's small JSON API endpoints, not the LFS/xet CDN)

| repo | role | revision (commit sha) | license tag |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | model | 7ae557604adf67be50417f59c2c2f167def9a775 | license:apache-2.0 |
| HuggingFaceH4/ultrafeedback_binarized | DPO preference dataset | 3949bf5f8c17c394422ccfab0c31ea9c20bdeb85 | license:mit |
| allenai/ai2_arc (ARC-Easy config) | sanity benchmark | 210d026faf9955653af8916fad021475a3f00453 | license:cc-by-sa-4.0 |

All three licenses are permissive and consistent with PLAN.md's expectations (model
Apache-2.0 confirmed as assumed; dataset MIT, which is in the permissive allow-list; ARC
cc-by-sa-4.0, permissive/attribution-only). No abort triggered on license grounds.

## BLOCKER: actual file bytes (model.safetensors, dataset parquet files) could not be fetched

`huggingface.co/.../resolve/<rev>/<file>` for every LFS-tracked file we tried (model
.safetensors for Qwen2.5-0.5B-Instruct, and for sanity-checking also gpt2 and
bert-base-uncased; train_prefs parquet for ultrafeedback_binarized) returns an HTTP 302
redirect to `https://us.aws.cdn.hf.co/xet-bridge-us/...` (Hugging Face's newer "Xet"
storage backend). That domain is NOT on the approved network allow-list (huggingface.co /
hf.co CDN was assumed to cover it, but the sandbox network proxy treats
`us.aws.cdn.hf.co` as a distinct, unapproved domain and returns 403 Forbidden /
"Approval required for us.aws.cdn.hf.co:443"). The same is true of the xet client's own
direct CAS endpoint, `cas-server.xethub.hf.co`, hit when `hf-xet` is left enabled (also
403 Forbidden by the sandbox proxy).

This is a platform-wide HF storage migration, not specific to one repo: every LFS file we
probed (several different models, one dataset) redirects to the same blocked
`us.aws.cdn.hf.co` host regardless of repo or revision (including the repo's very first
commit). Only small, non-LFS files (config.json, tokenizer files, README.md metadata —
~12MB total landed in the cache) downloaded successfully directly from huggingface.co,
since those don't need the CDN redirect.

Per the lab's network-boundary rule ("a blocked domain is a boundary, never an obstacle:
do not work around it, report the domain"), no workaround was attempted (no alternate
mirrors, no manual CDN URL construction, no requesting sandbox policy approval). Reporting
the blocked domains: **us.aws.cdn.hf.co** and **cas-server.xethub.hf.co**.

Practical effect: the actual Qwen2.5-0.5B-Instruct weights, the ultrafeedback_binarized
parquet files, and the ai2_arc parquet files could NOT be downloaded in this environment.
Steps 3 and 4 (the timed trial and the 13-model projection) could not be executed as a
result — see final report.

HF_HUB_DISABLE_XET=1 was set (to avoid depending on the `hf-xet` package's own direct CAS
calls) but does not change the server-side 302 redirect target for `resolve` URLs, so it
did not help.
