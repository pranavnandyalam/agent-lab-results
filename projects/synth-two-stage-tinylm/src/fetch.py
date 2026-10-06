"""Doc-level seeded disjoint split of a TinyStories parquet shard into R_gen/R_train/R_dev/R_val.
Writes data/<split>.txt with documents separated by a line containing only <|eod|>."""
import argparse, hashlib, json, os
import numpy as np
import pyarrow.parquet as pq

EOD = "<|eod|>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    for s in ["gen", "train", "dev", "val"]:
        ap.add_argument(f"--{s}_chars", type=int, required=True)
    a = ap.parse_args()
    texts = pq.read_table(a.parquet, columns=["text"]).column("text").to_pylist()
    n_raw = len(texts)
    texts = [t.strip() for t in texts if t and t.strip() and EOD not in t]
    n_nonempty = len(texts)
    # Exact-duplicate removal (sha256 of stripped text), keeping first occurrence, BEFORE the
    # seeded permutation so no identical document can land in two splits.
    seen, uniq = set(), []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        if h not in seen:
            seen.add(h); uniq.append(t)
    texts = uniq
    order = np.random.default_rng(a.seed).permutation(len(texts))
    budgets = [("R_val", a.val_chars), ("R_dev", a.dev_chars), ("R_train", a.train_chars), ("R_gen", a.gen_chars)]
    i, meta = 0, {"seed": a.seed, "n_docs_in_shard_raw": n_raw, "n_docs_nonempty": n_nonempty,
                  "exact_duplicates_dropped": n_nonempty - len(texts), "n_docs_unique": len(texts),
                  "dedup": "sha256 of stripped text, first occurrence kept, before permutation",
                  "splits": {}}
    for name, budget in budgets:
        docs, n = [], 0
        while n < budget:
            if i >= len(order):
                raise SystemExit(f"shard exhausted while filling {name}")
            t = texts[order[i]]; i += 1
            docs.append(t); n += len(t)
        with open(os.path.join(a.out, f"{name}.txt"), "w", encoding="utf-8") as f:
            for t in docs:
                f.write(t + "\n" + EOD + "\n")
        meta["splits"][name] = {"docs": len(docs), "chars": n}
    with open(os.path.join(a.out, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))

if __name__ == "__main__":
    main()
