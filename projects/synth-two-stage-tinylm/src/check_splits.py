"""Verify zero exact-text overlap (stripped document text) across all pairs of splits,
and zero within-split duplicates. Usage: check_splits.py --data data/ ; exits 1 on any overlap."""
import argparse, hashlib, itertools, os, sys

EOD = "<|eod|>"
SPLITS = ["R_gen", "R_train", "R_dev", "R_val"]

def doc_hashes(path):
    with open(path, encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split("\n" + EOD + "\n") if d.strip()]
    hs = [hashlib.sha256(d.encode("utf-8")).hexdigest() for d in docs]
    return hs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    H, bad = {}, 0
    for s in SPLITS:
        hs = doc_hashes(os.path.join(a.data, f"{s}.txt"))
        H[s] = set(hs)
        dup = len(hs) - len(H[s])
        bad += dup
        print(f"{s}: docs={len(hs)} unique={len(H[s])} within_split_dups={dup}")
    for x, y in itertools.combinations(SPLITS, 2):
        n = len(H[x] & H[y])
        bad += n
        print(f"overlap {x} & {y}: {n}")
    print("RESULT:", "PASS (zero exact-text overlap)" if bad == 0 else f"FAIL ({bad})")
    sys.exit(0 if bad == 0 else 1)

if __name__ == "__main__":
    main()
