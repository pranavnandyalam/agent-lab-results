"""Drop empty documents (zero-token docs = an <|eod|> immediately following another <|eod|>, or an <|eod|>
at stream position 0) from a uint16 token stream (format: doc tokens + <|eod|>, no BOS). Non-empty docs are
kept byte-identical and in order, each still terminated by exactly one <|eod|>. Logged deviation (PLAN cycle 16).
Usage: python -I src/filter_s1.py --inp data/S1.bin --out data/S1f.bin --meta results/S1_filter.json"""
import argparse, hashlib, json, os
import numpy as np

EOD_ID = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def doc_lengths(a):
    e = np.flatnonzero(a == EOD_ID)
    return np.diff(np.concatenate([[-1], e])) - 1, e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/S1.bin")
    ap.add_argument("--out", default="data/S1f.bin")
    ap.add_argument("--meta", default="results/S1_filter.json")
    a = ap.parse_args()
    x = np.fromfile(resolve(a.inp), dtype=np.uint16)
    if x[-1] != EOD_ID:
        raise SystemExit("input does not end with <|eod|>")
    lens, eods = doc_lengths(x)
    prev = np.concatenate([[EOD_ID], x[:-1]])  # position 0 counts as preceded by a separator
    drop = (x == EOD_ID) & (prev == EOD_ID)
    y = x[~drop]
    y.tofile(resolve(a.out))
    lens2, _ = doc_lengths(y)
    # self-checks
    assert y[-1] == EOD_ID and not np.any((y[1:] == EOD_ID) & (y[:-1] == EOD_ID)) and y[0] != EOD_ID
    assert (lens2 == lens[lens > 0]).all() and int(drop.sum()) == int((lens == 0).sum())
    meta = {"inp": a.inp, "inp_sha256": sha(resolve(a.inp)), "inp_tokens": int(len(x)), "inp_docs": int(len(lens)),
            "empty_docs_dropped": int((lens == 0).sum()), "eod_tokens_dropped": int(drop.sum()),
            "out": a.out, "out_sha256": sha(resolve(a.out)), "out_tokens": int(len(y)), "out_docs": int(len(lens2)),
            "out_doc_len_mean": float(lens2.mean()), "out_doc_len_min": int(lens2.min()),
            "out_ge_4M_plus1": bool(len(y) >= 4194304 + 1),
            "rule": "drop every <|eod|> whose previous token is <|eod|> (or that is at index 0); nonempty docs unchanged"}
    json.dump(meta, open(resolve(a.meta), "w"), indent=1)
    print(json.dumps(meta, indent=1))

if __name__ == "__main__":
    main()
