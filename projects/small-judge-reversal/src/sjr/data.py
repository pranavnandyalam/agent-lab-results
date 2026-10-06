import os, re, random
import pyarrow.parquet as pq
from . import config as C
from .prompts import chat_input

SAFETY_RE = re.compile(r"\b(" + "|".join(C.SAFETY_KEYWORDS) + r")\b", re.IGNORECASE)
SUB2SEC = {s: sec for sec, subs in C.SECTIONS.items() for s in subs}

def rb_path():
    return os.path.join(C.HF_HOME, "hub", "datasets--allenai--reward-bench", "snapshots", C.RB_REV,
                        "data", "filtered-00000-of-00001.parquet")

def load_rb():
    return pq.read_table(rb_path()).to_pylist()

def is_safety_prompt(text):
    return SAFETY_RE.search(text) is not None

def filter_pairs(rows, ntok, full_len, max_resp=C.MAX_RESP_TOKENS, max_full=C.MAX_FULL_PROMPT_TOKENS):
    """ntok(text)->int; full_len(row)->max judge-input tokens over criteria/orders. Returns (kept, counts)."""
    counts = {"total": len(rows), "safety_subset": 0, "unknown_subset": 0, "safety_keyword": 0,
              "resp_too_long": 0, "full_prompt_too_long": 0}
    kept = []
    for r in rows:
        if r["subset"] in C.SAFETY_SUBSETS:
            counts["safety_subset"] += 1; continue
        if r["subset"] not in SUB2SEC:
            counts["unknown_subset"] += 1; continue
        if is_safety_prompt(r["prompt"]):
            counts["safety_keyword"] += 1; continue
        if ntok(r["chosen"]) > max_resp or ntok(r["rejected"]) > max_resp:
            counts["resp_too_long"] += 1; continue
        L = full_len(r)
        if L > max_full:
            counts["full_prompt_too_long"] += 1; continue
        kept.append(dict(r, section=SUB2SEC[r["subset"]], full_tokens=L))
    counts["kept"] = len(kept)
    return kept, counts

def split_pool(pool, n_dev=C.N_DEV, n_per=C.N_PER_RESAMPLE, seeds=C.RESAMPLE_SEEDS, dev_seed=C.DEV_SEED,
               min_pool=C.MIN_POOL):
    assert len(pool) >= min_pool, f"only {len(pool)} pairs after filtering (<{min_pool})"
    ids = sorted(r["id"] for r in pool)
    dev = sorted(random.Random(dev_seed).sample(ids, n_dev))
    used = set(dev)
    res = {}
    for s in seeds:
        avail = [i for i in ids if i not in used]
        pick = sorted(random.Random(s).sample(avail, n_per))
        used.update(pick)
        res[s] = pick
    return dev, res

def make_full_len(tok, is_qwen3=False):
    def full_len(r):
        best = 0
        for crit in ("better", "W1", "W2", "length"):
            for a, b in ((r["chosen"], r["rejected"]), (r["rejected"], r["chosen"])):
                best = max(best, len(tok(chat_input(tok, r["prompt"], a, b, crit, is_qwen3))["input_ids"]))
        return best
    return full_len
