import os, re, random, difflib
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

def make_full_len(tok, is_qwen3=False, crits=("better", "W1", "W2", "length")):
    def full_len(r):
        best = 0
        for crit in crits:
            for a, b in ((r["chosen"], r["rejected"]), (r["rejected"], r["chosen"])):
                best = max(best, len(tok(chat_input(tok, r["prompt"], a, b, crit, is_qwen3))["input_ids"]))
        return best
    return full_len


# ---- non-code pair set (referee: >=200 non-code pairs, substantively different responses, no 150-token filter) ----
NC_SECTIONS = ("chat", "chat-hard", "reasoning")  # reasoning restricted to non-code subsets (math-prm); hep-* excluded
NC_N, NC_SEED, NC_MAX_FULL, NC_MAX_SIM = 200, 20261007, 600, 0.9

def text_sim(a, b):
    """Character-level difflib ratio in [0,1] (1 = identical)."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def noncode_select(rows, full_len, exclude_ids, n=NC_N, seed=NC_SEED, max_full=NC_MAX_FULL, max_sim=NC_MAX_SIM):
    """Deterministic non-code pair set. Filters (in order): safety subsets / unknown subsets; code subsets (hep-*);
    safety keyword in prompt; id already used (dev or main resamples); full judge input > max_full tokens
    (full_len(row), max over the core passes); responses not substantively different (whitespace-normalised equal or
    char similarity >= max_sim). No per-response length filter. Selection: proportional allocation of n over sections
    (largest remainder, ties by section name), then random.Random(seed).sample of the sorted ids within each section.
    Returns (sorted ids, counts, meta)."""
    counts = {"total": len(rows), "safety_subset": 0, "unknown_subset": 0, "code_subset": 0, "safety_keyword": 0,
              "already_used": 0, "full_prompt_too_long": 0, "not_substantively_different": 0}
    pool = {}
    for r in rows:
        if r["subset"] in C.SAFETY_SUBSETS: counts["safety_subset"] += 1; continue
        if r["subset"] not in SUB2SEC: counts["unknown_subset"] += 1; continue
        if r["subset"].startswith("hep"): counts["code_subset"] += 1; continue
        if is_safety_prompt(r["prompt"]): counts["safety_keyword"] += 1; continue
        if r["id"] in exclude_ids: counts["already_used"] += 1; continue
        if full_len(r) > max_full: counts["full_prompt_too_long"] += 1; continue
        if " ".join(r["chosen"].split()) == " ".join(r["rejected"].split()) or text_sim(r["chosen"], r["rejected"]) >= max_sim:
            counts["not_substantively_different"] += 1; continue
        pool.setdefault(SUB2SEC[r["subset"]], []).append(r["id"])
    counts["eligible"] = sum(len(v) for v in pool.values())
    assert counts["eligible"] >= n, f"only {counts['eligible']} eligible non-code pairs (<{n})"
    secs = sorted(pool)
    quota = {s: n * len(pool[s]) // counts["eligible"] for s in secs}
    rem = sorted(secs, key=lambda s: (-(n * len(pool[s]) % counts["eligible"]), s))
    for s in rem[: n - sum(quota.values())]: quota[s] += 1
    ids = []
    for s in secs:
        ids += random.Random(f"{seed}-{s}").sample(sorted(pool[s]), quota[s])
    meta = {"pool_by_section": {s: len(pool[s]) for s in secs}, "quota_by_section": quota}
    return sorted(ids), counts, meta
