"""Pair-level taxonomy and metrics. A judgment record per pair: dict keyed (criterion, order) -> letter 'A'/'B'.
order 'cf' = chosen shown as A; 'rf' = rejected shown as A."""
import itertools
import numpy as np

CATS = ["correct_reversal", "position_locked", "criterion_blind", "reversed_consistent", "other"]

def content(letter, order):
    """Map letter to 'C' (chosen) or 'R' (rejected)."""
    return "C" if (letter == "A") == (order == "cf") else "R"

def classify(j, worse="W1"):
    bc, br = content(j[("better", "cf")], "cf"), content(j[("better", "rf")], "rf")
    wc, wr = content(j[(worse, "cf")], "cf"), content(j[(worse, "rf")], "rf")
    letters = {j[("better", "cf")], j[("better", "rf")], j[(worse, "cf")], j[(worse, "rf")]}
    if bc == br == "C" and wc == wr == "R":
        return "correct_reversal"
    if len(letters) == 1:
        return "position_locked"
    if bc == br == wc == wr:
        return "criterion_blind"
    if bc == br == "R" and wc == wr == "C":
        return "reversed_consistent"
    return "other"

def pair_flags(j, worse="W1"):
    bc, br = content(j[("better", "cf")], "cf"), content(j[("better", "rf")], "rf")
    wc, wr = content(j[(worse, "cf")], "cf"), content(j[(worse, "rf")], "rf")
    return {"better_ok": bc == br == "C", "worse_flip": wc == wr == "R", "cat": classify(j, worse)}

def summarize(flags):
    """flags: list of pair_flags dicts. Returns metric dict (NaN where undefined)."""
    n = len(flags)
    out = {"n": n}
    for c in CATS:
        out["share_" + c] = sum(f["cat"] == c for f in flags) / n if n else float("nan")
    out["acc_better_poscons"] = sum(f["better_ok"] for f in flags) / n if n else float("nan")
    cond = [f for f in flags if f["better_ok"]]
    out["n_cond"] = len(cond)
    out["cond_flip_rate"] = sum(f["worse_flip"] for f in cond) / len(cond) if cond else float("nan")
    uf = sum(f["worse_flip"] for f in flags) / n if n else float("nan")
    out["uncond_flip_rate"] = uf
    out["uncond_flip_chance_norm"] = (uf - 0.25) / 0.75  # random judge = 0.25 -> 0; perfect -> 1
    noncr = [f for f in flags if f["cat"] != "correct_reversal"]
    out["n_noncr"] = len(noncr)
    out["pos_locked_share_of_noncr"] = (sum(f["cat"] == "position_locked" for f in noncr) / len(noncr)
                                        if noncr else float("nan"))
    return out

def reference_rows(worse="W1"):
    """Analytic rows: uniform random judge (exact enumeration of 16 outcomes) and always-A judge."""
    keys = [("better", "cf"), ("better", "rf"), (worse, "cf"), (worse, "rf")]
    rand = [pair_flags(dict(zip(keys, ls)), worse) for ls in itertools.product("AB", repeat=4)]
    always_a = [pair_flags({k: "A" for k in keys}, worse)]
    return {"random_judge": summarize(rand), "always_A_judge": summarize(always_a)}

def bootstrap(flags, stat, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(flags)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(summarize([flags[i] for i in idx])[stat])
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if not len(vals): return float('nan'), float('nan'), 0
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)), int(len(vals))

def paired_bootstrap_diff(flags_small, flags_big, stat, n_boot=2000, seed=0, alpha=0.05):
    """flags_* aligned on the same pairs. CI of stat(big)-stat(small)."""
    assert len(flags_small) == len(flags_big)
    rng = np.random.default_rng(seed)
    n = len(flags_small)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(summarize([flags_big[i] for i in idx])[stat] - summarize([flags_small[i] for i in idx])[stat])
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if not len(vals): return float('nan'), float('nan'), 0
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)), int(len(vals))
