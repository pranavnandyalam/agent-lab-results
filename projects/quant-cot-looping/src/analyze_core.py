"""Core analysis (PLAN rev 3b + cycle-3 deviations): H1-H3 with problem-level bootstrap, sensitivity detectors.

Reads every results/core/<tag>_chunk<k>.json present (read-only; works on partial data and lists missing batches).
Writes results/core/analysis.json and results/core/analysis.md.
--audit: samples 20 traces flagged by the frozen detector (fixed seed) into results/core/audit_sample.md
         (short excerpts around the longest loop span) for the post-run precision audit; needs the tokenizer (offline cache).

Usage (from anywhere; single-threaded, light):
  .venv/bin/python -I src/analyze_core.py            # stats
  .venv/bin/python -I src/analyze_core.py --audit    # stats + audit sample
Seeds: bootstrap/permutation RNG = np.random.default_rng([SEED, level_index, stat_index]), SEED = 0; audit sample seed 0.
"""
import argparse, glob, json, os, re, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from detector import loop_mask

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(HERE, "results", "core")
LEVELS = ["fp32", "w5g64", "w4g64", "w3g32"]                      # PLAN levels
EFF_BITS = {"fp32": 32.0, "w5g64": 5.5, "w4g64": 4.5, "w3g32": 4.0}  # incl. 16-bit scale+zero per group
CHUNKS = [0, 1]
N_TEST, CAP = 16, 2048
SEED, NBOOT, NPERM = 0, 10_000, 10_000
H2_MIN_PAIRS, H2_TOL = 15, 0.30
COLLAPSE_FRAC = 2 / 16                                              # accuracy <= 2/16 on every seed -> collapsed
DETECTORS = {"primary_A20x4+B": 20, "sens_A16x4+B": 16, "sens_A12x4+B": 12}  # sensitivity: only rule A's n changes
MODEL, MODEL_REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"


# ---------------- loading / scoring ----------------
def load_batches():
    data, present = {}, []
    for f in sorted(glob.glob(os.path.join(CORE, "*_chunk*.json"))):
        m = re.fullmatch(r"(.+)_chunk(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        tag, k = m.group(1), int(m.group(2))
        d = json.load(open(f))
        present.append(f"{tag}_chunk{k}")
        data.setdefault(tag, []).extend(d["rows"])
    missing = [f"{t}_chunk{k}" for t in LEVELS for k in CHUNKS if f"{t}_chunk{k}" not in present]
    return data, present, missing


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _num(s):
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").replace("\\%", "").replace("%", "").strip().rstrip(".")
    m = NUM.findall(s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def extract(text, truncated):
    """PLAN Method: closed think block -> text after </think>: last \\boxed{}, else last number.
    No </think>: truncated -> None (incorrect); ended (EOS) without think block -> extract from the whole text."""
    if "</think>" in text:
        seg = text.split("</think>")[-1]
    elif truncated:
        return None
    else:
        seg = text
    seg = seg.replace("<|im_end|>", "")
    bx = re.findall(r"\\boxed\{([^{}]*)\}", seg)
    if bx:
        return _num(bx[-1])
    nums = NUM.findall(seg)
    return float(nums[-1].replace(",", "")) if nums else None


def correct(row):
    p = extract(row["text"], row["truncated"])
    g = _num(row["gold"])
    return p is not None and g is not None and abs(p - g) < 1e-6


def masks(row):
    """Rule-B mask once, rule-A masks per n; union == loop_mask(ids, prompt_ids, n_a=n) exactly."""
    ids, pr = row["token_ids"], row["prompt_ids"]
    mb = loop_mask(ids, prompt_ids=pr, n_a=len(ids) + 1)                     # rule A disabled -> B only
    out = {}
    for name, n in DETECTORS.items():
        ma = loop_mask(ids, prompt_ids=pr, n_a=n, nb_min=1, nb_max=0)     # rule B disabled -> A only
        out[name] = [a or b for a, b in zip(ma, mb)]
    return out


# ---------------- stats helpers ----------------
def ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


def auroc(score, y):
    """P(score_incorrect > score_correct), ties 0.5 (Mann-Whitney with average ranks). None if one class absent."""
    y = np.asarray(y, bool); s = np.asarray(score, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = _ranks(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ranks(s):
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ties
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def rng_for(li, si):
    return np.random.default_rng([SEED, li, si])


# ---------------- per-level tables ----------------
def per_trace(data, det):
    T = {}
    for tag, rows in data.items():
        T[tag] = [dict(problem=r["problem"], seed=r["seed"], n=r["gen_tokens"], trunc=bool(r["truncated"]),
                       loop=int(sum(r["_masks"][det])), correct=r["_correct"]) for r in rows]
    return T


def level_summary(tr):
    seeds = sorted({t["seed"] for t in tr}); probs = sorted({t["problem"] for t in tr})
    acc_seed = {s: float(np.mean([t["correct"] for t in tr if t["seed"] == s])) for s in seeds}
    fin = [t for t in tr if not t["trunc"]]
    return dict(n_traces=len(tr), n_problems=len(probs), seeds=seeds,
                accuracy=float(np.mean([t["correct"] for t in tr])),
                accuracy_by_seed=acc_seed,
                accuracy_finished=float(np.mean([t["correct"] for t in fin])) if fin else None,
                n_finished=len(fin), n_truncated=len(tr) - len(fin),
                mean_tokens=float(np.mean([t["n"] for t in tr])), mean_loop_tokens=float(np.mean([t["loop"] for t in tr])),
                n_flagged=int(sum(t["loop"] > 0 for t in tr)),
                collapsed=bool(all(a <= COLLAPSE_FRAC + 1e-12 for a in acc_seed.values())))


def h1(T, tag, li):
    """share_q = (L_q - L_fp) / (T_q - T_fp); bootstrap over problems (paired: same resampled problems for both)."""
    probs = sorted({t["problem"] for t in T[tag]} & {t["problem"] for t in T["fp32"]})
    def agg(tr):
        a = np.zeros((len(probs), 3))
        for t in tr:
            if t["problem"] in probs:
                i = probs.index(t["problem"]); a[i] += (t["n"], t["loop"], 1)
        return a
    q, f = agg(T[tag]), agg(T["fp32"])
    def stats(qs, fs):
        Tq, Lq = qs[..., 0] / qs[..., 2], qs[..., 1] / qs[..., 2]
        Tf, Lf = fs[..., 0] / fs[..., 2], fs[..., 1] / fs[..., 2]
        A, AL = Tq - Tf, Lq - Lf
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = np.where(A > 0, AL / np.where(A > 0, A, 1), np.nan)
        return Tq, Lq, Tf, Lf, A, AL, sh
    pt = stats(q.sum(0), f.sum(0))
    idx = rng_for(li, 1).integers(0, len(probs), size=(NBOOT, len(probs)))
    bs = stats(q[idx].sum(1), f[idx].sum(1))
    names = ["T_q", "L_q", "T_fp32", "L_fp32", "A_q", "AL_q", "share_q"]
    out = {"n_problems": len(probs)}
    for nm, p, b in zip(names, pt, bs):
        out[nm] = None if not np.isfinite(p) else float(p)
        out[nm + "_ci95"] = ci(b)
    out["share_defined"] = bool(pt[4] > 0)
    out["share_boot_undefined_frac"] = float(np.mean(~np.isfinite(bs[6])))
    out["note"] = "A_q is a lower bound (2048 cap)."
    return out


def h2(T, tag, li):
    f = {(t["problem"], t["seed"]): t for t in T["fp32"]}
    pairs = []
    for t in T[tag]:
        g = f.get((t["problem"], t["seed"]))
        if g is not None and not t["trunc"] and not g["trunc"]:
            df, dq = g["n"] - g["loop"], t["n"] - t["loop"]
            pairs.append((t["problem"], (dq - df) / df))
    out = {"n_pairs": len(pairs), "min_pairs": H2_MIN_PAIRS, "tolerance": H2_TOL}
    if not pairs:
        out.update(testable=False, median_rel_change=None, median_ci95=[None, None], verdict="not testable"); return out
    rel = np.array([r for _, r in pairs]); med = float(np.median(rel))
    probs = sorted({p for p, _ in pairs}); byp = {p: [r for pp, r in pairs if pp == p] for p in probs}
    rng = rng_for(li, 2); bs = []
    for _ in range(NBOOT):
        s = rng.integers(0, len(probs), len(probs))
        bs.append(float(np.median(np.concatenate([byp[probs[i]] for i in s]))))
    testable = len(pairs) >= H2_MIN_PAIRS
    out.update(testable=testable, n_problems=len(probs), median_rel_change=med, median_ci95=ci(bs),
               verdict=("not testable" if not testable else ("within +-30%" if abs(med) <= H2_TOL else "outside +-30%")))
    return out


def h3(tr, li):
    """Exploratory: AUROC for incorrectness; bootstrap over problems; shuffled-label null (permutation of labels)."""
    res = {}
    for sub, sel in (("all", tr), ("non_truncated", [t for t in tr if not t["trunc"]])):
        y = np.array([not t["correct"] for t in sel], bool)
        feats = {"loop_fraction": [t["loop"] / max(t["n"], 1) for t in sel], "total_length": [t["n"] for t in sel],
                 "truncated": [float(t["trunc"]) for t in sel]}
        probs = sorted({t["problem"] for t in sel}); pidx = {p: [i for i, t in enumerate(sel) if t["problem"] == p] for p in probs}
        rs = {"n": len(sel), "n_incorrect": int(y.sum())}
        for fi, (fn, x) in enumerate(feats.items()):
            x = np.asarray(x, float); a = auroc(x, y) if len(sel) else None
            d = {"auroc": a}
            if a is not None and len(np.unique(x)) > 1:
                rng = rng_for(li, 10 + 10 * fi + (sub == "all"))
                bs = []
                for _ in range(NBOOT):
                    s = rng.integers(0, len(probs), len(probs)); ii = np.concatenate([pidx[probs[j]] for j in s])
                    bs.append(auroc(x[ii], y[ii]))
                d["ci95"] = ci(bs); d["boot_undefined_frac"] = float(np.mean([b is None for b in bs]))
                r = _ranks(x); n1, n0 = int(y.sum()), int((~y).sum())
                perm = np.argsort(rng.random((NPERM, len(y))), axis=1)        # shuffled labels
                yp = y[perm]
                null = (yp * r).sum(1) - n1 * (n1 + 1) / 2
                null = null / (n1 * n0)
                d["null_mean"] = float(null.mean()); d["null_ci95"] = ci(null)
                d["perm_p_two_sided"] = float((np.sum(np.abs(null - 0.5) >= abs(a - 0.5) - 1e-12) + 1) / (NPERM + 1))
            else:
                d["note"] = "undefined (one class absent or constant feature)"
            rs[fn] = d
        res[sub] = rs
    return res


def analyse(data, det):
    T = per_trace(data, det)
    present = [l for l in LEVELS if l in T]
    out = {"levels": {l: level_summary(T[l]) for l in present}, "H1": {}, "H2": {}, "H3": {}}
    for l in present:
        out["H3"][l] = h3(T[l], LEVELS.index(l))
    if "fp32" not in T:
        out["H1"] = out["H2"] = "fp32 missing: not computable"
        return out
    for l in present:
        if l == "fp32":
            continue
        li = LEVELS.index(l)
        coll = out["levels"][l]["collapsed"]
        out["H1"][l] = {"excluded_collapsed": True} if coll else h1(T, l, li)
        out["H2"][l] = {"excluded_collapsed": True} if coll else h2(T, l, li)
    ok = [l for l in sorted(out["H1"], key=lambda l: -EFF_BITS[l]) if isinstance(out["H1"][l], dict)
          and out["H1"][l].get("share_q") is not None]
    sh = [out["H1"][l]["share_q"] for l in ok]
    noncoll = [l for l in present if l != "fp32" and not out["levels"][l]["collapsed"]]
    lowest = min(noncoll, key=lambda l: EFF_BITS[l]) if noncoll else None
    out["H1_summary"] = {"order_by_eff_bits_desc": ok, "shares": sh,
                         "monotone_nondecreasing_as_bits_drop": bool(all(b >= a for a, b in zip(sh, sh[1:]))) if len(sh) > 1 else None,
                         "lowest_noncollapsed_level": lowest,
                         "share_at_lowest_gt_0.5": (out["H1"][lowest].get("share_q") is not None and out["H1"][lowest]["share_q"] > 0.5) if lowest else None,
                         "complete_levels": all(l in present for l in LEVELS)}
    return out


# ---------------- markdown ----------------
def f(x, k=3):
    return "NA" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x))


def fci(c, k=3):
    return "NA" if not c or c[0] is None else f"[{c[0]:.{k}f}, {c[1]:.{k}f}]"


def to_md(res):
    L = ["# Core analysis (auto-generated by src/analyze_core.py; do not hand-edit)", "",
         f"Batches present: {', '.join(res['batches_present']) or 'none'}  ",
         f"Batches MISSING: {', '.join(res['batches_missing']) or 'none'}  ",
         f"Bootstrap: {NBOOT} resamples over problems (all seeds of a problem together), seed {SEED}; null: {NPERM} label shuffles.  ",
         "PARTIAL DATA: numbers are provisional." if res["batches_missing"] else "All 8 batches present.", ""]
    for det, r in res["detectors"].items():
        L += [f"## Detector: {det}", "", "| level | eff bits | traces | problems | acc | acc by seed | acc finished (n) | truncated | mean tokens | mean loop tokens | flagged traces | collapsed |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for l, s in r["levels"].items():
            L.append(f"| {l} | {EFF_BITS[l]} | {s['n_traces']} | {s['n_problems']} | {f(s['accuracy'])} | "
                     f"{', '.join(f(v, 2) for v in s['accuracy_by_seed'].values())} | {f(s['accuracy_finished'])} ({s['n_finished']}) | "
                     f"{s['n_truncated']} | {f(s['mean_tokens'], 1)} | {f(s['mean_loop_tokens'], 1)} | {s['n_flagged']} | {s['collapsed']} |")
        L.append("")
        if isinstance(r["H1"], str):
            L += [f"H1/H2: {r['H1']}", ""]
        else:
            L += ["**H1** share_q = AL_q / A_q (A_q lower bound due to 2048 cap)", "",
                  "| level | problems | T_q | T_fp32 | A_q [CI] | AL_q [CI] | share_q [CI] | boot undefined frac |", "|---|---|---|---|---|---|---|---|"]
            for l, h in r["H1"].items():
                if h.get("excluded_collapsed"):
                    L.append(f"| {l} | collapsed, excluded | | | | | | |"); continue
                L.append(f"| {l} | {h['n_problems']} | {f(h['T_q'], 1)} | {f(h['T_fp32'], 1)} | {f(h['A_q'], 1)} {fci(h['A_q_ci95'], 1)} | "
                         f"{f(h['AL_q'], 1)} {fci(h['AL_q_ci95'], 1)} | {f(h['share_q'])} {fci(h['share_q_ci95'])} | {f(h['share_boot_undefined_frac'])} |")
            s = r["H1_summary"]
            L += ["", f"H1 summary: order {s['order_by_eff_bits_desc']}, shares {[round(x, 3) for x in s['shares']]}, "
                  f"monotone as bits drop: {s['monotone_nondecreasing_as_bits_drop']}, lowest non-collapsed: {s['lowest_noncollapsed_level']}, "
                  f"share > 0.5 there: {s['share_at_lowest_gt_0.5']}, all levels present: {s['complete_levels']}", "",
                  "**H2** median paired relative change in distinct tokens (both finished)", "",
                  "| level | pairs | median rel change [CI] | verdict |", "|---|---|---|---|"]
            for l, h in r["H2"].items():
                if h.get("excluded_collapsed"):
                    L.append(f"| {l} | collapsed, excluded | | |"); continue
                L.append(f"| {l} | {h['n_pairs']} | {f(h['median_rel_change'])} {fci(h['median_ci95'])} | {h['verdict']} |")
            L.append("")
        L += ["**H3** (exploratory) AUROC for incorrectness [bootstrap CI] (null mean [null CI], perm p)", "",
              "| level | subset | n (incorrect) | loop fraction | total length | truncated |", "|---|---|---|---|---|---|"]
        for l, h in r["H3"].items():
            for sub, d in h.items():
                cells = []
                for fn in ("loop_fraction", "total_length", "truncated"):
                    x = d[fn]
                    cells.append("NA" if x.get("auroc") is None or "ci95" not in x else
                                 f"{x['auroc']:.3f} {fci(x['ci95'])} (null {x['null_mean']:.3f} {fci(x['null_ci95'])}, p={x['perm_p_two_sided']:.3f})")
                L.append(f"| {l} | {sub} | {d['n']} ({d['n_incorrect']}) | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L) + "\n"


# ---------------- audit ----------------
def audit(data, n=20):
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/models/hf"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    det = "primary_A20x4+B"
    flagged = sorted([(LEVELS.index(tag), r["problem"], r["seed"], tag, r) for tag, rows in data.items() for r in rows
                      if any(r["_masks"][det])], key=lambda x: x[:3])
    k = min(n, len(flagged))
    pick = sorted(np.random.default_rng(SEED).choice(len(flagged), size=k, replace=False).tolist()) if k else []
    L = ["# Precision audit sample (auto-generated by src/analyze_core.py --audit; sample seed 0)", "",
         f"Flagged traces available (frozen detector): {len(flagged)}; sampled: {k} (target {n}).  ",
         "Excerpt = 30 tokens of context, then the longest contiguous loop span in [[...]] (first 120 tokens of it). Newlines shown as \\n.",
         "Label each: D = degenerate loop, S = soft loop (stuck re-quoting), F = false positive (legitimate re-statement).", ""]
    for j, i in enumerate(pick, 1):
        _, p, s, tag, r = flagged[i]
        m, ids = r["_masks"][det], r["token_ids"]
        best, cur, st = (0, 0), 0, 0
        for t, v in enumerate(m + [False]):
            if v:
                if cur == 0: st = t
                cur += 1
            else:
                if cur > best[1] - best[0]: best = (st, st + cur)
                cur = 0
        a, b = best
        pre = tok.decode(ids[max(0, a - 30):a]); span = tok.decode(ids[a:min(b, a + 120)])
        ex = (pre + "[[" + span + ("..." if b - a > 120 else "") + "]]").replace("\n", "\\n")
        L += [f"## {j}. {tag} problem {p} seed {s}", "",
              f"tokens {r['gen_tokens']}, truncated {r['truncated']}, correct {r['_correct']}, loop tokens {sum(m)}, longest span {b - a} tokens at {a}", "",
              "```", ex, "```", "", "Label: [ ] D  [ ] S  [ ] F", ""]
    open(os.path.join(CORE, "audit_sample.md"), "w").write("\n".join(L))
    return len(flagged), k


def main():
    global CORE
    ap = argparse.ArgumentParser(); ap.add_argument("--audit", action="store_true")
    ap.add_argument("--core-dir", default=CORE, help="input/output dir (default results/core; other dirs for testing only)")
    a = ap.parse_args(); CORE = os.path.abspath(a.core_dir)
    data, present, missing = load_batches()
    print("present:", present, "| MISSING:", missing)
    for rows in data.values():
        for r in rows:
            r["_masks"] = masks(r); r["_correct"] = correct(r)
    res = {"batches_present": present, "batches_missing": missing, "seed": SEED, "n_boot": NBOOT, "n_perm": NPERM,
           "eff_bits": EFF_BITS, "detectors_n_a": DETECTORS, "collapse_rule": "accuracy <= 2/16 on every seed",
           "notes": ["problem-text exclusion uses the saved in-context prompt_ids (chat template + problem)",
                     "sensitivity variants change only rule A's n-gram length; rule B kept (as in src/dev_analysis.py)",
                     "partial data: problem sets per level differ; H1/H2 use problems present at both fp32 and q"],
           "detectors": {d: analyse(data, d) for d in DETECTORS}}
    json.dump(res, open(os.path.join(CORE, "analysis.json"), "w"), indent=1)
    open(os.path.join(CORE, "analysis.md"), "w").write(to_md(res))
    print(f"wrote {CORE}/analysis.json, analysis.md")
    if a.audit:
        nf, k = audit(data); print(f"audit: {k} of {nf} flagged traces sampled -> {CORE}/audit_sample.md")


if __name__ == "__main__":
    main()
