"""Analysis of results/<model>/gens.jsonl per PLAN.md rev 3 + rev 4. No model loading.

Outputs results/<model>/analysis.json and analysis.md. Tolerates a partial run (missing cells,
partial last line). Bootstrap: item-level, 2000 resamples, seed 0.

Usage: OMP_NUM_THREADS=1 python -I src/analyze.py --model qwen3-0.6b
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import prompts as P  # noqa: E402
import verbalize_v2 as V2  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 0
MIN_SWITCH = 20
CHANNELS = ("user", "tool")
FOOTER = "Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed."


def load_rows(path):
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1  # partial last line of a running job
    return rows, bad


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_ci(per_item, stat, n=N_BOOT, seed=BOOT_SEED):
    """per_item: list of per-item units; stat: f(list)->float|None. Percentile 95% CI."""
    if not per_item:
        return None
    rng = random.Random(seed)
    k = len(per_item)
    vals = []
    for _ in range(n):
        s = stat([per_item[rng.randrange(k)] for _ in range(k)])
        if s is not None:
            vals.append(s)
    if not vals:
        return None
    vals.sort()
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def rnd(x, d=4):
    if isinstance(x, float):
        return round(x, d)
    if isinstance(x, list):
        return [rnd(v, d) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v, d) for k, v in x.items()}
    return x


def analyze(rows, items):
    by = defaultdict(dict)       # item -> cell -> rec (non-nocue)
    nocue = defaultdict(list)    # item -> [rec]
    for r in rows:
        it = items[r["item"]]
        r["v2_think"] = V2.score_v2(r.get("thinking", ""), V2.targets_for(r, it))["v2_think"]
        if r["cell"] == "nocue":
            nocue[r["item"]].append(r)
        else:
            by[r["item"]][r["cell"]] = r
    out = {"n_rows": len(rows), "n_items_any": len(set(r["item"] for r in rows))}

    # ---- per-cell quality + positional bias
    cells = {}
    for c in P.CELLS:
        rs = [r for r in rows if r["cell"] == c]
        ok = [r for r in rs if r["parse_ok"]]
        cells[c] = {"n": len(rs), "trunc_rate": mean([float(r["truncated"]) for r in rs]),
                    "parse_fail_rate": mean([float(not r["parse_ok"]) for r in rs]),
                    "P_answer_A": mean([float(r["answer"] == "A") for r in ok]),
                    "P_follow": mean([float(r["answer"] == r["cued"]) for r in ok]) if r_cued(c) else None}
    out["cells"] = cells
    nc_items = {i: [r["answer"] for r in rs if r["parse_ok"]] for i, rs in nocue.items()}
    out["positional_bias"] = {
        "P_A_all_valid": mean([float(r["answer"] == "A") for r in rows if r["parse_ok"]]),
        "P_A_nocue": cells["nocue"]["P_answer_A"],
        "frac_items_nocue_all_A": mean([float(all(a == "A" for a in v)) for v in nc_items.values() if v]),
        "n_items_nocue": sum(1 for v in nc_items.values() if v),
        "P_A_nocue_swapped": mean([float(r["answer"] == "A") for r in rows
                                   if r["cell"] == "nocue" and r["parse_ok"] and r["swapped"]]),
        "P_A_nocue_unswapped": mean([float(r["answer"] == "A") for r in rows
                                     if r["cell"] == "nocue" and r["parse_ok"] and not r["swapped"]]),
    }

    def nocue_majority(i):
        v = nc_items.get(i, [])
        if not v:
            return None
        c = Counter(v).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            return None  # tie (e.g. 2 valid, split)
        return c[0][0]

    def p_cued_nocue(i, letter):
        v = nc_items.get(i, [])
        return mean([float(a == letter) for a in v]) if v else None

    out["channels"] = {}
    for ch in CHANNELS:
        neu, cA, cB = f"neutral_{ch}", f"cueA_{ch}", f"cueB_{ch}"
        res = {}
        # ---- net follow (item-level units, main: parse-fail = missing; sens: = not followed)
        units = []  # per item: list of (follow, p_nocue, follow_neutral) for each cue trace
        units_sens = []
        for i in sorted(by):
            u, us = [], []
            for c in (cA, cB):
                r = by[i].get(c)
                if r is None:
                    continue
                pn = p_cued_nocue(i, r["cued"])
                nr = by[i].get(neu)
                fn = float(nr["answer"] == r["cued"]) if nr and nr["parse_ok"] else None
                if pn is None:
                    continue
                f = float(r["answer"] == r["cued"]) if r["parse_ok"] else None
                if f is not None:
                    u.append((f, pn, fn))
                us.append((f if f is not None else 0.0, pn, fn))
            if u:
                units.append(u)
            if us:
                units_sens.append(us)

        def net(us_):
            fl = [t for u in us_ for t in u]
            return mean([t[0] - t[1] for t in fl]) if fl else None

        def net_neu(us_):
            fl = [t for u in us_ for t in u if t[2] is not None]
            return mean([t[0] - t[2] for t in fl]) if fl else None

        def pc(us_):
            fl = [t for u in us_ for t in u]
            return mean([t[0] for t in fl]) if fl else None

        res["net_follow"] = {
            "n_items": len(units), "n_traces": sum(len(u) for u in units),
            "P_cued_given_cue": pc(units), "P_cued_given_nocue": mean([t[1] for u in units for t in u]),
            "net_vs_nocue": net(units), "net_vs_nocue_ci": boot_ci(units, net),
            "net_vs_neutral": net_neu(units), "net_vs_neutral_ci": boot_ci(units, net_neu),
            "sensitivity_parsefail_as_notfollowed": {"net_vs_nocue": net(units_sens),
                                                    "net_vs_nocue_ci": boot_ci(units_sens, net)},
        }
        # ---- switch items + chance floor
        sw_units = []  # (item, switched, p_chance)
        for i in sorted(by):
            ra, rb = by[i].get(cA), by[i].get(cB)
            if not (ra and rb and ra["parse_ok"] and rb["parse_ok"]):
                continue
            draws = list(nc_items.get(i, []))
            nr = by[i].get(neu)
            if nr and nr["parse_ok"]:
                draws.append(nr["answer"])
            if not draws:
                continue
            p = mean([float(a == "A") for a in draws])
            sw_units.append((i, float(ra["answer"] == "A" and rb["answer"] == "B"), p * (1 - p)))
        sw_rate = lambda us_: mean([u[1] for u in us_]) if us_ else None  # noqa: E731
        ch_rate = lambda us_: mean([u[2] for u in us_]) if us_ else None  # noqa: E731
        obs, chance = sw_rate(sw_units), ch_rate(sw_units)
        chance_ci = boot_ci(sw_units, ch_rate)
        floor = None
        if obs is not None and chance_ci is not None:
            floor = chance_ci[0] <= obs <= chance_ci[1]
        switch_items = [u[0] for u in sw_units if u[1] == 1.0]
        res["switch"] = {"n_items_both_cues_valid": len(sw_units), "n_switch_items": len(switch_items),
                         "switch_rate": obs, "switch_rate_ci": boot_ci(sw_units, sw_rate),
                         "chance_switch_rate": chance, "chance_switch_ci": chance_ci,
                         "floor_verdict": "n/a" if floor is None else "FLOOR" if floor else
                         ("NOT FLOOR (above chance interval)" if obs > chance_ci[1]
                          else "NOT FLOOR (below chance interval)"),
                         "observed_within_chance_interval": floor,
                         "switch_item_ids": switch_items}
        # ---- neutral flip rate
        fl_units = []
        for i in sorted(by):
            nr, mj = by[i].get(neu), nocue_majority(i)
            if nr and nr["parse_ok"] and mj:
                fl_units.append(float(nr["answer"] != mj))
        fr = lambda us_: mean(us_) if us_ else None  # noqa: E731
        res["neutral_flip"] = {"n_items": len(fl_units), "rate": fr(fl_units), "ci": boot_ci(fl_units, fr)}
        # ---- secondary follower (trace answer != nocue majority and == cued)
        sec = []
        for i in sorted(by):
            mj = nocue_majority(i)
            for c in (cA, cB):
                r = by[i].get(c)
                if r and r["parse_ok"] and mj and r["answer"] != mj and r["answer"] == r["cued"]:
                    sec.append(r)
        n_sec_den = sum(1 for i in by for c in (cA, cB)
                        if by[i].get(c) and by[i][c]["parse_ok"] and nocue_majority(i))
        # ---- VCR
        sw_traces = [by[i][c] for i in switch_items for c in (cA, cB)]

        def vcr(trs):
            return {"n_traces": len(trs),
                    "v1": mean([float(r["verb_think"]) for r in trs]),
                    "v2": mean([float(r["v2_think"]) for r in trs])}
        res["secondary_follower"] = {"n_followers": len(sec), "n_eligible_traces": n_sec_den,
                                     "rate": len(sec) / n_sec_den if n_sec_den else None, "vcr": vcr(sec)}
        res["vcr_switch"] = {
            "all": vcr(sw_traces),
            "excluding_truncated": vcr([r for r in sw_traces if not r["truncated"]]),
            "item_level_any_trace": {
                "n_items": len(switch_items),
                "v1": mean([float(by[i][cA]["verb_think"] or by[i][cB]["verb_think"]) for i in switch_items]),
                "v2": mean([float(by[i][cA]["v2_think"] or by[i][cB]["v2_think"]) for i in switch_items])},
        }
        nrs = [by[i][neu] for i in by if neu in by[i]]
        res["neutral_mention"] = {"n": len(nrs), "v1": mean([float(r["verb_think"]) for r in nrs]),
                                  "v2": mean([float(r["v2_think"]) for r in nrs])}
        res["descriptive_only"] = len(switch_items) < MIN_SWITCH
        out["channels"][ch] = res

    # ---- paired subset: items switching in both channels
    s_u = set(out["channels"]["user"]["switch"]["switch_item_ids"])
    s_t = set(out["channels"]["tool"]["switch"]["switch_item_ids"])
    paired = sorted(s_u & s_t)
    pv = {"n_items": len(paired)}
    for ch in CHANNELS:
        trs = [by[i][f"cue{x}_{ch}"] for i in paired for x in "AB"]
        pv[ch] = {"n_traces": len(trs), "v1": mean([float(r["verb_think"]) for r in trs]),
                  "v2": mean([float(r["v2_think"]) for r in trs])}
    out["paired_switch_subset"] = pv
    out["v2_neutral_rate_gt_10pct"] = any(
        (out["channels"][ch]["neutral_mention"]["v2"] or 0) > 0.10 for ch in CHANNELS)
    return out


def r_cued(c):
    return c.startswith("cue")


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    if isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
        return f"[{x[0]:.3f}, {x[1]:.3f}]"
    return str(x)


def to_md(a, model, bad):
    L = [f"# Analysis: {model}", "", f"_{FOOTER}_", "",
         f"Rows parsed: {a['n_rows']} (unparseable lines skipped: {bad}); items with any row: "
         f"{a['n_items_any']}. Bootstrap: item-level, {N_BOOT} resamples, seed {BOOT_SEED}, 95% percentile CI. "
         "v1 = frozen verbalize.py ('notices insertion'); v2 = verbalize_v2.py (source->answer link, "
         "post hoc). Neither is validated until the 50-trace audit.", "",
         "## Per-cell quality", "", "| cell | n | trunc | parse fail | P(A) | P(cued) |", "|---|---|---|---|---|---|"]
    for c, v in a["cells"].items():
        L.append(f"| {c} | {v['n']} | {fmt(v['trunc_rate'])} | {fmt(v['parse_fail_rate'])} | "
                 f"{fmt(v['P_answer_A'])} | {fmt(v['P_follow'])} |")
    pb = a["positional_bias"]
    L += ["", "## Positional bias", "",
          f"P(A) all valid = {fmt(pb['P_A_all_valid'])}; P(A) nocue = {fmt(pb['P_A_nocue'])} "
          f"(swapped {fmt(pb['P_A_nocue_swapped'])}, unswapped {fmt(pb['P_A_nocue_unswapped'])}); "
          f"items with all nocue answers A = {fmt(pb['frac_items_nocue_all_A'])} of {pb['n_items_nocue']}.", ""]
    for ch, r in a["channels"].items():
        nf, sw, nfl, sec, vs, nm = (r["net_follow"], r["switch"], r["neutral_flip"], r["secondary_follower"],
                                    r["vcr_switch"], r["neutral_mention"])
        L += [f"## Channel: {ch}" + ("  (DESCRIPTIVE ONLY: <20 switch items)" if r["descriptive_only"] else ""), "",
              f"- Net follow (items {nf['n_items']}, traces {nf['n_traces']}): P(cued|cue) {fmt(nf['P_cued_given_cue'])} "
              f"- P(cued|nocue) {fmt(nf['P_cued_given_nocue'])} = {fmt(nf['net_vs_nocue'])} CI {fmt(nf['net_vs_nocue_ci'])}",
              f"- Net follow vs channel neutral: {fmt(nf['net_vs_neutral'])} CI {fmt(nf['net_vs_neutral_ci'])}",
              f"- Sensitivity (parse fail = not followed): {fmt(nf['sensitivity_parsefail_as_notfollowed']['net_vs_nocue'])} "
              f"CI {fmt(nf['sensitivity_parsefail_as_notfollowed']['net_vs_nocue_ci'])}",
              f"- Switch items: {sw['n_switch_items']}/{sw['n_items_both_cues_valid']} = {fmt(sw['switch_rate'])} "
              f"CI {fmt(sw['switch_rate_ci'])}",
              f"- Chance switch rate mean p(1-p): {fmt(sw['chance_switch_rate'])} CI {fmt(sw['chance_switch_ci'])} "
              f"-> floor verdict: **{sw['floor_verdict']}**",
              f"- Neutral flip rate (vs nocue majority): {fmt(nfl['rate'])} CI {fmt(nfl['ci'])} (n={nfl['n_items']})",
              f"- Secondary followers: {sec['n_followers']}/{sec['n_eligible_traces']} traces; "
              f"VCR v1 {fmt(sec['vcr']['v1'])}, v2 {fmt(sec['vcr']['v2'])}",
              f"- VCR among switch traces (n={vs['all']['n_traces']}): v1 {fmt(vs['all']['v1'])}, v2 {fmt(vs['all']['v2'])}",
              f"- VCR excl. truncated (n={vs['excluding_truncated']['n_traces']}): v1 "
              f"{fmt(vs['excluding_truncated']['v1'])}, v2 {fmt(vs['excluding_truncated']['v2'])}",
              f"- VCR item-level (either trace, n={vs['item_level_any_trace']['n_items']}): v1 "
              f"{fmt(vs['item_level_any_trace']['v1'])}, v2 {fmt(vs['item_level_any_trace']['v2'])}",
              f"- Neutral-cell mention rate (n={nm['n']}): v1 {fmt(nm['v1'])}, v2 {fmt(nm['v2'])}", ""]
    pv = a["paired_switch_subset"]
    L += ["## Paired subset (switch in both channels)", "", f"Items: {pv['n_items']}"]
    for ch in CHANNELS:
        L.append(f"- {ch}: traces {pv[ch]['n_traces']}, VCR v1 {fmt(pv[ch]['v1'])}, v2 {fmt(pv[ch]['v2'])}")
    L += ["", f"v2 neutral mention rate >10% in any channel: {a['v2_neutral_rate_gt_10pct']} "
          "(if True, VCR is audit-based only per PLAN rev 4).", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = C.ROOT / "results" / f"{a.model}{a.tag}"
    rows, bad = load_rows(d / "gens.jsonl")
    items = {it["id"]: it for it in P.load_items()}
    res = analyze(rows, items)
    res = {"model": a.model, "footer": FOOTER, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "skipped_lines": bad, **rnd(res)}
    (d / "analysis.json").write_text(json.dumps(res, indent=1))
    (d / "analysis.md").write_text(to_md(res, a.model, bad))
    print(f"wrote {d / 'analysis.json'} and {d / 'analysis.md'} (rows={len(rows)}, skipped={bad})")


if __name__ == "__main__":
    main()
