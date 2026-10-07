"""Generate paper figures and the per-cell table from the raw result files.

Inputs (read only):
  results/qwen3-0.6b/analysis.json   (output of src/analyze.py)
  results/qwen3-0.6b/audit_key.json  + audit_labels.json  (50-trace audit)
  results/qwen3-0.6b/gens.jsonl      (540 raw generations; used for token-length medians)
Outputs:
  paper/figures/fig_follow.pdf, paper/figures/fig_vcr.pdf  (pgfplots via pdflatex)
  paper/figures/tab_cells.tex                               (per-cell quality table body)
  paper/figures/numbers.txt                                 (every plotted number, for checking)

Stdlib only (no matplotlib in the env). Needs pdflatex + pgfplots.
Usage: python3 -I projects/cue-verbalize-sub4b/paper/make_figures.py

Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.
"""
import json
import math
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RES = ROOT / "results" / "qwen3-0.6b"
FIG = PAPER / "figures"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def audit_counts():
    key = json.loads((RES / "audit_key.json").read_text())["key"]
    lab = json.loads((RES / "audit_labels.json").read_text())["labels"]
    out = {}
    for e in key:
        s = e["stratum"]
        n, y = out.get(s, (0, 0))
        out[s] = (n + 1, y + (lab[e["audit_id"]] == "Y"))
    return out


def compile_standalone(name, body):
    doc = (r"\documentclass[border=2pt]{standalone}" "\n"
           r"\usepackage{pgfplots}\pgfplotsset{compat=1.17}\usetikzlibrary{patterns}" "\n"
           r"\begin{document}" "\n" + body + "\n" r"\end{document}" "\n")
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / f"{name}.tex").write_text(doc)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                        f"{name}.tex"], cwd=td, check=True, stdout=subprocess.DEVNULL)
        shutil.copy(Path(td) / f"{name}.pdf", FIG / f"{name}.pdf")


def bar_plot(groups, series, ylabel, legend_cols=3, width="8.4cm", height="4.6cm"):
    """series: list of (legend, style, [(x, y, lo, hi), ...])."""
    plots = []
    for leg, style, pts in series:
        # asymmetric error bars via explicit +=/-=
        coords = " ".join(f"({x},{y:.4f}) += (0,{hi - y:.4f}) -= (0,{y - lo:.4f})"
                          for x, y, lo, hi in pts)
        has_ci = any(hi - lo > 0 for _, _, lo, hi in pts)
        eb = ", error bars/.cd, y dir=both, y explicit, error bar style={black}" if has_ci else ""
        if not has_ci:
            coords = " ".join(f"({x},{y:.4f})" for x, y, lo, hi in pts)
        plots.append(rf"\addplot[{style}{eb}] coordinates {{{coords}}};"
                     "\n" rf"\addlegendentry{{{leg}}}")
    return (r"\begin{tikzpicture}\begin{axis}[ybar, bar width=7pt, width=" + width + ", height=" + height + ","
            r"ymin=0, ymax=1.08, ylabel={" + ylabel + r"}, symbolic x coords={" + ",".join(groups) + "},"
            r"xtick=data, enlarge x limits=0.4, area legend, ytick={0,0.2,0.4,0.6,0.8,1},"
            r"legend style={at={(0.5,1.03)}, anchor=south, legend columns=" + str(legend_cols) + r", font=\scriptsize, draw=none},"
            r"tick label style={font=\scriptsize}, label style={font=\small}, ymajorgrids, grid style={gray!25}]" "\n"
            + "\n".join(plots) + "\n" r"\end{axis}\end{tikzpicture}")


def main():
    FIG.mkdir(exist_ok=True)
    a = json.loads((RES / "analysis.json").read_text())
    ch = a["channels"]
    nums = []

    # Figure 1: switch rate vs chance switch rate vs neutral flip rate (item-bootstrap 95% CIs)
    s_switch, s_chance, s_flip = [], [], []
    for c, lab in (("user", "user"), ("tool", "tool")):
        sw = ch[c]["switch"]
        nf = ch[c]["neutral_flip"]
        s_switch.append((lab, sw["switch_rate"], *sw["switch_rate_ci"]))
        s_chance.append((lab, sw["chance_switch_rate"], *sw["chance_switch_ci"]))
        s_flip.append((lab, nf["rate"], *nf["ci"]))
        nums += [f"{c} switch {sw['n_switch_items']}/{sw['n_items_both_cues_valid']} = {sw['switch_rate']} CI {sw['switch_rate_ci']}",
                 f"{c} chance switch {sw['chance_switch_rate']} CI {sw['chance_switch_ci']}",
                 f"{c} neutral flip {nf['rate']} CI {nf['ci']} n={nf['n_items']}"]
    compile_standalone("fig_follow", bar_plot(
        ["user", "tool"],
        [("observed switch rate", "fill=black!70, draw=black", s_switch),
         ("chance switch rate", "fill=black!15, draw=black", s_chance),
         ("neutral flip rate", "fill=white, draw=black, postaction={pattern=north east lines}", s_flip)],
        "rate (items)"))

    # Figure 2: verbalization by channel. Regex rates have no CI in analysis.json (plotted without bars);
    # audit rates get Wilson 95% intervals.
    au = audit_counts()
    reg_v1, reg_v2, pair_v2, aud = [], [], [], []
    for c in ("user", "tool"):
        v = ch[c]["vcr_switch"]["all"]
        p = a["paired_switch_subset"][c]
        reg_v1.append((c, v["v1"], v["v1"], v["v1"]))
        reg_v2.append((c, v["v2"], v["v2"], v["v2"]))
        pair_v2.append((c, p["v2"], p["v2"], p["v2"]))
        n, y = au[f"cue_{c}"]
        lo, hi = wilson(y, n)
        aud.append((c, y / n, lo, hi))
        nums += [f"{c} regex VCR switch traces n={v['n_traces']}: v1 {v['v1']} v2 {v['v2']}",
                 f"{c} paired subset n_traces={p['n_traces']}: v2 {p['v2']}",
                 f"{c} audit {y}/{n} Wilson95 [{lo:.4f}, {hi:.4f}]"]
    for s in ("neutral_user", "neutral_tool"):
        n, y = au[s]
        nums.append(f"{s} audit {y}/{n}")
    compile_standalone("fig_vcr", bar_plot(
        ["user", "tool"],
        [("regex v1, switch traces", "fill=black!15, draw=black", reg_v1),
         ("regex v2, switch traces", "fill=black!45, draw=black", reg_v2),
         ("regex v2, paired subset", "fill=white, draw=black, postaction={pattern=north east lines}", pair_v2),
         ("audit (Wilson 95\\%)", "fill=black!80, draw=black", aud)],
        "verbalization rate", legend_cols=2))

    # Table: per-cell quality + median generated tokens (from raw rows)
    toks = {}
    for line in (RES / "gens.jsonl").open():
        r = json.loads(line)
        toks.setdefault(r["cell"], []).append(r["n_new_tokens"])
    rows = []
    names = {"nocue": "no cue (3 seeds)", "neutral_user": "neutral, user", "neutral_tool": "neutral, tool",
             "cueA_user": "cue$\\to$A, user", "cueB_user": "cue$\\to$B, user",
             "cueA_tool": "cue$\\to$A, tool", "cueB_tool": "cue$\\to$B, tool"}
    for cell, q in a["cells"].items():
        med = statistics.median(toks[cell])
        pc = "--" if q["P_follow"] is None else f"{q['P_follow']:.3f}"
        rows.append(f"{names[cell]} & {q['n']} & {q['trunc_rate']:.3f} & {q['parse_fail_rate']:.3f} & "
                    f"{q['P_answer_A']:.3f} & {pc} & {med:.1f} \\\\")
        nums.append(f"cell {cell}: n={q['n']} trunc={q['trunc_rate']} parsefail={q['parse_fail_rate']} "
                    f"P(A)={q['P_answer_A']} P(cued)={q['P_follow']} median_tokens={med}")
    (FIG / "tab_cells.tex").write_text(
        "% generated by paper/make_figures.py from results/qwen3-0.6b/analysis.json and gens.jsonl\n"
        + "\n".join(rows) + "\n")
    (FIG / "numbers.txt").write_text("\n".join(nums) + "\n")
    print("\n".join(nums))


if __name__ == "__main__":
    main()
