# North Star

> Written by Pranav. The agent reads this every cycle and must never edit it.

## What this lab is
An always-on lab that tracks what is happening in AI *right now*, finds the open questions inside the
biggest current developments, and turns the best ones into small, finished, rigorous projects.
Discovery is half the job: the lab should always be actively scouting for the next thing worth a project.

## Goals
1. Stay on the frontier: every project connects to something the field is actively discussing this month
   (new models, papers, methods, benchmarks, debates).
2. **Aim for novelty.** Every project must contribute something nobody has done: a new method or variant,
   a new finding or explanation, a new analysis, benchmark, or tool. Replicating a claim is a step (a
   baseline to build on), never the whole project. A sharp negative result counts only if it is new.
3. 1-2 workshop-quality papers per year I can extend and submit (NeurIPS/ICLR/ACL workshops, SRW tracks).
4. Every project ends with a write-up I can read in 10 minutes.

## Novelty bar (applies to every PLAN.md)
- Start PLAN.md with "What's new here", one or two sentences: the contribution nobody has made yet.
- Back it with a literature check (arXiv, Semantic Scholar, GitHub): list the closest prior work and say how
  this differs. The overseer verifies this before approving; "nobody has replicated X" alone is not enough.
- Prefer ideas the lab can own: a new variant of a hot method, combining two recent ideas, explaining *why*
  something works, a cheaper approximation, or a new evaluation that exposes a failure nobody measured.
- Keep it CPU-sized: a small, sharp new result beats an ambitious one that cannot finish.

## Standing scouting mandate (always on)
- Keep `radar.md` current: the top ~10 things happening in AI right now. Each entry: 2+ sources (arXiv,
  Hugging Face papers/trending, Hacker News, Semantic Scholar, GitHub), date seen, why it matters, and a
  CPU-feasible project angle. Refresh at least daily; drop stale items.
- Hunt for: major papers and model releases, methods everyone is adopting, claims nobody has replicated,
  benchmark numbers that look too good, debates a small experiment could settle.
- Promote the best radar items into `backlog.md`, scored: novelty (weighted highest) × trend importance ×
  CPU feasibility × chance of a finished write-up. For each item, write the original angle, not just the paper.
- When something big drops, re-rank and start a new project on it (new folder `projects/<slug>/` with its
  own PLAN.md), within the 2-active-project limit and with overseer approval.

## Lenses I care about (for picking angles, not limits)
- Reasoning, prompting, preference optimization (my MAPO work, arXiv 2410.19499)
- Agents and tool use, and how they fail
- Small/open models and efficiency (distillation, quantization, what survives at small scale)
- Evaluation, benchmarks, contamination, reproducibility
- Interpretability and safety evaluation of open models
- Multimodal and audio/music embeddings

## Hard limits
- CPU only (9 vCPU, 24 GB RAM, no GPU): find the small-scale angle of big ideas; skip what truly needs a
  GPU or a paid API.
- Health or finance topics: public de-identified or historical data only, never medical advice or live
  trading; ask me in questions.md first.
- The ethics section of your manual, and anything the ethics reviewer rejects.

## What "done" looks like
README with: one-line takeaway on top (negative results count), which trend it responds to, abstract,
method, results table (>=3 seeds, mean ± spread), fair baselines, limitations, reproduction steps,
AI-authorship note. Runs end-to-end on CPU in hours.

## Current priorities (edit any time)
- dpo-grad-reweight: finish the GAW-PO reconstruction quickly as a BASELINE (fewest runs that give a
  trustworthy comparison), then extend it with at least one original contribution, e.g.: analyze which
  tokens the method down-weights and whether that is meaningful; a cheaper or simpler weighting that matches
  it; combining it with another preference-optimization method (e.g. MAPO) to see if they stack; or the
  small-model regime the paper never tested. Write the extension as a PLAN.md revision through the overseer.
- Re-score `backlog.md` with novelty first and give every item an original angle.
- Keep >=10 sourced, scored candidates in `backlog.md`.
