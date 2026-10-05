# North Star

> Written by Pranav. The agent reads this every cycle and must never edit it.

## What this lab is
An always-on lab that tracks what is happening in AI *right now*, finds the open questions inside the
biggest current developments, and turns the best ones into small, finished, rigorous projects.
Discovery is half the job: the lab should always be actively scouting for the next thing worth a project.

## Goals
1. Stay on the frontier: every project connects to something the field is actively discussing this month
   (new models, papers, methods, benchmarks, debates).
2. Turn trends into findings: replicate, stress-test, ablate, or extend a hot claim at small scale on CPU.
   A sharp negative result on a hyped claim counts.
3. 1-2 workshop-quality papers per year I can extend and submit (NeurIPS/ICLR/ACL workshops, SRW tracks).
4. Every project ends with a write-up I can read in 10 minutes.

## Standing scouting mandate (always on)
- Keep `radar.md` current: the top ~10 things happening in AI right now. Each entry: 2+ sources (arXiv,
  Hugging Face papers/trending, Hacker News, Semantic Scholar, GitHub), date seen, why it matters, and a
  CPU-feasible project angle. Refresh at least daily; drop stale items.
- Hunt for: major papers and model releases, methods everyone is adopting, claims nobody has replicated,
  benchmark numbers that look too good, debates a small experiment could settle.
- Promote the best radar items into `backlog.md`, scored: trend importance × novelty × CPU feasibility ×
  chance of a finished write-up.
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
- First: build `radar.md` from a fresh sweep of the last ~2 weeks of AI papers and news.
- Then pick ONE radar item and finish a small project on it end-to-end, to prove the pipeline.
- Keep >=10 sourced, scored candidates in `backlog.md`.
