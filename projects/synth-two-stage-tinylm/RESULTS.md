# RESULTS: synth-two-stage-tinylm (Family A + matched-token real-only control)

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

**Question.** Li & Zou (arXiv 2609.09572, linear-regression theory) predict that training on synthetic data first and real data second (S→R) beats mixing them, and avoids an error floor. Does this hold for a tiny GPT (0.79M non-embedding params, d=128, 4 layers, 2048 BPE) on TinyStories? This is an independent conceptual test and extension, not a replication of their LM setup: Li & Zou report their own small-LM test (WikiText-103, per search snippets; their Appendix H.2 is still unread), ours differs in data (TinyStories), recency controls, N-scaling and a matched-token baseline.

**Headline (n=3 seeds, R_val perplexity, lower is better; all runs 4 threads, LR 3e-3, one cosine schedule):**

| N real tok | arm (total tokens) | R_val ppl mean ± std |
|---|---|---|
| 4M | real-only (4M) | 25.99 ± 0.42 |
| 4M | mixed (8M) | 17.39 ± 0.20 |
| 4M | S→R (8M) | 16.12 ± 0.28 |
| 4M | R→S (8M) | 19.11 ± 0.35 |
| 4M | mixed→real-tail (12M) | 13.30 ± 0.09 |
| 4M | real-only-2N (8.38M fresh real, matched total) | 15.23 ± 0.21 |
| 4M | **real-only-2ep (same 4M real tokens ×2, matched total)** | **15.18 ± 0.14** |
| 1M | real-only (1M) | 77.62 ± 1.88 |
| 1M | mixed / S→R / R→S (2M) | 44.58 ± 0.47 / 44.37 ± 0.32 / 45.55 ± 1.54 |
| 1M | real-only-2N (2.1M) / real-only-2ep | 44.20 ± 1.48 / 44.72 ± 1.25 |
| 1M | mixed→real-tail (3M) | 32.56 ± 0.64 |

Full tables with paired differences and CIs: `results/analysis.md` (`src/analyze.py`).

**Findings.**
1. *Ordering effect at 4M (S→R < mixed):* S→R beats mixed by 1.27 ppl (hierarchical bootstrap 95% CI [−1.42, −1.11]; per-seed −1.27, −1.43, −1.11) and beats R→S by 2.99. At 1M the S→R vs mixed difference is within noise (−0.21, CI [−0.79, +0.22], seed signs inconsistent).
2. *H1 (pre-registered rule: g = ppl(mixed) − ppl(S→R) > 0 at 4M for all seeds, and 4M interval excludes 0, and g(4M) > g(1M) with CI excluding 0): met.* *H1 contrast g(4M)−g(1M)* (computed as ppl(S→R) − ppl(mixed), i.e. −1 × the pre-registered g) = −1.06, CI [−1.36, −0.64], per-seed −1.24, −0.63, −1.33: the S→R advantage grows with N in this range (two N values only; no claim about an asymptotic floor).
3. *Controls for the step-count confound (matched total tokens).* Two real-only arms: `real-only-2N` (2N fresh real tokens) and `real-only-2ep` (Family B: the same N real tokens, 2 epochs; at 1M fresh real data beats repeated real by 0.52 ppl, CI [+0.05, +1.31]). At 4M, replacing N synthetic tokens with N additional fresh real tokens is better: real-only-2N beats S→R by 0.89 ppl (S→R − 2N = +0.89, CI [+0.64, +1.05], all seeds) and mixed by 2.16. At 1M, mixed (−0.15, CI [−1.50, +0.63]) and S→R (−0.36, CI [−1.28, +0.55]) are indistinguishable from `real-only-2ep`, while all three beat real-only(N) by ~33 ppl: i.e. at 1M the large gain of mixed/S→R over real-only(N) is consistent with total tokens/steps, with no detectable benefit of the synthetic tokens over repeating the real ones. At 4M, `real-only-2ep` (15.18 ± 0.14) beats S→R by 0.94 ppl (S→R − 2ep = +0.94, CI [+0.82, +1.09], all seeds) and mixed by 2.21, and ties `real-only-2N` (−0.05, CI [−0.19, +0.11]). So the gain of S→R/mixed over real-only(N) is accounted for by total tokens/steps at 1M (no detectable difference from real-only-2ep) and more than accounted for at 4M (real-only-2ep better by 0.94 ppl): in this setup, a second epoch over the same real tokens did at least as well as adding N G0 synthetic tokens; the S→R > mixed ordering effect holds, but neither synthetic arm beats matched-token real-only. Whether synthetic data helps when real data would need many more than 2 epochs (heavier repetition) was not tested.
4. mixed→real-tail wins because it sees more real tokens (2N real + N synthetic) and a longer run; it is not comparable and is reported descriptively.

**Data and licensing.** Real data: TinyStories (Eldan & Li 2023, arXiv 2305.07759; HF `roneneldan/TinyStories`, CDLA-Sharing-1.0, attribution required). G0-generated synthetic text and derived token splits are derivative data: if ever published they must remain under CDLA-Sharing-1.0 with attribution; none is committed here (`data/`, `ckpt/` are gitignored).

**Not done / limitations.** One dataset, one tiny model, one synthetic fraction (0.5), one generator depth, no per-stage-restart schedule ablation (the continuous cosine schedule does not satisfy the theorem's identical-real-stage condition, so H2-style comparisons are descriptive), no N=2M, Family B is 2 epochs only (no larger repeat counts). LR chosen at 1M tokens on R_dev (3e-3) may not transfer to 4M. Seed-level bootstrap with 3 seeds is coarse; per-seed signs are the primary evidence. G0 samples have 0.75% empty docs (filtered; deviation logged in PLAN.md) and shorter docs than real. At N=4M mixed→real-tail overlaps its real slices by 11,633 tokens. Semantic Scholar was rate-limited during the plan-stage novelty check; a later scout (2026-10-07) found no empirical ordering comparison on small LMs (adjacent: 2412.14689, 2510.01631, not read in full).

**Reproduce.** `./fetch_data.sh`; `uv venv .venv && UV_TORCH_BACKEND=cpu uv pip install -r requirements.txt`; build tokenizer/splits `src/tok.py`; G0 + S1: `results/commands_G0_S1.txt`; `python src/make_plans.py`; `BUDGET_S=... bash src/run_grid.sh`; `bash src/run_extra.sh`; `bash src/run_familyB.sh`; `.venv/bin/python -I src/analyze.py > results/analysis.md`. Environment: Python 3.14.4, torch 2.14.1+cpu, numpy 2.5.3, tokenizers 0.23.2. Data: roneneldan/TinyStories (CDLA-Sharing-1.0, commit pinned in fetch_data.sh). Raw per-run logs: `results/grid/*/`. Code/analysis commit: the commit containing this file (exact hash to be added in a follow-up once all runs finish).
