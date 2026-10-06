team: main
# PLAN: synth-two-stage-tinylm (rev 3, 2026-10-06; rev1+rev2 overseer REJECT, rev 3 overseer APPROVE with conditions (applied), ethics APPROVE_WITH_CONDITIONS)

## What's new here
Li & Zou (arXiv 2609.09572) prove for SGD in high-dimensional linear regression that mixing synthetic+real data leaves an error floor as data grows, while two-stage training (synthetic then real, identical real-stage updates) avoids it. Marchi et al. (2609.18878) give a Fisher-Rao minimum-human-data-fraction theory (not about ordering). Empirical neighbours: Strong Model Collapse 2410.04840 (mixed only), Gerstgrasser 2404.01413 (replace vs accumulate), TinyStories recursive collapse 2412.14872 (pure synthetic), Silent Collapse 2605.14588 (fraction schedules), procedural pre-pretraining 2601.21725 / 2505.22308 (different goal). New here: a test of the floor-vs-no-floor prediction (perplexity vs real-token budget N) on a small GPT with recency controls. Novelty: "not found in a web-only scout, medium confidence"; a Semantic Scholar/GitHub check is required before any grid run (timed trial and fetch_data.sh may proceed first). All arXiv IDs get re-verified before RESULTS.

## Setup (one family, settles H2 source)
- Data: `roneneldan/TinyStories` (CDLA-Sharing-1.0; pin commit hash in fetch_data.sh; cite Eldan & Li 2023). Disjoint splits: R_gen (trains generator G0), R_train (real pool for arms), R_dev (LR/design choices only), R_val (final eval, never tuned on).
- Synthetic S: sampled at T=1.0 from G0 (trained on R_gen, same architecture/tokenizer). Depth 1 only for the main grid; depth 2 (G1 trained on S1, samples S2) only if compute allows.
- Model: GPT d=128, 4 layers, vocab 2048 BPE (tokenizers, trained on R_gen) ≈ 0.8M non-embedding params. Final size set by a timed 1M-token trial; budget FLOPs ≈ 6·P·tokens, expect ~minutes/run at 2-5M tokens.
- Real-token budgets N ∈ {1M, 2M, 4M} real tokens (R_train). Synthetic tokens S = N (constant synthetic fraction 0.5 in `mixed` at every N, matching the theorem's 'fixed fraction' regime); S->R uses the same S=N.
- ONE continuous LR schedule (warmup + cosine over the whole run) for all arms; a per-stage-restart ablation is reported separately only.

## Arm table (real tokens N; synthetic tokens S=N)
| arm | order | real tok | synth tok | notes |
|---|---|---|---|---|
| real-only | R | N | 0 | baseline (fewer total steps) |
| mixed | R+S interleaved | N | N | matched real tokens |
| two-stage S->R | S then R | N | N | real stage identical in tokens to real-only |
| reverse R->S | R then S | N | N | recency control |
| mixed->real-tail | (R+S mixed) then R | N | N | real tail of N real tokens preceded by a mixed phase of N synth + N real tokens (so uses 2N real total; breaks real-token match, labelled as such). Tail length matches S->R's real stage |
Family A (main): matched real tokens as above. Family B: matched total tokens (real-only gets repeated epochs to equal 2N total tokens). Each hypothesis states its family.

## Hypotheses and pre-registered decision rules (metric: R_val perplexity; 3 seeds, paired by seed; with 3 seeds 'CI excludes 0' is operationalised as: all 3 paired seed differences agree in sign AND a bootstrap over R_val documents (nested in seeds) 95% CI excludes 0)
- H1 (floor): in Family A, gap g(N)=PPL(mixed)-PPL(S->R). SUPPORTED if g>0 by the rule below at the largest N and g grows with N (g(4M)>g(1M), CI of difference excluding 0). With n=3 seeds, otherwise report "inconclusive/underpowered".
- H1b (ordering vs tail-length): S->R vs mixed->real-tail (equal-length real tail, but the latter saw 2N real tokens). Reported descriptively; a null is 'inconclusive', never 'recency explains it'.
- H2 (Family A): S->R vs real-only at same real tokens. The continuous cosine schedule does NOT give 'identical real-stage updates' (theorem condition), so H2 here is descriptive; the per-stage-restart variant (real stage = real-only run schedule exactly) is run for N=1M,4M as the H2-primary test. Reported either way (theory says two-stage can beat real-only only under a condition; no prediction forced).
- Arms/hypotheses are not changed after seeing results; deviations logged.

## Compute
Runs: Family A 5 arms x 3 N x 3 seeds = 45; Family B real-only (descriptive) 9; restart variant ~12; R_dev LR tuning ~6; est. ~200M training tokens, 5-25 h: the 6 h gate will likely trigger. Ordered cuts: (1) drop N=2M (H1 stays testable with N in {1M,4M}); (2) drop Family B; (3) restart variant only N=1M; (4) mixed->real-tail only N=1M,4M; (5) shrink d/tokens. Warmup = 5% of total steps in every arm. G0: trained on R_gen to a fixed budget; report its R_val perplexity.
Common: 4 threads, one job at a time, resumable (gitignored ckpt dir under the project), `timeout`-wrapped; deps torch (CPU, uv, pinned), tokenizers, datasets, pyarrow, numpy pinned in requirements.txt. Stop criterion: 3 cycles without a new result -> write up and archive. Non-embedding params ≈ 0.8M (12·d²·L); timed trial sets final size.

## Risks / limitations (to be in RESULTS)
One dataset, one tiny model, 3 seeds; no generalization to large models; linear-regression theory differs; real tokens are matched in Family A but total tokens differ (Family B covers the other); single synthetic fraction (no 0.25/0.75); no bias evaluation (children's-story text, low risk). Synthetic corpora stay local and uncommitted (CDLA-Sharing); only small, skimmed sample excerpts are committed. README/RESULTS carry the AI-authorship note and license/revision.

## Novelty to-do (before any grid run)
Record Semantic Scholar + GitHub queries, date, top hits, how each differs (S2 was 429-rate-limited 2026-10-06; retry, else arXiv-listing fallback). TinyStories hash + dep versions recorded in fetch_data.sh/requirements.txt and RESULTS.

## Novelty check result (2026-10-06, scout, web-search only; S2 API 429, arXiv export timed out)
Verdict PARTLY TAKEN, medium confidence. Li & Zou 2609.09572 themselves report a small-LM experiment (per search snippets, not fully read): ~16M 4-layer transformer on WikiText-103, DistilGPT-2 teacher, synthetic fractions 0.3/0.6/0.9; two-stage closer to real-only than mixed. Not found on TinyStories, nor with a real-token-budget axis, recency controls (reverse, mixed->real-tail) or Family A/B accounting. Other neighbours: 2609.38764 (abstract-data first phase), 2606.22942 (distillation), 2410.04840, 2404.01413, 2401.16380, 2510.08245. No GitHub repo found. Reframing: this is an independent replication + extension (recency controls, N-scaling), NOT a first test. To do: read the full text of 2609.09572's LM section before RESULTS.
