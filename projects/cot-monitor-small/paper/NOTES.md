# Paper notes: cot-monitor-small

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

Build: `cd projects/cot-monitor-small/paper && timeout 300 latexmk -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex && latexmk -c` (7 pages: body ends on p. 6, references on pp. 6-7; no overfull boxes; revised 2026-10-07 after referee round 1; revised again after referee round 2, still 7 pages, 0 overfull boxes). Fig. 2 (fig_effects.pdf) is still generated but no longer included in the paper.
Figures: `python -I paper/make_figures.py` from the project dir; needs matplotlib, which is not in the project venv (used a throwaway venv with matplotlib 3.11.2, numpy 2.5.3, Python 3.14.4). Input: `results/analysis_test.json` only.

## Number provenance
- Round 2 tables VI-IX (K per prompt with CIs and sign-flip p, per-family action AUROC with CIs and no-cue columns, hack-variant breakdown, per-family O): `results/posthoc_test_r2.md`; all 257 'value [lo, hi]' triples in main.tex were checked by script to occur verbatim in posthoc_test_r2.md or analysis_test.md.
- Tables VII-VIII (post-hoc, round 1): `results/posthoc_test.md`, checked cell by cell with a script (dev K no longer used).
- Tables IV-VI: `results/analysis_test.md` / `.json`, skeleton-bootstrap unit; checked cell by cell against the JSON with a script.
- Table III (dev): `results/dev_summary.json` (`auroc_B_vs_A`, `mean_auroc_B_vs_A`), rounded to 3 d.p.
- Revisions: `results/scores_test_v2_*.meta.json`. Timings: `results/timings.jsonl`.
- The one optimistic 72-cluster number quoted (Qwen3-0.6B CoT-only Acue-A 0.218 [0.001, 0.431]) is from the family_x_p table and is labeled as optimistic and not used for a verdict.
- Derived, not in RESULTS.md: "changes it by at most 0.045" (max |B-A both - B-A cot| over the three other monitors, 0.962-0.916), "logged times sum to about 1.6 hours" (5910.67 s), and the observation that v2 was not best on dev for the two larger monitors (v1 0.997/1.000, v3 0.998/0.911).

## References
14 of 14 verified (2608.04735 and 2609.19101 added in round 2, arXiv API 2026-10-07); earlier: 12 of 12 verified via the arXiv API (10 on 2026-10-06; 2511.08525 and 2601.05752 on 2026-10-07, ACL 2026 Findings per the arXiv comment of 2601.05752) (title/authors/year; venue only where the arXiv comment states it: NeurIPS 2025 for 2505.23575, ICLR 2026 for 2510.01367). Model details of 2608.00583 (gpt-5.4-nano / claude-sonnet-4.6 / gemini-2.5-flash; Qwen3-8B/32B surrogates) checked against its arXiv HTML v3. Qwen tech reports cite truncated author lists ("others").

## Weaknesses a reviewer will attack
1. Synthetic, agent-written, lexically regular templates; the same agent wrote templates and hypotheses. No real agent traces.
2. Effective n is about 12 (shared skeletons). Percentile bootstrap with 12 clusters may under-cover; many estimands, no multiplicity correction.
3. BoW reaches 1.000 when wording leaks across families and 0.758 without; monitor AUROCs cannot be attributed to intent reading. The skeleton-grouped BoW rule is near-unfalsifiable.
4. H2 untestable due to the global ceiling rule; a per-pair rule would have marked Qwen2.5 supported (CI lower bound 0.002). A reviewer may call the global rule convenient; it was pre-registered before the test analysis (asserted).
5. H3 not supported because the Acue condition has very wide CIs; also Acue uses negated cues ("I will not cheat"), which conflates cue sensitivity with negation handling.
6. H4 "supported" only via a monitor that does not detect hack actions at all (both AUROCs < 0.5).
7. Single prompt per model; strong prompt sensitivity on dev (v1 inverts Qwen2.5-0.5B at 0.207). v2 contains cue stems and may prime K and Acue.
8. Selection rule averages over monitors; v2 was not the best prompt for the two larger monitors on dev (v1: 0.997, 1.000).
9. Pre-registration timing: dev_select.py written after the first dev file; the analysis-choice pre-registration was committed in the same commit (42188b9) as the first monitor's test scores, so ordering is not demonstrable from git.
10. Dev and test share task/action/slot templates; only reasoning skeletons differ.
11. Toy one-line actions; action-only near chance may reflect our action format rather than monitor ability.
12. CPU model not recorded; 128 Qwen2.5-1.5B passes have no logged timing.
13. Novelty is narrow (sub-2B scale + keyword minimal pair) and rests on a limited arXiv/web search (Semantic Scholar 429; no GitHub search).

## Status note
README.md still says "under overseer/ethics review"; the Lead stated RESULTS.md is approved. README status should be updated by the Lead.
