# Response to referee round 1

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

All new numbers come from `results/referee_r1_ablation.md` (raw: `results/referee_r1/r_val_ppl_raw.json`) and the RESULTS.md addendum (referee round 1). No other results changed.

1. **"Advantage grows with N" overclaims (only two budgets).** Done. Removed from the abstract, the Discussion and the Conclusion. In Results (H1 paragraph) the pre-registered H1 outcome is still reported, because it was pre-registered. It now says only that the S->R advantage is larger at 4M than at 1M, and that two budgets do not support a trend in N. The Conclusion explicitly makes no claim about scaling with data.

2. **Ordering claim in the title is too strong.** Done. New title: "Synthetic-First vs. Mixed Training in a Tiny Language Model: An Ordering Effect at One of Two Budgets, and No Gain over Matched-Token Real Data". The Discussion now says the ordering effect holds at N=4M only (none detectable at 1M).

3. **"Self-generated" is misleading.** Done. The paper now says "same-architecture generator" throughout. The Zhu et al. sentence now says "synthetic data".

4. **Li & Zou LM experiment (App. H.2) and venue.** Partly addressed, and only by disclosure. We still have **not** read App. H.2: the HTML was truncated and the PDF could not be parsed. Related Work now says this and states that no side-by-side comparison with their LM experiment is claimed. The differences we list are qualified as "as far as we can tell from the retrieved text". No venue is asserted. The bib entry is an arXiv preprint marked "publication venue not verified"; the referee's "ICML 2026 workshop" was not confirmed.

5. **LR-schedule-tail confound for S->R.** Done for S->R at N=4M (3 seeds). With a per-phase warmup+cosine restarted at the real phase, S->R gets 15.96 ± 0.13 vs. 16.12 ± 0.28 with the global schedule (per-seed −0.06, −0.09, −0.34). That is still about 1.4 better than mixed (17.39 ± 0.20) and about 0.8 worse than real-only-2ep (15.18 ± 0.14). This is new Section V-D and Table III, and it is mentioned in the abstract. No bootstrap CI was computed (n=3). Not done: per-phase runs for R->S, mixed->real-tail, or N=1M.

6. **LR sensitivity.** Only a hint. At LR 1e-3 we ran seed 0 only: mixed 22.88, S->R 22.19, real-only-2ep 21.99. The ranking is unchanged, the gaps are smaller, and all three are much worse than at 3e-3. The paper labels this a one-seed hint, not a result. Not done: other seeds, LRs above 3e-3, and per-arm re-tuning.

7. **Missing related work.** Done. I checked each one on its arXiv abs page (2026-10-07) and took titles and authors from there:
   - 2509.15248, Yang, Zhang, Liu, Hashimoto, Candès, Wang, Pang, "Synthetic bootstrapped pretraining"
   - 2508.10975, DatologyAI (Maini et al.), "BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining"
   - 2509.14786, Kim, Kotha, Liang, Hashimoto, "Pre-training under infinite compute"

   They appear in a new Related Work paragraph. It states that matched-token repetition baselines are not new, that well-built synthetic data beats repetition at large scale, and that our result concerns a weak generator. The novelty statement is narrowed to match. We have read only the abstracts of these three papers.

8. **Stronger generator, heavier repetition (4–16 epochs), N=2M, more budgets.** **Not done.** These are listed as limitations in Section VI (items iii–v) and as open questions in the Conclusion. The matched-token negative result therefore applies only to a same-architecture generator and 2-epoch repetition.

9. **Reproduction fixes.** `src/make_plans.py` now also writes the real-only-2N and real-only-2ep plans, and they are byte-identical to the committed ones. The `run_familyB.sh` comment is fixed. The Reproducibility section now includes `src/run_referee_r1.sh` and `results/referee_r1/make_table.py`.

Remaining weaknesses we acknowledge:
- There is one tiny model and one dataset.
- 3 seeds is coarse.
- No multiple-comparison correction was applied.
- The theorem's identical-real-stage condition is still not met for comparisons against real-only.
