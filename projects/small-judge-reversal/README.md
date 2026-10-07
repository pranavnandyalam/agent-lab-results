# small-judge-reversal

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

**Abstract.** Can tiny open LLM judges (0.5-1.7B) follow a reversed criterion ("which response is WORSE?") in pairwise judging? We ran Qwen2.5-0.5B/1.5B-Instruct and Qwen3-0.6B/1.7B on 300 RewardBench pairs (both orders, both criteria) and classified each pair as correct reversal, position-locked, criterion-blind, reversed-consistent or other. Result: essentially none reverse correctly under argmax (0, 0, 0, 1 of 300). **Corrected after referee review:** a letter-offset-calibrated re-analysis shows content preference under "better" and "worse" is strongly positively correlated (r=0.63-0.95), i.e. the judges are criterion-blind; argmax position-locking is explained by a letter offset (post-hoc analysis) and H2 is near-tautological against a uniform null (see RESULTS.md). H1 (flip rate rises with size) inconclusive because almost no pair is correct under "better" in both orders for the smallest models. **Larger judge (cycle 22, post hoc):** Qwen3-4B on resample 1 only (100 pairs) shows much weaker coupling (r=0.10 [-0.09,0.30], same sign 0.59, calibrated accuracy 0.86, calibrated reversal 0.31 [0.23,0.40]); on the same 100 pairs its r is 0.475 below Qwen3-1.7B's (paired bootstrap CI [-0.72,-0.229]). It is only a partial positive control (r and slope ~0, not negative); n=100, one resample, one family, so a scale trend is suggestive, not established.

Method: PLAN.md. Results and limitations: RESULTS.md. Tables: results/analysis.md.

## Reproduce
```
cd projects/small-judge-reversal
uv venv .venv --python 3.14 && uv pip install --python .venv/bin/python -r requirements.txt
./fetch_data.sh                       # pinned model + dataset revisions (several GB)
OMP_NUM_THREADS=4 .venv/bin/python -I src/run.py prepare
for M in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen3-0.6B Qwen3-1.7B; do for s in 0 1 2; do
  OMP_NUM_THREADS=4 .venv/bin/python -I src/run.py run --model $M --resample $s; done; done   # ~25 min per Qwen3-1.7B/Qwen2.5-1.5B run, 4-11 min for the 0.5-0.6B (measured, ~4 h total)
OMP_NUM_THREADS=4 .venv/bin/python -I src/run.py run --model Qwen3-4B --resample 1   # positive control, resample 1 only (1848 s summed forward time); fetch_data.sh does NOT download Qwen3-4B: fetch Qwen/Qwen3-4B at rev 1cfa9a7208912126459214e8b04321603b3df60c into the same HF cache first
.venv/bin/python -I src/calibrated.py   # -> results/calibrated.{md,json}, incl. Qwen3-4B rows
.venv/bin/python -I src/analyze.py    # reads committed results/raw_*.json
```
Raw outputs for the reported run are committed in `results/`, so only the analysis lines are needed to re-derive the tables. Caveat: `src/sjr/config.py` now includes Qwen3-4B, for which only resample 1 exists, and `src/analyze.py` skips Qwen3-4B, so the argmax tables cover the 4 core judges only (re-run: output identical). Same-pair comparison of all five judges: `python -I paper/make_figures.py` (venv from `paper/requirements-paper.txt`) -> `paper/figures/paper_stats.json` (`positive_control_r1`).
