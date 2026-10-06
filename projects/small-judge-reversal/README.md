# small-judge-reversal

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

**Abstract.** Can tiny open LLM judges (0.5-1.7B) follow a reversed criterion ("which response is WORSE?") in pairwise judging? We ran Qwen2.5-0.5B/1.5B-Instruct and Qwen3-0.6B/1.7B on 300 RewardBench pairs (both orders, both criteria) and classified each pair as correct reversal, position-locked, criterion-blind, reversed-consistent or other. Result: essentially none reverse correctly (0, 0, 0, 1 of 300). The smallest judges are position-locked (Qwen2.5-0.5B answers "A" on every pass; Qwen3-0.6B 89% of pairs); the 1.7B shows less position-locking but shifts to inconsistent behaviour. H2 (position-locking dominates) supported; H1 (flip rate rises with size) inconclusive because almost no pair is correct under "better" in both orders for the smallest models.

Method: PLAN.md. Results and limitations: RESULTS.md. Tables: results/analysis.md.

## Reproduce
```
cd projects/small-judge-reversal
uv venv .venv --python 3.14 && uv pip install --python .venv/bin/python -r requirements.txt
./fetch_data.sh                       # pinned model + dataset revisions (several GB)
OMP_NUM_THREADS=4 .venv/bin/python -I src/run.py prepare
for M in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen3-0.6B Qwen3-1.7B; do for s in 0 1 2; do
  OMP_NUM_THREADS=4 .venv/bin/python -I src/run.py run --model $M --resample $s; done; done   # ~25 min per Qwen3-1.7B/Qwen2.5-1.5B run, 4-11 min for the 0.5-0.6B (measured, ~4 h total)
.venv/bin/python -I src/analyze.py    # reads committed results/raw_*.json
```
Raw outputs for the reported run are committed in `results/`, so only the last line is needed to re-derive the tables.
