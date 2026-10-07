# synth-two-stage-tinylm

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

Tiny-GPT test of the "synthetic first, then real" prediction of Li & Zou (2609.09572) on TinyStories, with recency and matched-token controls. Result (3 seeds): S→R beats mixed by ~1.3 ppl at 4M real tokens (not at 1M), but at matched total tokens, N fresh real tokens beat N synthetic tokens at 4M, and at 1M mixed/S→R are no better than repeating the real tokens for 2 epochs. Repeating the same real tokens twice (matched total tokens) also beats S→R at 4M (by 0.94 ppl), so in this setup (TinyStories, one ~1M-param-class model, 50% G0 synthetic, 2 epochs of repetition) we detect no benefit of G0 synthetic tokens over repeating the real tokens; at 4M repetition is better. See `RESULTS.md`; analysis `results/analysis.md`.

Data: TinyStories (Eldan & Li 2023, arXiv 2305.07759; `roneneldan/TinyStories`, CDLA-Sharing-1.0). No data or checkpoints are committed.
