# agent-lab

Autonomous AI/ML research lab. See `north-star.md` for mission, `radar.md` for current AI trends being
tracked, `backlog.md` for the ranked project queue, and `STATE.md` for what's happening right now.

## Projects

| slug | status | one-line summary |
|---|---|---|
| [dpo-grad-reweight](projects/dpo-grad-reweight/) | ARCHIVED | DPO baseline (Qwen2.5-0.5B, CPU) failed the pre-registered mean-loss rule at all 3 LRs, so no GAW-PO-lite comparison was run. Aborted experiment, no result; honest write-up. |
| [small-judge-reversal](projects/small-judge-reversal/) | BUILT (sweep pending) | Does pairwise criterion reversal ("pick the worse") break small 0.5-1.7B judges, via position-locking or criterion-blindness? Plan rev 3c approved; harness + trial done. |
| [quant-cot-looping](projects/quant-cot-looping/) | PLANNING (team beta) | Under weight quantization, how much of the longer CoT of Qwen3-0.6B is looping vs distinct reasoning? Plan rev 3 under review; core run awaits Pranav's OK (Q-20261006-2). |

## AI authorship

This repo's commits, code, and write-ups are produced by an autonomous AI agent (Claude) on behalf of
@pranavnandyalam, operating under the rules in `north-star.md` and the lab's operating manual. Not peer
reviewed. Every project README/RESULTS.md repeats this note.
