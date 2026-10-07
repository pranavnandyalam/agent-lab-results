# agent-lab

Autonomous AI/ML research lab. See `north-star.md` for mission, `radar.md` for current AI trends being
tracked, `backlog.md` for the ranked project queue, and `STATE.md` for what's happening right now.

## Projects

| slug | status | one-line summary |
|---|---|---|
| [dpo-grad-reweight](projects/dpo-grad-reweight/) | ARCHIVED | DPO baseline (Qwen2.5-0.5B, CPU) failed the pre-registered mean-loss rule at all 3 LRs, so no GAW-PO-lite comparison was run. Aborted experiment, no result; honest write-up. |
| [small-judge-reversal](projects/small-judge-reversal/) | DONE (paper: [pdf](projects/small-judge-reversal/paper/main.pdf)) | Does pairwise criterion reversal ("pick the worse") break small 0.5-1.7B judges? No correct reversal in 300 pairs for any of 4 judges; calibrated analysis shows criterion-blindness (corrected after referee review). |
| [synth-two-stage-tinylm](projects/synth-two-stage-tinylm/) | IN PROGRESS | Does two-stage training (synthetic then real) beat mixing on a tiny GPT trained on TinyStories? Independent test + extension of Li&Zou 2609.09572; Grid complete; S-R > mixed at 4M, but matched-token real-only is as good or better; Family B control done: repeating real tokens matches/beats synthetic arms; RESULTS awaiting overseer re-review. |
| [quant-cot-looping](projects/quant-cot-looping/) | DONE (team beta; [paper](projects/quant-cot-looping/paper/main.pdf)) | Qwen3-0.6B weight quantization: loop tokens are only ~10-14% of added CoT length at 4.5-5.5 bits; 4.0 bits collapses into loops. 16 problems, preliminary. |
| [cot-monitor-small](projects/cot-monitor-small/) | ARCHIVED as pilot (team beta; [paper](projects/cot-monitor-small/paper/main.pdf)); 3 referee rounds, rebuild needed for stronger claims | Do 0.5-1.7B monitors catch hacks only the reasoning reveals? Matched synthetic transcripts, keyword-vs-intent control. |

## AI authorship

This repo's commits, code, and write-ups are produced by an autonomous AI agent (Claude) on behalf of
@pranavnandyalam, operating under the rules in `north-star.md` and the lab's operating manual. Not peer
reviewed. Every project README/RESULTS.md repeats this note.
