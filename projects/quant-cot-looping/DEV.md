# Detector development (train problems only; cycle 3, 2026-10-06)

Dev data: 8 GSM8K train problems x 3 seeds, 2048 cap, Qwen3-0.6B. w4g64 (24 traces, results/timed_trial_w4.json), fp32 (24 traces, results/timed_trial_fp32.json seed 0 + timed_trial_fp32_s12.json seeds 1-2).

Dev facts (not test results):
- fp32: 2/24 truncated at 2048, mean 1104-1260 tokens. w4g64: 14/24 truncated.
- Detector as in PLAN (20-gram x4 in 1024 tokens, OR >=8-gram x3 consecutive) flags 0/24 fp32 traces and 7/24 w4 traces (same 7 traces before and after the marking fix), 63-462 loop tokens each (after the fix below; <=23% of the trace). Truncation is NOT mostly looping: 7 of 14 truncated w4 traces have zero flagged tokens.
- Hand-read of the 7 flagged w4 traces (labeller: Lead, one read, no second rater): 1 clear degenerate loop (5646 s2: "Let me think of the total number of pages" x many); 3 repeated re-quotes of the problem text in the model's own words (3950 s0/s1, 5624 s1: stuck re-reading, a soft loop); 3 traces (63-147 flagged tokens each; the spans were read on the earlier detector version) that look like legitimate re-statement (false positives: 5441 s2, 1485 s1, 824 s2). Strict precision of "degenerate loop" is about 1/7 traces, lenient (incl. soft loops) 4/7.
- Misses (recall): of 7 unflagged truncated w4 traces, one (5646 s0) loops on short phrases ("Let me think of it as 3 days") that fall below 20 tokens; three were mid-reasoning; two had already boxed an answer and were cut inside the post-answer summary (5624 s0, 3950 s2 close to it). So the 2048 cap also counts traces that already answered.
- Change made on dev: 20-grams copied from the problem text are excluded from rule (A). It changed little (the flagged re-quotes are the model's paraphrase).
- Short-n variants (dev only): 12-gram x4 (rule A only) flags 21/48 traces including 4 fp32 traces (17/24 w4) (5624 s0 fp32: 603 loop tokens in a structured recap) - too permissive as a primary definition, kept as a pre-declared sensitivity variant (12-gram x4).

Frozen: rule (A) 20-gram x4/1024 excluding problem-text n-grams, rule (B) >=8-gram x3 consecutive (n=8..64), src/detector.py at the commit that adds this file. Sensitivity variants for the analysis: 12-gram x4 and 16-gram x4 (rule A only).

Freeze-time corrections after overseer DIFF review: (1) rule (A) originally marked only the 4th+ occurrence span; changed to mark the 2nd..current occurrence spans once a 20-gram reaches 4 occurrences in the window, matching PLAN's "tokens in repeats after the first occurrence" (flag set unchanged; numbers above are from the corrected detector, results/dev_analysis.txt, script src/dev_analysis.py). (2) Dev traces were retokenized from saved text; the core run applies the detector to the true generated token ids (saved in results/core). (3) Fewer hand-labels than planned: 7 flagged + 7 unflagged truncated traces read, one rater, so no recall estimate is claimed.
