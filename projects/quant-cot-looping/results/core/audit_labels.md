# Manual precision audit labels (20 flagged traces, sample in audit_sample.md)
Labeller: Lead agent (Claude), single rater, read from the printed excerpts only (not full traces). D = degenerate loop, S = soft loop (stuck re-quoting/recomputing), F = false positive (legitimate re-statement).
1 F | 2 S | 3 F | 4 S | 5 S | 6 F | 7 D | 8 S | 9 D | 10 D | 11 D | 12 D | 13 D | 14 D | 15 D | 16 F | 17 D | 18 D | 19 D | 20 D
Totals: D 12, S 4, F 4. Precision (D+S)/20 = 0.80; strict D-only 0.60.
By level: w3g32 (items 7-20, n=14): D 12, S 1, F 1. fp32/w5g64/w4g64 (items 1-6, n=6): D 0, S 3, F 3 -> half of flags at the non-collapsed levels are re-quoted problem text or short recomputation, not degenerate loops. Suggests over-counting at non-collapsed levels; n=6, single rater, no recall estimate; effect on the H1 share is not established.
