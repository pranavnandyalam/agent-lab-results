# Family A analysis (R_val perplexity; lower is better; PRELIMINARY unless n=3)

| N | arm | n seeds | R_val ppl mean ± std | per-seed |
|---|---|---|---|---|
| 1M | real-only | 3 | 77.62 ± 1.88 | 76.62, 76.45, 79.79 |
| 1M | real-only-2N | 3 | 44.20 ± 1.48 | 43.42, 43.27, 45.91 |
| 1M | real-only-2ep | 3 | 44.72 ± 1.25 | 43.54, 44.61, 46.03 |
| 1M | mixed | 3 | 44.58 ± 0.47 | 44.14, 45.08, 44.51 |
| 1M | S-R | 3 | 44.37 ± 0.32 | 44.11, 44.27, 44.72 |
| 1M | R-S | 3 | 45.55 ± 1.54 | 44.05, 45.46, 47.13 |
| 1M | mixed-realtail | 3 | 32.56 ± 0.64 | 32.21, 32.17, 33.30 |
| 4M | real-only | 3 | 25.99 ± 0.42 | 26.23, 25.50, 26.22 |
| 4M | real-only-2N | 3 | 15.23 ± 0.21 | 15.26, 15.01, 15.42 |
| 4M | real-only-2ep | 3 | 15.18 ± 0.14 | 15.06, 15.13, 15.34 |
| 4M | mixed | 3 | 17.39 ± 0.20 | 17.16, 17.46, 17.55 |
| 4M | S-R | 3 | 16.12 ± 0.28 | 15.89, 16.03, 16.44 |
| 4M | R-S | 3 | 19.11 ± 0.35 | 18.71, 19.28, 19.34 |
| 4M | mixed-realtail | 3 | 13.30 ± 0.09 | 13.39, 13.22, 13.29 |

## Paired differences (a − b, ppl; negative = a better)

| N | a vs b | n | mean diff | hierarchical-bootstrap 95% CI | per-seed diffs | rule met (sign agree & CI excl. 0) |
|---|---|---|---|---|---|---|
| 1M | S-R − mixed | 3 | -0.21 | [-0.79, +0.22] | -0.03, -0.81, +0.22 | False |
| 1M | S-R − real-only | 3 | -33.25 | [-34.97, -31.89] | -32.51, -32.18, -35.06 | True |
| 1M | mixed − real-only | 3 | -33.04 | [-35.21, -31.29] | -32.48, -31.37, -35.28 | True |
| 1M | S-R − R-S | 3 | -1.18 | [-2.38, +0.04] | +0.06, -1.19, -2.40 | False |
| 1M | S-R − mixed-realtail | 3 | +11.81 | [+11.41, +12.16] | +11.90, +12.10, +11.42 | True |
| 1M | mixed − real-only-2N | 3 | +0.38 | [-1.39, +1.79] | +0.72, +1.81, -1.40 | False |
| 1M | S-R − real-only-2N | 3 | +0.17 | [-1.16, +1.01] | +0.69, +1.00, -1.18 | False |
| 1M | mixed − real-only-2ep | 3 | -0.15 | [-1.50, +0.63] | +0.60, +0.48, -1.52 | False |
| 1M | S-R − real-only-2ep | 3 | -0.36 | [-1.28, +0.55] | +0.57, -0.33, -1.30 | False |
| 1M | real-only-2ep − real-only | 3 | -32.90 | [-33.91, -31.80] | -33.09, -31.84, -33.76 | True |
| 1M | real-only-2ep − real-only-2N | 3 | +0.52 | [+0.05, +1.31] | +0.12, +1.34, +0.12 | True |
| 4M | S-R − mixed | 3 | -1.27 | [-1.42, -1.11] | -1.27, -1.43, -1.11 | True |
| 4M | S-R − real-only | 3 | -9.87 | [-10.36, -9.43] | -10.34, -9.47, -9.79 | True |
| 4M | mixed − real-only | 3 | -8.59 | [-9.09, -8.05] | -9.07, -8.04, -8.68 | True |
| 4M | S-R − R-S | 3 | -2.99 | [-3.23, -2.80] | -2.82, -3.25, -2.91 | True |
| 4M | S-R − mixed-realtail | 3 | +2.82 | [+2.51, +3.14] | +2.50, +2.81, +3.15 | True |
| 4M | mixed − real-only-2N | 3 | +2.16 | [+1.91, +2.43] | +1.91, +2.45, +2.13 | True |
| 4M | S-R − real-only-2N | 3 | +0.89 | [+0.64, +1.05] | +0.64, +1.02, +1.02 | True |
| 4M | mixed − real-only-2ep | 3 | +2.21 | [+2.09, +2.34] | +2.10, +2.33, +2.21 | True |
| 4M | S-R − real-only-2ep | 3 | +0.94 | [+0.82, +1.09] | +0.83, +0.90, +1.10 | True |
| 4M | real-only-2ep − real-only | 3 | -10.81 | [-11.22, -10.34] | -11.17, -10.37, -10.88 | True |
| 4M | real-only-2ep − real-only-2N | 3 | -0.05 | [-0.19, +0.11] | -0.19, +0.12, -0.08 | False |

## H1 contrast: g(4M) - g(1M), g = ppl(S-R) - ppl(mixed) (negative = S-R advantage grows with N)

n seeds = 3; mean -1.06; hierarchical-bootstrap 95% CI [-1.36, -0.64]; per-seed -1.24, -0.63, -1.33

Note: with 3 seeds the seed-level bootstrap is coarse; per-seed signs are the primary evidence. Rows with n<3 are preliminary and never count as 'rule met'.
real-only-2ep = Family B: the same N real tokens, 2 epochs (matched total tokens/steps, no extra fresh real data).
real-only-2N = real-only with matched TOTAL tokens (2N; 4M capped at R_train length 8.38M) to separate step-count from synthetic-data effects.
