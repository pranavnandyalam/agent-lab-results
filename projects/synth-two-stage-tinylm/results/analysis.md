# Family A analysis (R_val perplexity; lower is better; PRELIMINARY unless n=3)

| N | arm | n seeds | R_val ppl mean ± std | per-seed |
|---|---|---|---|---|
| 1M | real-only | 3 | 77.62 ± 1.88 | 76.62, 76.45, 79.79 |
| 1M | mixed | 3 | 44.58 ± 0.47 | 44.14, 45.08, 44.51 |
| 1M | S-R | 3 | 44.37 ± 0.32 | 44.11, 44.27, 44.72 |
| 1M | R-S | 3 | 45.55 ± 1.54 | 44.05, 45.46, 47.13 |
| 1M | mixed-realtail | 3 | 32.56 ± 0.64 | 32.21, 32.17, 33.30 |
| 4M | real-only | 2 | 25.87 ± 0.52 | 26.23, 25.50 |
| 4M | mixed | 2 | 17.31 ± 0.21 | 17.16, 17.46 |
| 4M | S-R | 2 | 15.96 ± 0.09 | 15.89, 16.03 |
| 4M | R-S | 2 | 19.00 ± 0.40 | 18.71, 19.28 |
| 4M | mixed-realtail | 2 | 13.30 ± 0.12 | 13.39, 13.22 |

## Paired differences (a − b, ppl; negative = a better)

| N | a vs b | n | mean diff | doc-bootstrap 95% CI | per-seed diffs | rule met (sign agree & CI excl. 0) |
|---|---|---|---|---|---|---|
| 1M | S-R − mixed | 3 | -0.21 | [-0.28, -0.14] | -0.03, -0.81, +0.22 | False |
| 1M | S-R − real-only | 3 | -33.25 | [-33.84, -32.68] | -32.51, -32.18, -35.06 | True |
| 1M | mixed − real-only | 3 | -33.04 | [-33.62, -32.47] | -32.48, -31.37, -35.28 | True |
| 1M | S-R − R-S | 3 | -1.18 | [-1.29, -1.06] | +0.06, -1.19, -2.40 | False |
| 1M | S-R − mixed-realtail | 3 | +11.81 | [+11.63, +12.00] | +11.90, +12.10, +11.42 | True |
| 4M | S-R − mixed | 2 | -1.35 | [-1.39, -1.31] | -1.27, -1.43 | False |
| 4M | S-R − real-only | 2 | -9.91 | [-10.10, -9.71] | -10.34, -9.47 | False |
| 4M | mixed − real-only | 2 | -8.55 | [-8.72, -8.38] | -9.07, -8.04 | False |
| 4M | S-R − R-S | 2 | -3.04 | [-3.10, -2.97] | -2.82, -3.25 | False |
| 4M | S-R − mixed-realtail | 2 | +2.66 | [+2.59, +2.72] | +2.50, +2.81 | False |
