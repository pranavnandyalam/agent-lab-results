# Referee round 2 analyses (from existing logits; no new inference)

d_q = (s_q^cf - s_q^rf)/2 (A-minus-B logit, letter offset cancels; >0 = favours chosen C). Bootstrap over pairs, 2000 reps, seed 0 (re-seeded per block). Script: `src/referee2.py`.

## (a) Length positive control (resample 0, 'Which response is longer?')

dlen_char = len(C)-len(R) in characters. r = Pearson, rho = Spearman. acc_len_cal = share of non-tied pairs with sign(d_length)==sign(dlen_char).

| judge | n (non-tied) | r(d_length, dlen_char) [CI] | rho(d_length, dlen_char) | acc_len_cal [CI] | r(d_length, d_better) [CI] (n=100) | r(d_better, dlen_char) | median abs d_length | median abs d_better (r0) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 72 | 0.12 [-0.20, 0.40] | 0.17 | 0.65 [0.54, 0.75] | 0.96 [0.91, 0.98] | 0.06 | 0.045 | 0.051 |
| Qwen2.5-1.5B-Instruct | 72 | 0.68 [0.46, 0.81] | 0.57 | 0.61 [0.50, 0.71] | 0.89 [0.81, 0.94] | 0.50 | 0.358 | 0.426 |
| Qwen3-0.6B | 72 | 0.60 [0.33, 0.76] | 0.47 | 0.64 [0.53, 0.75] | 0.91 [0.83, 0.94] | 0.50 | 0.301 | 0.214 |
| Qwen3-1.7B | 72 | 0.46 [0.20, 0.64] | 0.30 | 0.58 [0.47, 0.69] | 0.74 [0.61, 0.84] | 0.34 | 4.048 | 4.483 |

## (b) Calibrated size trend within family (paired bootstrap over the same 300 pairs)

diff = larger minus smaller judge. corr = r(d_better, d_W1); same_sign = share sign(d_better)==sign(d_W1).

| family | n | metric | small | large | diff (large-small) [95% CI] |
|---|---|---|---|---|---|
| Qwen2.5 0.5B->1.5B | 300 | corr | 0.95 | 0.63 | -0.32 [-0.43, -0.22] |
| Qwen2.5 0.5B->1.5B | 300 | same_sign | 0.92 | 0.71 | -0.21 [-0.27, -0.15] |
| Qwen2.5 0.5B->1.5B | 300 | acc_cal | 0.61 | 0.50 | -0.12 [-0.19, -0.04] |
| Qwen3 0.6B->1.7B | 300 | corr | 0.90 | 0.63 | -0.27 [-0.37, -0.18] |
| Qwen3 0.6B->1.7B | 300 | same_sign | 0.87 | 0.76 | -0.10 [-0.16, -0.05] |
| Qwen3 0.6B->1.7B | 300 | acc_cal | 0.57 | 0.81 | +0.23 [0.17, 0.29] |

## (c) Per judge: OLS slope, magnitude ratio, order-averaged probabilities (all 300 pairs)

slope = OLS of d_W1 on d_better (with intercept). ratio = median over pairs of |d_W1|/|d_better| (pairs with d_better != 0). P_C(q) = order-averaged P(pick C) = [sigmoid(s_cf) + sigmoid(-s_rf)]/2 (A-vs-B two-way softmax). Sum S = P_C(better)+P_C(W1): criterion-following => S~1 with P_C(better) far from 0.5; criterion-blind => P_C(better)~P_C(W1) so S~2*P_C(better); a letter-only judge gives P_C=0.5 for both, S=1. So S~1 alone does not identify criterion-blindness; Delta = P_C(better)-P_C(W1) (blind ~0, following >0) is reported alongside.

| judge | n | slope [CI] | median abs ratio | mean P_C(better) | mean P_C(W1) | mean S [CI] | mean abs(S-1) | mean Delta [CI] | mean abs(P_C(better)-0.5) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 300 | 0.82 [0.77, 0.87] | 0.90 | 0.501 | 0.501 | 1.002 [1.00, 1.01] | 0.015 | -0.000 [-0.00, 0.00] | 0.007 |
| Qwen2.5-1.5B-Instruct | 300 | 0.39 [0.31, 0.48] | 0.82 | 0.505 | 0.492 | 0.997 [0.98, 1.01] | 0.104 | +0.013 [0.00, 0.02] | 0.083 |
| Qwen3-0.6B | 300 | 0.67 [0.61, 0.73] | 0.89 | 0.511 | 0.504 | 1.014 [1.00, 1.03] | 0.090 | +0.007 [0.00, 0.01] | 0.053 |
| Qwen3-1.7B | 300 | 0.26 [0.20, 0.31] | 0.35 | 0.678 | 0.543 | 1.220 [1.18, 1.26] | 0.318 | +0.135 [0.11, 0.16] | 0.249 |

## (d) HEP pairs: token edit distance between C and R, stratified

Whitespace tokens, token-level Levenshtein. n_hep = 217; edit distance quantiles (0/10/25/50/75/90/100%): 1/1/1/1/2/3/24; pairs with distance 0: 0.

| judge | stratum | n | r(d_better,d_W1) [CI] | same_sign | acc_cal(better) [CI] | median abs d_better |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | ed<5 | 204 | 0.96 [0.93, 0.97] | 0.91 | 0.69 [0.63, 0.75] | 0.033 |
| Qwen2.5-0.5B-Instruct | 5<=ed<20 | 12 | 0.97 [0.92, 0.99] | 0.92 | 0.50 [0.25, 0.75] | 0.126 |
| Qwen2.5-0.5B-Instruct | ed>=20 | 1 | nan [nan, nan] | 1.00 | 1.00 [nan, nan] | 0.028 |
| Qwen2.5-0.5B-Instruct | all hep | 217 | 0.96 [0.94, 0.97] | 0.91 | 0.68 [0.62, 0.75] | 0.035 |
| Qwen2.5-0.5B-Instruct | hep excl. ed<5 | 13 | 0.97 [0.92, 1.00] | 0.92 | 0.54 [0.31, 0.77] | 0.125 |
| Qwen2.5-0.5B-Instruct | fine: ed=1 | 131 | 0.96 [0.94, 0.98] | 0.92 | 0.73 [0.66, 0.81] | 0.029 |
| Qwen2.5-0.5B-Instruct | fine: ed=2 | 50 | 0.95 [0.89, 0.98] | 0.86 | 0.60 [0.46, 0.74] | 0.031 |
| Qwen2.5-0.5B-Instruct | fine: ed>=3 | 36 | 0.97 [0.93, 0.98] | 0.94 | 0.61 [0.44, 0.78] | 0.058 |
| Qwen2.5-1.5B-Instruct | ed<5 | 204 | 0.58 [0.39, 0.74] | 0.71 | 0.47 [0.40, 0.53] | 0.349 |
| Qwen2.5-1.5B-Instruct | 5<=ed<20 | 12 | 0.23 [-0.30, 0.79] | 0.67 | 0.75 [0.50, 1.00] | 0.184 |
| Qwen2.5-1.5B-Instruct | ed>=20 | 1 | nan [nan, nan] | 1.00 | 1.00 [nan, nan] | 0.181 |
| Qwen2.5-1.5B-Instruct | all hep | 217 | 0.58 [0.38, 0.72] | 0.71 | 0.48 [0.42, 0.55] | 0.341 |
| Qwen2.5-1.5B-Instruct | hep excl. ed<5 | 13 | 0.18 [-0.32, 0.71] | 0.69 | 0.77 [0.54, 1.00] | 0.181 |
| Qwen2.5-1.5B-Instruct | fine: ed=1 | 131 | 0.70 [0.56, 0.79] | 0.73 | 0.43 [0.34, 0.51] | 0.362 |
| Qwen2.5-1.5B-Instruct | fine: ed=2 | 50 | 0.26 [-0.19, 0.77] | 0.66 | 0.50 [0.36, 0.64] | 0.299 |
| Qwen2.5-1.5B-Instruct | fine: ed>=3 | 36 | 0.63 [0.26, 0.84] | 0.69 | 0.67 [0.50, 0.83] | 0.309 |
| Qwen3-0.6B | ed<5 | 204 | 0.88 [0.83, 0.92] | 0.86 | 0.61 [0.54, 0.68] | 0.169 |
| Qwen3-0.6B | 5<=ed<20 | 12 | 0.99 [0.88, 1.00] | 0.83 | 0.58 [0.33, 0.83] | 0.183 |
| Qwen3-0.6B | ed>=20 | 1 | nan [nan, nan] | 1.00 | 0.00 [nan, nan] | 0.003 |
| Qwen3-0.6B | all hep | 217 | 0.91 [0.84, 0.95] | 0.86 | 0.61 [0.54, 0.67] | 0.169 |
| Qwen3-0.6B | hep excl. ed<5 | 13 | 0.99 [0.79, 1.00] | 0.85 | 0.54 [0.23, 0.77] | 0.174 |
| Qwen3-0.6B | fine: ed=1 | 131 | 0.88 [0.83, 0.93] | 0.85 | 0.60 [0.52, 0.69] | 0.145 |
| Qwen3-0.6B | fine: ed=2 | 50 | 0.90 [0.80, 0.95] | 0.88 | 0.68 [0.54, 0.80] | 0.218 |
| Qwen3-0.6B | fine: ed>=3 | 36 | 0.95 [0.66, 0.99] | 0.89 | 0.53 [0.36, 0.67] | 0.172 |
| Qwen3-1.7B | ed<5 | 204 | 0.71 [0.62, 0.78] | 0.78 | 0.89 [0.85, 0.93] | 5.313 |
| Qwen3-1.7B | 5<=ed<20 | 12 | -0.04 [-0.83, 0.70] | 0.50 | 0.75 [0.50, 1.00] | 2.019 |
| Qwen3-1.7B | ed>=20 | 1 | nan [nan, nan] | 1.00 | 0.00 [nan, nan] | 10.936 |
| Qwen3-1.7B | all hep | 217 | 0.69 [0.59, 0.77] | 0.77 | 0.88 [0.84, 0.92] | 5.243 |
| Qwen3-1.7B | hep excl. ed<5 | 13 | 0.40 [-0.69, 0.83] | 0.54 | 0.69 [0.46, 0.92] | 2.097 |
| Qwen3-1.7B | fine: ed=1 | 131 | 0.70 [0.57, 0.80] | 0.78 | 0.89 [0.83, 0.94] | 5.090 |
| Qwen3-1.7B | fine: ed=2 | 50 | 0.68 [0.50, 0.82] | 0.84 | 0.96 [0.90, 1.00] | 5.934 |
| Qwen3-1.7B | fine: ed>=3 | 36 | 0.67 [0.35, 0.85] | 0.64 | 0.75 [0.61, 0.89] | 3.706 |

Reference: non-HEP pairs (n=83) median token edit distance 45.
