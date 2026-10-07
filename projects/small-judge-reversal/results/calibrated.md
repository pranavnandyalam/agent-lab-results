# Calibrated re-analysis (letter offset removed)

d = (score_cf - score_rf)/2 per criterion; `better` vs `W1`. corr = Pearson(d_better,d_W1); same_sign = share with sign(d_better)==sign(d_W1) (criterion-blind if high; reversal if ~0); acc_cal = share with d_better>0; reverse_cal = share with d_better>0 & d_W1<0. Hep = HumanEvalPack ids (hep-*). CI = bootstrap over pairs, 95%.

| model | subset | n | corr(better,W1) [CI] | same_sign [CI] | acc_cal(better) | reverse_cal [CI] | P(letter A) better/W1 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | all | 300 | 0.95 [0.92,0.97] | 0.92 [0.88,0.95] | 0.61 | 0.06 [0.03,0.08] | 1.00/1.00 |
| Qwen2.5-0.5B-Instruct | non-hep | 83 | 0.95 [0.92,0.97] | 0.93 [0.87,0.98] | 0.43 | 0.04 [0.00,0.08] | 1.00/1.00 |
| Qwen2.5-0.5B-Instruct | hep | 217 | 0.96 [0.94,0.97] | 0.91 [0.88,0.94] | 0.68 | 0.06 [0.03,0.10] | 1.00/1.00 |
| Qwen2.5-0.5B-Instruct | mass>0.9 all 4 passes | 300 | 0.95 [0.92,0.97] | 0.92 [0.88,0.95] | 0.61 | 0.06 [0.03,0.08] | 1.00/1.00 |
| Qwen2.5-0.5B-Instruct | letter-null (all four passes same letter) | 300 | expected 1.00 | observed 1.00 | | | |
| Qwen2.5-1.5B-Instruct | all | 300 | 0.63 [0.51,0.73] | 0.71 [0.66,0.76] | 0.50 | 0.22 [0.18,0.27] | 0.20/0.01 |
| Qwen2.5-1.5B-Instruct | non-hep | 83 | 0.72 [0.61,0.82] | 0.70 [0.59,0.80] | 0.53 | 0.18 [0.11,0.27] | 0.20/0.01 |
| Qwen2.5-1.5B-Instruct | hep | 217 | 0.58 [0.38,0.72] | 0.71 [0.65,0.77] | 0.48 | 0.24 [0.18,0.30] | 0.20/0.01 |
| Qwen2.5-1.5B-Instruct | mass>0.9 all 4 passes | 288 | 0.60 [0.47,0.71] | 0.70 [0.65,0.75] | 0.50 | 0.23 [0.18,0.27] | 0.20/0.01 |
| Qwen2.5-1.5B-Instruct | letter-null (all four passes same letter) | 300 | expected 0.63 | observed 0.71 | | | |
| Qwen3-0.6B | all | 300 | 0.90 [0.87,0.93] | 0.87 [0.82,0.90] | 0.57 | 0.08 [0.05,0.11] | 0.94/0.94 |
| Qwen3-0.6B | non-hep | 83 | 0.91 [0.88,0.94] | 0.88 [0.81,0.94] | 0.48 | 0.04 [0.00,0.08] | 0.94/0.94 |
| Qwen3-0.6B | hep | 217 | 0.91 [0.84,0.95] | 0.86 [0.82,0.91] | 0.61 | 0.09 [0.06,0.13] | 0.94/0.94 |
| Qwen3-0.6B | mass>0.9 all 4 passes | 286 | 0.91 [0.88,0.94] | 0.88 [0.84,0.91] | 0.57 | 0.06 [0.03,0.09] | 0.94/0.94 |
| Qwen3-0.6B | letter-null (all four passes same letter) | 300 | expected 0.79 | observed 0.89 | | | |
| Qwen3-1.7B | all | 300 | 0.63 [0.55,0.71] | 0.76 [0.71,0.81] | 0.81 | 0.18 [0.14,0.23] | 0.34/0.12 |
| Qwen3-1.7B | non-hep | 83 | 0.57 [0.38,0.72] | 0.75 [0.65,0.84] | 0.61 | 0.12 [0.06,0.19] | 0.34/0.12 |
| Qwen3-1.7B | hep | 217 | 0.69 [0.59,0.77] | 0.77 [0.71,0.82] | 0.88 | 0.20 [0.15,0.26] | 0.34/0.12 |
| Qwen3-1.7B | mass>0.9 all 4 passes | 140 | 0.63 [0.50,0.73] | 0.75 [0.68,0.82] | 0.72 | 0.18 [0.12,0.24] | 0.34/0.12 |
| Qwen3-1.7B | letter-null (all four passes same letter) | 300 | expected 0.34 | observed 0.43 | | | |
| Qwen3-4B | all | 100 | 0.10 [-0.09,0.30] | 0.59 [0.49,0.68] | 0.86 | 0.31 [0.23,0.40] | 0.45/0.30 |
| Qwen3-4B | non-hep | 32 | -0.19 [-0.57,0.20] | 0.38 [0.22,0.56] | 0.66 | 0.38 [0.22,0.53] | 0.45/0.30 |
| Qwen3-4B | hep | 68 | 0.30 [0.10,0.49] | 0.69 [0.57,0.79] | 0.96 | 0.28 [0.18,0.40] | 0.45/0.30 |
| Qwen3-4B | mass>0.9 all 4 passes | 97 | 0.09 [-0.11,0.29] | 0.59 [0.49,0.68] | 0.87 | 0.31 [0.23,0.40] | 0.45/0.30 |
| Qwen3-4B | letter-null (all four passes same letter) | 100 | expected 0.17 | observed 0.06 | | | |
