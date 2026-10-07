# Generation-based validation of the logit readout (referee fix 1)

Greedy decoding, max_new_tokens=16, identical chat prompt (Qwen3: enable_thinking=False), resample 0, criteria `better` and `W1` (worse), both orders (4 judgments/pair). Verdict = first standalone `A`/`B` in the decoded text (regex `(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])`). Logit readout = argmax over tokens `A` vs `B` from raw_<model>_r0.json. Taxonomy/metrics via src/sjr/metrics.py on pairs where all 4 generated verdicts parsed; logit rows use the same pairs. acc = chosen preferred under `better` in both orders; reversal = acc AND rejected named `worse` in both orders (correct_reversal share); position-locked = same letter in all 4 judgments. CI: bootstrap over pairs (2000, seed 0).

## Qwen3-1.7B (rev 70d244cc86cc)

- judgments generated: 400 (100 complete pairs); parsed: 400/400 = 1.00
- agreement generated vs logit argmax: 361/400 = 0.90 (better: 0.83, n=200; W1: 0.97, n=200)
- per-pair taxonomy category agreement (gen vs logit): 77/100 = 0.77
- P(letter A) better/W1: generated 0.49/0.11; logit 0.33/0.10 (n=100 pairs x 2 orders)
- most frequent generations (first 30 chars, count): [('B', 129), ('A', 51), ('B\n\n**Reason:**  \nResponse B is', 34), ('Response A is better. \n\n**Reas', 26), ('Response A is better.\n\n**Reaso', 19), ('B\n\n**Reasoning:**  \nResponse B', 13), ('Response B is worse. \n\n**Reaso', 12), ('B\n\n**Reason:** Response B is w', 11)]
- unparsable examples: []

| readout | n pairs | acc(better, both orders) [CI] | correct_reversal | position_locked | criterion_blind | reversed_consistent | other | cond_flip | pos_locked share of non-CR |
|---|---|---|---|---|---|---|---|---|---|
| generated | 100 | 0.51 [0.41,0.60] | 0.04 | 0.22 | 0.10 | 0.00 | 0.64 | 0.08 (n=51) | 0.23 (n=96) |
| logit (same pairs) | 100 | 0.44 [0.34,0.53] | 0.01 | 0.43 | 0.10 | 0.00 | 0.46 | 0.02 (n=44) | 0.43 (n=99) |
| logit (all complete pairs) | 100 | 0.44 [0.34,0.53] | 0.01 | 0.43 | 0.10 | 0.00 | 0.46 | 0.02 (n=44) | 0.43 (n=99) |

## Qwen2.5-1.5B-Instruct

not run
