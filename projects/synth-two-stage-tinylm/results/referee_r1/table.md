| arm | description | seeds done | R_val ppl per seed | R_val mean ± std | R_dev mean |
|---|---|---|---|---|---|
| S-R | S->R, global schedule (lr 3e-3) | 0,1,2 | s0: 15.895 / s1: 16.028 / s2: 16.437 | 16.12 ± 0.28 | 16.36 |
| S-R-perphase | S->R, per-phase warmup+cosine restart (lr 3e-3) | 0,1,2 | s0: 15.839 / s1: 15.940 / s2: 16.102 | 15.96 ± 0.13 | 16.20 |
| mixed | mixed, global (lr 3e-3) | 0,1,2 | s0: 17.163 / s1: 17.462 / s2: 17.548 | 17.39 ± 0.20 | 17.60 |
| real-only-2ep | real-only-2ep, global (lr 3e-3) | 0,1,2 | s0: 15.064 / s1: 15.132 / s2: 15.341 | 15.18 ± 0.14 | 15.40 |
| mixed-lr1e-3 | mixed, global, lr 1e-3 | 0 | s0: 22.884 | 22.88 ± n/a | 23.13 |
| S-R-lr1e-3 | S->R, global, lr 1e-3 | 0 | s0: 22.195 | 22.19 ± n/a | 22.48 |
| real-only-2ep-lr1e-3 | real-only-2ep, global, lr 1e-3 | 0 | s0: 21.992 | 21.99 ± n/a | 22.28 |
