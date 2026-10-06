# dpo-grad-reweight: RESULTS (ARCHIVED, no headline result)

*Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed.*

**Takeaway:** we did not obtain a vanilla-vs-GAW-PO comparison. Plain DPO with our recipe (Qwen2.5-0.5B-Instruct,
RMSprop, batch size 1, CPU) failed the pre-registered mean-loss rule at every tried learning rate (per-step loss is spiky/heavy-tailed), so there was no trustworthy baseline to
build the extension on. Archived after cycle 5 per north-star ("do not keep it just because it is already started").

## What was established (all preliminary, n=1 seed, dev split only; not experimental results)
| Item | Value | Source |
|---|---|---|
| Timed trial, vanilla DPO, beta 0.1, seed 0, lr 5e-6, 100 train/80 eval/200 ARC | 789 s/run (train 372, eval 111, ARC 140, ref 145) | `results/step3_trial.log` |
| Projected full grid (12 runs + extras) at that size | ~135 min; ~83 min if trimmed to 50 train/100 ARC | PLAN Step 4 |
| Dev LR check (50 dev pairs, train==eval, sanity only), mean online DPO loss vs ln2=0.693 | lr 1e-6: 0.80; 5e-6: 1.18; 1e-5: 2.36 | `results/dev_lr_check.json` |

Rule (written into PLAN.md before the run output was read; git cannot timestamp rule and outcome separately, both are in commit 91eef01): an lr qualifies only if mean dev loss is finite and < ln2 AND mean implicit margin > 0. Margins were positive at all three LRs (16.6, 42.7, 38.1) but mean loss failed, so the grid was not run.
Context (overseer recomputation from the raw losses): medians are 0.670/0.682/0.541 (below ln2) while max single-step losses are 5.18/7.45/16.06; dropping the largest spike gives means 0.712/1.056/2.081. So the evidence is "heavy-tailed spiky loss at batch size 1", not proof of divergence. No firm LR trend is claimed.
Likely cause (untested hypothesis): RMSprop with batch size 1. Gradient accumulation / lower lr is a candidate fix (untested). Step 3 trial metrics were also at chance (eval acc 0.4875, margin-sign 0.5).

Deviation from plan: PLAN.md said the extension analysis would still run if no LR qualified; it was not run, because the lab decided under the north-star novelty rule that the extension was not strong enough to continue with an unusable baseline.

## Exact commands
Dev check (commit 91eef01, seed 0, n_dev 50): `cd projects/dpo-grad-reweight && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1500 .venv/bin/python src/train_dpo.py --mode dev_lr --method vanilla --beta 0.1 --seed 0 --n_dev 50` (LR list 1e-6/5e-6/1e-5 is built into the mode).
Timed trial (commit 279496a): same script with `--mode full --method vanilla --beta 0.1 --seed 0 --lr 5e-6 --n_train 100 --n_eval 80 --n_arc 200`.

## Novelty check (cycle 4 scout)
No prior work found that reweights DPO's rejected tokens by gradient alignment with the chosen response. Closest:
Gradient Entanglement (arXiv:2410.13828, diagnostic only), TDPO (2404.11999), SePO (2408.13518), TIS-DPO (2410.04350).
Our "GAW-PO-lite" is a fallback formula and is NOT the paper's Eq. 8 (arXiv:2610.01511), so nothing here validates or refutes that paper.

## Data and licenses
Model Qwen2.5-0.5B-Instruct (apache-2.0, rev 7ae55760); UltraFeedback binarized (MIT, rev 3949bf5f); ARC-Easy (CC-BY-SA-4.0, rev 210d026f); used locally only, not redistributed. Citation IDs (2610.01511 and the four above) came from scout fetches and were not independently re-verified by the Lead; the "no prior work" statement means none found in a limited search.

## Limitations
No 3-seed results, no test-set numbers, no extension analysis was run. Scripts (`src/`) and the pre-registered dev-LR mode are reusable.

## To resume (if ever)
Add gradient accumulation (batch 8-16), re-run `--mode dev_lr` with a new pre-registered rule, then the trimmed grid (see Exact commands above).
