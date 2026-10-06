# Trial G0_1M (timing check only)

- The trial (`plans/trial_G0_1M.json`, 1,048,576 tokens of R_gen, seed 0, defaults lr 3e-3 bs 32 d128 L4 H4 ctx256)
  was a throughput/timing check. Its R_val perplexity was NOT used for any decision.
  All future evaluations used for tuning (LR etc.) use R_dev; R_val is reserved for final reporting.
- Pre-dedup run (`results/trial_G0_1M_s0_pre_dedup/`): R_val ppl 75.9 (nll 4.3299), 27,474 tok/s.
- After the overseer fix (exact-duplicate docs dropped before the split: 20,250 of 529,875 non-empty docs),
  the splits and the BPE tokenizer changed, so the pre-dedup numbers are inconsistent with the current data.
  The trial was re-run fresh on the new data with the same command:
  `OMP_NUM_THREADS=4 .venv/bin/python -I src/train.py --plan plans/trial_G0_1M.json --ckpt ckpt/trial_G0_1M_s0 --seed 0 --log_every 16 --eval R_val=data/R_val.bin`
  -> `results/trial_G0_1M_s0/`: R_val ppl 92.85 (nll 4.5310), final train loss 4.568, 32,265 tok/s.
  Not comparable to 75.9 (different tokenizer, different docs). Also not used for any decision.
- `results/extrapolation.json` was computed from the pre-dedup 27,474 tok/s; the new run is faster (~32k tok/s),
  so that budget estimate is conservative.
