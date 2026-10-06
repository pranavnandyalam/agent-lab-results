team: main
# PLAN: synth-two-stage-tinylm (rev 1, 2026-10-06)

## What's new here
Li & Zou (arXiv 2609.09572) prove, for SGD in high-dimensional linear regression, that mixing synthetic and real data gives an error floor while two-stage training (synthetic first, then real) avoids it. Nobody appears to have tested this on a language model. Closest empirical work: Strong Model Collapse (2410.04840, mixed only), Gerstgrasser et al. (2404.01413, replace vs accumulate), TinyStories recursive collapse (2412.14872, pure synthetic), Silent Collapse (2605.14588, mixing-fraction schedules, not two-stage). Difference: we run an ordering experiment (synthetic->real vs real->synthetic vs mixed vs real-only) with matched real tokens on a tiny GPT. Novelty check: scout, 2026-10-06, web-only (medium confidence).

## Hypotheses
- H1: At equal total tokens, two-stage (S->R) reaches lower held-out real perplexity than mixed (same synthetic fraction), and the gap grows with synthetic generation depth.
- H2: Two-stage S->R is not better than real-only at equal real tokens when synthetic data comes from a model trained on the same real data (null-expected; the informative control).
- Reverse order (R->S) is a control: expected worse than S->R.

## Method
- Data: TinyStories (HF roajon/TinyStories-style official `roneneldan/TinyStories`, license CDLA-Sharing-1.0, pin revision), ~20M-token real subset split into disjoint R_train (train), R_gen (to train generation-0 generator), R_val (held-out).
- Model: GPT ~8M params (4 layers, d=256), BPE vocab 4096 trained on R_train (tokenizers lib). 4 threads.
- Generator G0 trained on R_gen; synthetic corpus S1 sampled from G0 (T=1.0). Generation 2: G1 trained on S1, samples S2 (depth 2). Depth in {1,2}; optional 3.
- Arms (same total token budget N, same real tokens for all arms except where stated): real-only; mixed (synthetic fraction 0.5 interleaved); two-stage S->R; reverse R->S. Same LR schedule per stage, tuned on a dev split only.
- Metric: perplexity on R_val (never tuned on). Also distinct-n of samples.
- Seeds: 3 per arm. Report mean±std, n.
- Compute budget: est. 8M model ~ 10 min per 20M-token run on 4 threads; ~30 runs total ~5 h over several cycles, resumable chunks, checkpoint to ~/scratch. Timed trial first; cut depth/token budget if projection >6 h.
- Stop criteria: 3 cycles without new result -> write up and archive.

## Limitations to report
Single dataset, tiny model, no real-token matching confound beyond the arms above; linear-regression theory differs in setup.
