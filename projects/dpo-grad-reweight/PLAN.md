# PLAN: dpo-grad-reweight (rev 2 — revised after overseer REJECT)

## Trend this responds to
GAW-PO ("GAW-PO: Preference Optimization with Gradient-Aligned Token Weights", Dutulescu et al.,
arXiv:2610.01511, submitted Oct 1 2026): confirmed via direct fetch of arxiv.org/abs/2610.01511 (200 OK).
Abstract states the reweighting applies **"for each rejected token"** — i.e. only rejected-response
tokens are reweighted, by how much their gradient aligns with the preferred (chosen-response) update
direction: aligned rejected tokens get a weaker penalty, conflicting ones keep the full penalty. Claims
+0.97 over vanilla DPO across 11 benchmarks, more stable at low beta. No public code repo found.
Directly adjacent to Pranav's own preference-optimization work (MAPO, arXiv:2410.19499).

## Rev 2 changelog (fixes from overseer REJECT + ethics APPROVE_WITH_CONDITIONS)
1. Fallback formula now matches the abstract's description (rejected-tokens-only, direction-aware).
2. Gradient-alignment stand-in is now closed-form and cheap (logit-space cross-entropy gradient), not a
   per-token full-parameter backward pass.
3. Added metrics that can actually detect the paper's claimed "low-beta degradation" (length-normalized
   log-probs, implicit-reward margin, KL-from-reference, a small log-likelihood benchmark) instead of
   raw summed log-probs, which confound accuracy with response length and can't distinguish degradation
   from improvement.
4. Added a 50-pair dev split used only to confirm the LR actually moves the model under vanilla DPO
   before the eval split is touched.
5. Capped sequence length, precompute frozen reference log-probs once, and shrank train/eval set sizes
   (not seed count) to fit the CPU time budget, after measuring real per-step wall-clock first.
6. Added a minimum-detectable-effect note and switched to paired per-item comparison.
7. Ethics conditions baked in: AI-authorship + non-peer-reviewed disclosure, "best-effort conceptual
   reconstruction (GAW-PO-lite), not a verified replication" language throughout (never shortened to
   "replication" of GAW-PO), license + revision-hash verification before training (abort/swap dataset if
   non-permissive/unclear), and plain-language reporting of null/negative results in the summary itself.

## Step 0 (required before any training code): confirm the exact method
- Try to fetch the full paper text (not just the abstract) via arxiv.org/pdf/2610.01511 or the abs page's
  full HTML, within the allowed network list (arxiv.org / export.arxiv.org only — no ar5iv or other
  mirrors). If a WebFetch of the PDF yields the actual reweighting formula, use it and cite the exact
  equation/section in README.md.
- If the full method is still unavailable or ambiguous, use this explicit, pre-registered fallback
  ("GAW-PO-lite", clearly labeled as our reconstruction in every output):
  - For the **rejected** response only, at each token t with target id y_t and logits z_t (vocab-dim):
    the standard cross-entropy gradient w.r.t. logits is g_t = softmax(z_t) - one_hot(y_t) (closed form,
    no extra backward pass).
  - For the **chosen** response, compute the mean gradient-direction vector g_chosen = mean_t'(
    softmax(z_t') - one_hot(y_t')) over its tokens (same closed form).
  - Alignment score a_t = cosine_similarity(g_t, g_chosen), computed with NO gradient flowing through it
    (detached/no_grad — this is a scalar weight, not a differentiable loss term). Pre-registered reading:
    g_t is the direction that *increases* CE loss on (decreases probability of) the rejected token; -g_t
    is the direction that *decreases* its loss (raises its probability). Pushing the rejected token's
    probability DOWN (the DPO objective) moves along +g_t. Raising the chosen response's probability
    moves along -g_chosen. When a_t = cos(g_t, g_chosen) is HIGH, +g_t and -g_chosen point in roughly
    opposite directions, so pushing this rejected token down actively conflicts with raising the chosen
    response — these are "chosen-like" tokens (e.g. shared prefixes/generic tokens) where a full penalty
    would collaterally fight the chosen objective. When a_t is LOW/negative, pushing the token down does
    not conflict with (or even helps) raising the chosen response — a genuinely distinguishing token.
  - Per-token weight on the rejected-side DPO loss term: w_t = 1 - lambda * clip(a_t, 0, 1), lambda=0.5
    fixed (not tuned on eval), detached from the autograd graph. High-a_t ("chosen-like"/conflicting)
    tokens get down-weighted toward 0.5x penalty; low/negative-a_t (genuinely distinguishing) tokens keep
    full weight 1.0. Chosen-side loss terms are never reweighted (matches "for each rejected token" in
    the abstract).
  - Exact loss placement: inside the DPO sigmoid, the rejected-side term becomes
    beta * sum_t [ w_t * (logpi(y_t) - logpi_ref(y_t)) ] in place of the usual
    beta * sum_t (logpi(y_t) - logpi_ref(y_t)) — i.e. w_t multiplies each rejected token's per-token
    log-ratio (both the policy and reference log-prob terms together, since it's their difference that's
    weighted), summed before going into the sigmoid. The chosen-side sum is unweighted.
  - This is a closed-form, single-forward-pass stand-in for "gradient alignment" — document as an
    approximation of the paper's (unknown to us) exact method, not a reproduction of it. Limitation to
    disclose: comparing logit-gradient vectors across positions is mostly a token-identity/softmax-overlap
    signal, not a deep measure of "gradient alignment" in the full parameter space.

## Method
- **Model**: Qwen2.5-0.5B-Instruct. Note (overseer-flagged): this checkpoint is already
  instruction/preference-tuned, which may compress differences between vanilla DPO and GAW-PO-lite —
  disclose this explicitly as a limitation and watch for suspiciously-small effects across the board as a
  possible symptom. Apache 2.0 per card — confirm exact license + revision hash at fetch time, record in
  RESULTS.md. safetensors only, no trust_remote_code.
- **Data**: a public DPO-style preference dataset (candidate: HuggingFaceH4/ultrafeedback_binarized or
  similar). Confirm license + revision at fetch time; abort/swap if non-permissive or unclear (ethics
  condition). Splits, all disjoint, fixed across all runs:
  - 50 dev pairs: used ONLY to sanity-check that the chosen LR moves vanilla DPO (beta=0.1) off the base
    model before touching eval. Try at most 3 LR values from the DPO literature (e.g. 1e-6, 5e-6, 1e-5);
    picking among at most 3 pre-specified values is a sanity check, not a search — do not go beyond 3 or
    this becomes hidden tuning. If none shows measurable movement, report that as a finding rather than
    trying a 4th value.
  - Train/eval sizes start small and are measured, not assumed (see Compute budget below): default
    starting point is 100 train pairs, 80 held-out eval pairs; cut further (never below ~40 eval pairs)
    if the timed trial run demands it. Standard error at p=0.5 is ~5pp at n=100 and ~7.9pp at n=40 —
    state whichever applies once the final size is fixed.
  - Max sequence length: 256 tokens, truncate longer examples (document how many are affected).
- **Conditions**: 2 methods (vanilla DPO, GAW-PO-lite) x 2 beta values (0.1, 0.01) x 3 seeds = 12 runs,
  1 epoch over the frozen train set. Seeds are NEVER cut to fit budget — cut train/eval size instead.
  Reference-model log-probs are precomputed once per split and reused across all 12 runs (reference is
  frozen, so this is exact, not an approximation).
- **Baselines**: vanilla DPO at both beta values is the primary baseline; base (untrained) model is the
  sanity floor.
- **Metrics** (all computed on the untouched eval split, paired per-item):
  1. Length-normalized log-prob preference accuracy: mean per-token logp(chosen) > mean per-token
     logp(rejected). (Raw summed log-probs are reported too but only as a secondary, clearly-labeled
     number — not the headline metric — since they confound with length.)
  2. Implicit reward margin (standard DPO quantity): beta*(logpi(chosen)-logpi_ref(chosen)) -
     beta*(logpi(rejected)-logpi_ref(rejected)); report accuracy of sign and the magnitude.
  3. Log-prob drift of chosen responses: mean per-token logp(chosen) under trained policy minus under
     reference model — a negative drift (policy likes the chosen answer LESS after training) is the
     clearest direct signal of the "degradation" the paper describes.
  4. KL(policy || reference) on held-out eval prompts (next-token distribution, averaged over the first
     token of the eval continuations — cheap, forward-only).
  5. A small external sanity benchmark, scored by log-likelihood only (no generation): ~200 ARC-Easy
     items (public, permissive license — confirm at fetch time), multiple-choice-by-log-likelihood
     accuracy. This is the one metric that isn't circularly defined by the training objective itself.
- **Reporting**: mean +/- std over 3 seeds per (method, beta) cell for every metric above, plus the base
  model floor. State the minimum detectable effect given eval-set size (~5pp SE at n=100, ~7.9pp at n=40)
  up front, and use paired (same eval item, both methods) comparisons, not independent
  two-sample tests. Note explicitly that seeds here vary only data order/batching/init, not independent
  resamples of the preference data, so cross-seed spread likely understates true variance — flag this as
  a limitation rather than implying it's a full uncertainty estimate.

## Compute budget and stop criteria (revised — overseer found the original estimate unrealistic)
- Hard cap: 2 hours CPU wall-clock total for the whole grid. Threads capped at 4 (OMP_NUM_THREADS=4,
  torch.set_num_threads(4)).
- Mandatory timed trial BEFORE the full grid: run ONE config (vanilla DPO, beta=0.1, seed=0) on the
  starting 100-pair/256-token setup end-to-end, including training AND the full eval-time cost (all 5
  metrics on the 80 held-out pairs, plus the ~200-item x 4-choice ARC-Easy log-likelihood pass) — not
  training time alone, since eval+ARC is repeated for every one of the 13 models (12 runs + base).
  Measure actual wall-clock for this one full train+eval cycle.
  - If projected total (single-model train+eval time x 13) exceeds ~90 minutes, cut train pairs (e.g.
    100 -> 50 -> 25) and/or the ARC subset size, and re-time, NOT seeds and NOT the beta/method grid.
  - If even a heavily-cut run (e.g. 25 pairs) can't fit, stop and report this as a negative/infeasibility
    finding in RESULTS.md ("CPU-only full fine-tuning of a 0.5B model is not practical at N>=3 seeds x 4
    conditions within our compute budget") rather than silently degrading the design further.
- If any run NaNs or the loss diverges: stop that cell, record it as a failure (not silently retried with
  different hyperparameters), report it as a finding.
- No background processes may outlive the cycle; checkpoint to ~/scratch if a run must pause mid-grid,
  record the exact resume command in STATE.md.

## Reproduction steps
To be finalized in README.md once code exists: exact commands, seeds, pinned requirements.txt, model
revision hash, dataset revision hash, git commit.

## Risks / limitations to disclose up front (expanded per overseer + ethics)
- This is explicitly NOT a verified replication of GAW-PO's exact method — see Step 0. Call it
  "GAW-PO-lite, our best-effort conceptual reconstruction" everywhere, never shortened to "replication."
- No held-out generation-quality judge (no GPU/API budget) — logprob-based metrics are a proxy, weaker
  evidence than human/judge-rated win rate.
- No related-work check yet of existing token-level DPO variants (e.g. TDPO, SePO) — add a quick
  novelty-check scout pass before finalizing the write-up, to situate (or retract) any novelty claim.
- Cross-seed variance likely understates true uncertainty (see Reporting above).
- An already-instruction/preference-tuned base checkpoint may compress method differences — note which
  checkpoint was actually used and flag this if it looks like it's suppressing all effects.

## Review routing — status: APPROVED
- ethics-reviewer: APPROVE_WITH_CONDITIONS (rev 1) — conditions baked into this plan (AI-authorship +
  non-peer-reviewed disclosure, "GAW-PO-lite... not a verified replication" language, license/revision
  verification before training, plain-language reporting of null results).
- overseer: REJECT on rev 1 (4 blocking issues, all fixed in rev 2) -> APPROVE on rev 2, conditioned on 5
  required fixes (weighting rationale/direction, detach w_t from autograd, exact loss placement, timed
  trial must include eval+ARC time not just training time, cap dev-split LR search at 3 values) — all 5
  applied directly to this file above.
- Builder work can now start: Step 0 paper-text fetch, fetch_data.sh + requirements.txt, baseline DPO
  implementation, then the mandatory timed trial run (train+eval+ARC, one config) before committing to
  the full 12-run grid.
