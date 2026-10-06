"""Shared utilities for dpo-grad-reweight: model/data loading, tokenization,
DPO + GAW-PO-lite loss, and the 5 PLAN.md metrics. CPU-only, threads capped at 4.

GAW-PO-lite is our pre-registered, explicitly-labeled "best-effort conceptual
reconstruction" fallback formula from PLAN.md Step 0 (NOT the real GAW-PO paper's
Eq. 8 — see README.md / final report for the Lead-relayed scout finding that the
real equation differs: dot-product + sigmoid link + a max over two alignment terms,
vs. our cosine-similarity + clipped-linear link, single alignment term).
"""
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

MAX_SEQ_LEN = 256
LAMBDA_GAWPO = 0.5  # fixed, not tuned on eval (PLAN.md)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_tokenizer(model_id: str, revision: str, hf_home: str):
    """Load tokenizer + model. safetensors only, no trust_remote_code, CPU float32."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = hf_home
    tok = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model.eval()
    return model, tok


def build_tiny_random_model(vocab_size=400, seed=0):
    """Build a tiny randomly-initialized Qwen2-architecture model, no network,
    used ONLY for the offline smoke test (--smoke_test), never for real results."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    set_all_seeds(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=MAX_SEQ_LEN,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def tokenize_pair(tokenizer, prompt_messages, response_text, max_len=MAX_SEQ_LEN):
    """Tokenize prompt (chat-templated) + response. Returns dict with input_ids,
    attention_mask, and a response_mask that is 1 only over response tokens
    (prompt tokens + padding are 0). Truncates to max_len (prompt truncated first
    from the left if needed, so the response end is preserved)."""
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    truncated = False
    if len(prompt_ids) + len(response_ids) > max_len:
        truncated = True
        keep_prompt = max(max_len - len(response_ids), 1)
        prompt_ids = prompt_ids[-keep_prompt:]
        response_ids = response_ids[: max_len - len(prompt_ids)]
    input_ids = prompt_ids + response_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "response_mask": response_mask,
        "truncated": truncated,
        "n_prompt": len(prompt_ids),
        "n_response": len(response_ids),
    }


def forward_logprobs(model, input_ids, response_mask, requires_grad: bool):
    """Run model forward, return per-token log-prob of the actual next token at
    each response position (list aligned to response tokens), plus the raw
    logits at those positions (for GAW-PO-lite gradient-direction computation).
    input_ids: 1D LongTensor [T]. response_mask: 1D LongTensor [T] (1 on response
    tokens, aligned to input_ids position of the TARGET token, i.e. mask[i]==1
    means input_ids[i] is a response token whose prob we score using logits at
    position i-1)."""
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        out = model(input_ids.unsqueeze(0))
        logits = out.logits[0]  # [T, V]
    # logits[i-1] predicts input_ids[i]
    shift_logits = logits[:-1, :]  # [T-1, V]
    shift_targets = input_ids[1:]  # [T-1]
    shift_mask = response_mask[1:]  # [T-1], aligned with targets
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs_all.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
    resp_positions = shift_mask.bool()
    resp_logp = token_logp[resp_positions]  # [T_resp], has grad if requires_grad
    resp_logits = shift_logits[resp_positions]  # [T_resp, V]
    resp_targets = shift_targets[resp_positions]  # [T_resp]
    return resp_logp, resp_logits, resp_targets


def gawpolite_weights(rej_logits, rej_targets, chosen_logits, chosen_targets):
    """PLAN.md pre-registered GAW-PO-lite fallback. All detached (no_grad): a
    scalar per-token weight on the rejected-side loss term only.
    g_t = softmax(z_t) - onehot(y_t) for each rejected token t.
    g_chosen = mean_t' (softmax(z_t') - onehot(y_t')) over chosen tokens.
    a_t = cos(g_t, g_chosen); w_t = 1 - lambda * clip(a_t, 0, 1).
    """
    with torch.no_grad():
        g_rej = F.softmax(rej_logits, dim=-1) - F.one_hot(
            rej_targets, num_classes=rej_logits.shape[-1]
        ).float()  # [T_rej, V]
        g_chosen_each = F.softmax(chosen_logits, dim=-1) - F.one_hot(
            chosen_targets, num_classes=chosen_logits.shape[-1]
        ).float()  # [T_chosen, V]
        g_chosen = g_chosen_each.mean(dim=0)  # [V]
        a_t = F.cosine_similarity(g_rej, g_chosen.unsqueeze(0), dim=-1)  # [T_rej]
        w_t = 1.0 - LAMBDA_GAWPO * a_t.clamp(0.0, 1.0)
    return w_t  # detached, [T_rej]


def dpo_loss_one_pair(
    policy_model,
    ref_chosen_logp_sum,
    ref_rejected_logp,
    chosen_input_ids,
    chosen_response_mask,
    rejected_input_ids,
    rejected_response_mask,
    beta: float,
    method: str,
):
    """Compute the DPO (or GAW-PO-lite) loss for a single preference pair.
    ref_chosen_logp_sum: precomputed frozen-reference summed log-prob for the
    chosen response (python float or 0-d tensor, no grad needed) -- chosen side
    is never reweighted, so only the sum is needed.
    ref_rejected_logp: precomputed frozen-reference PER-TOKEN log-probs for the
    rejected response (1D tensor [T_rej], no grad needed) -- needed per-token
    because GAW-PO-lite's w_t must multiply the whole per-token log-ratio
    (policy minus reference), not just the policy term (PLAN.md Step 0, "Exact
    loss placement").
    Returns (loss, extra_dict) where extra_dict carries quantities metrics need.
    """
    chosen_logp, chosen_logits, chosen_targets = forward_logprobs(
        policy_model, chosen_input_ids, chosen_response_mask, requires_grad=True
    )
    rejected_logp, rejected_logits, rejected_targets = forward_logprobs(
        policy_model, rejected_input_ids, rejected_response_mask, requires_grad=True
    )

    # unweighted, chosen side never reweighted: beta * sum_t(logpi - logpi_ref)
    chosen_term = beta * (chosen_logp.sum() - ref_chosen_logp_sum)

    rejected_log_ratio = rejected_logp - ref_rejected_logp  # per-token (policy - ref)
    if method == "vanilla":
        rejected_term = beta * rejected_log_ratio.sum()
    elif method == "gawpolite":
        w_t = gawpolite_weights(
            rejected_logits.detach(), rejected_targets, chosen_logits.detach(), chosen_targets
        )
        rejected_term = beta * (w_t * rejected_log_ratio).sum()
    else:
        raise ValueError(f"unknown method {method!r}")

    logits_diff = chosen_term - rejected_term
    loss = -F.logsigmoid(logits_diff)

    extra = {
        "chosen_logp_sum": chosen_logp.sum().item(),
        "rejected_logp_sum": rejected_logp.sum().item(),
        "chosen_logp_mean": chosen_logp.mean().item(),
        "rejected_logp_mean": rejected_logp.mean().item(),
        "n_chosen_tok": chosen_logp.shape[0],
        "n_rejected_tok": rejected_logp.shape[0],
    }
    return loss, extra
