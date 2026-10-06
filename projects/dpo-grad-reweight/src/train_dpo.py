"""Vanilla DPO / GAW-PO-lite training + eval for dpo-grad-reweight (PLAN.md).

Real run (needs network access to huggingface.co's Xet CDN, which is currently
BLOCKED in this environment — see results/fetch_notes.md):
    python src/train_dpo.py --mode full --method vanilla --beta 0.1 --seed 0

Offline smoke test (no network; real Qwen tokenizer + chat template, from-config
RANDOM weights, hand-written synthetic preference pairs) — validates the training
+ eval + ARC-style scoring code path end to end and reports wall-clock:
    python src/train_dpo.py --mode smoke_test --method vanilla --beta 0.1 --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SPLIT_SEED = 12345  # fixed across all runs/seeds: which examples go to dev/train/eval


# ---------------------------------------------------------------------------
# Data loading (real dataset path)
# ---------------------------------------------------------------------------
def load_ultrafeedback_pairs(dataset_id, revision, hf_home, n_dev, n_train, n_eval, seed=SPLIT_SEED):
    import os
    import random as _random

    os.environ["HF_HOME"] = hf_home
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train_prefs", revision=revision)
    idx = list(range(len(ds)))
    _random.Random(seed).shuffle(idx)
    need = n_dev + n_train + n_eval
    idx = idx[:need]
    dev_idx, train_idx, eval_idx = idx[:n_dev], idx[n_dev:n_dev + n_train], idx[n_dev + n_train:need]

    def extract(i):
        row = ds[i]
        chosen_msgs = row["chosen"]
        rejected_msgs = row["rejected"]
        prompt_msgs = chosen_msgs[:-1]
        return {
            "prompt_msgs": prompt_msgs,
            "chosen_text": chosen_msgs[-1]["content"],
            "rejected_text": rejected_msgs[-1]["content"],
        }

    return (
        [extract(i) for i in dev_idx],
        [extract(i) for i in train_idx],
        [extract(i) for i in eval_idx],
    )


def load_arc_items(arc_dataset_id, revision, hf_home, n):
    import os

    os.environ["HF_HOME"] = hf_home
    from datasets import load_dataset

    ds = load_dataset(arc_dataset_id, "ARC-Easy", split="test", revision=revision)
    ds = ds.select(range(min(n, len(ds))))
    items = []
    for row in ds:
        items.append(
            {
                "question": row["question"],
                "choices": row["choices"]["text"],
                "answer_idx": row["choices"]["label"].index(row["answerKey"])
                if row["answerKey"] in row["choices"]["label"]
                else None,
            }
        )
    return [it for it in items if it["answer_idx"] is not None]


# ---------------------------------------------------------------------------
# Synthetic data for the offline smoke test (no dataset download needed)
# ---------------------------------------------------------------------------
SMOKE_PAIRS = [
    {
        "prompt_msgs": [{"role": "user", "content": "What is the capital of France?"}],
        "chosen_text": "The capital of France is Paris.",
        "rejected_text": "The capital of France is Berlin, which is wrong.",
    },
    {
        "prompt_msgs": [{"role": "user", "content": "Name a primary color."}],
        "chosen_text": "Red is a primary color.",
        "rejected_text": "Purple is considered a primary color by some, incorrectly.",
    },
    {
        "prompt_msgs": [{"role": "user", "content": "Is water wet?"}],
        "chosen_text": "Yes, water is generally described as wet.",
        "rejected_text": "No, water can never be described as wet at all.",
    },
    {
        "prompt_msgs": [{"role": "user", "content": "What do bees produce?"}],
        "chosen_text": "Bees produce honey.",
        "rejected_text": "Bees produce gasoline, which is incorrect.",
    },
]
SMOKE_ARC = [
    {
        "question": "What gas do plants absorb from the atmosphere for photosynthesis?",
        "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"],
        "answer_idx": 1,
    },
    {
        "question": "Which organ pumps blood through the human body?",
        "choices": ["Lungs", "Liver", "Heart", "Kidney"],
        "answer_idx": 2,
    },
]


# ---------------------------------------------------------------------------
# Reference log-prob precompute (frozen, computed once, reused across methods/seeds)
# ---------------------------------------------------------------------------
def precompute_reference(ref_model, tokenizer, pairs, max_len, device="cpu"):
    out = []
    for p in pairs:
        c = common.tokenize_pair(tokenizer, p["prompt_msgs"], p["chosen_text"], max_len)
        r = common.tokenize_pair(tokenizer, p["prompt_msgs"], p["rejected_text"], max_len)
        c_ids = torch.tensor(c["input_ids"], dtype=torch.long)
        c_mask = torch.tensor(c["response_mask"], dtype=torch.long)
        r_ids = torch.tensor(r["input_ids"], dtype=torch.long)
        r_mask = torch.tensor(r["response_mask"], dtype=torch.long)
        c_logp, _, _ = common.forward_logprobs(ref_model, c_ids, c_mask, requires_grad=False)
        r_logp, _, _ = common.forward_logprobs(ref_model, r_ids, r_mask, requires_grad=False)
        out.append(
            {
                "chosen_ids": c_ids,
                "chosen_mask": c_mask,
                "rejected_ids": r_ids,
                "rejected_mask": r_mask,
                "ref_chosen_logp": c_logp.detach(),
                "ref_rejected_logp": r_logp.detach(),
                "ref_chosen_logp_sum": c_logp.sum().item(),
                "ref_rejected_logp_sum": r_logp.sum().item(),
                "ref_chosen_logp_mean": c_logp.mean().item(),
                "ref_rejected_logp_mean": r_logp.mean().item(),
                "truncated": c["truncated"] or r["truncated"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Training (1 epoch over train pairs)
# ---------------------------------------------------------------------------
def train_one_epoch(policy_model, train_cache, beta, method, lr, seed):
    common.set_all_seeds(seed)
    order = list(range(len(train_cache)))
    import random as _random

    _random.Random(seed).shuffle(order)
    opt = torch.optim.RMSprop(policy_model.parameters(), lr=lr)
    policy_model.train()
    losses = []
    for i in order:
        item = train_cache[i]
        opt.zero_grad()
        loss, extra = common.dpo_loss_one_pair(
            policy_model,
            item["ref_chosen_logp_sum"],
            item["ref_rejected_logp"],
            item["chosen_ids"],
            item["chosen_mask"],
            item["rejected_ids"],
            item["rejected_mask"],
            beta,
            method,
        )
        loss.backward()
        opt.step()
        losses.append(loss.item())
    policy_model.eval()
    return losses


# ---------------------------------------------------------------------------
# Metrics (5, on eval split, paired per-item) + ARC
# ---------------------------------------------------------------------------
def eval_metrics(policy_model, eval_cache, beta):
    n = len(eval_cache)
    len_norm_correct = 0
    raw_correct = 0
    margin_sign_correct = 0
    margins = []
    chosen_drifts = []
    per_item = []
    for item in eval_cache:
        c_logp, _, _ = common.forward_logprobs(
            policy_model, item["chosen_ids"], item["chosen_mask"], requires_grad=False
        )
        r_logp, _, _ = common.forward_logprobs(
            policy_model, item["rejected_ids"], item["rejected_mask"], requires_grad=False
        )
        c_mean, r_mean = c_logp.mean().item(), r_logp.mean().item()
        c_sum, r_sum = c_logp.sum().item(), r_logp.sum().item()
        if c_mean > r_mean:
            len_norm_correct += 1
        if c_sum > r_sum:
            raw_correct += 1
        margin = beta * (c_sum - item["ref_chosen_logp_sum"]) - beta * (
            r_sum - item["ref_rejected_logp_sum"]
        )
        margins.append(margin)
        if margin > 0:
            margin_sign_correct += 1
        chosen_drifts.append(c_mean - item["ref_chosen_logp_mean"])
        per_item.append(
            {"c_mean": c_mean, "r_mean": r_mean, "c_sum": c_sum, "r_sum": r_sum, "margin": margin}
        )

    # KL(policy || reference) on first response token given the prompt (forward-only)
    kls = []
    for item in eval_cache:
        prompt_len = int((item["chosen_mask"] == 0).sum().item())
        if prompt_len < 1 or prompt_len >= item["chosen_ids"].shape[0]:
            continue
        prompt_ids = item["chosen_ids"][:prompt_len].unsqueeze(0)
        with torch.no_grad():
            pol_logits = policy_model(prompt_ids).logits[0, -1, :]
        kls.append(float("nan"))  # placeholder overwritten below if ref logits available
    # NOTE: KL needs reference next-token logits too; computed in run() where both
    # models are in scope (kept here as a stub signature for clarity/testing).

    metrics = {
        "n_eval": n,
        "length_norm_logprob_acc": len_norm_correct / n,
        "raw_sumlogprob_acc_secondary": raw_correct / n,
        "implicit_reward_margin_sign_acc": margin_sign_correct / n,
        "implicit_reward_margin_mean": sum(margins) / n,
        "chosen_logprob_drift_mean": sum(chosen_drifts) / n,
    }
    return metrics, per_item


def eval_kl_first_token(policy_model, ref_model, eval_cache):
    kls = []
    for item in eval_cache:
        prompt_len = int((item["chosen_mask"] == 0).sum().item())
        if prompt_len < 1 or prompt_len >= item["chosen_ids"].shape[0]:
            continue
        prompt_ids = item["chosen_ids"][:prompt_len].unsqueeze(0)
        with torch.no_grad():
            pol_logits = policy_model(prompt_ids).logits[0, -1, :]
            ref_logits = ref_model(prompt_ids).logits[0, -1, :]
        pol_logp = F.log_softmax(pol_logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        kl = F.kl_div(ref_logp, pol_logp, log_target=True, reduction="sum").item()
        kls.append(kl)
    return sum(kls) / len(kls) if kls else float("nan")


def eval_arc_loglik(model, tokenizer, arc_items, max_len):
    correct = 0
    for it in arc_items:
        scores = []
        for choice in it["choices"]:
            prompt_msgs = [{"role": "user", "content": it["question"]}]
            enc = common.tokenize_pair(tokenizer, prompt_msgs, choice, max_len)
            ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            mask = torch.tensor(enc["response_mask"], dtype=torch.long)
            logp, _, _ = common.forward_logprobs(model, ids, mask, requires_grad=False)
            scores.append(logp.sum().item())
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == it["answer_idx"]:
            correct += 1
    return correct / len(arc_items) if arc_items else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "smoke_test"], default="smoke_test")
    ap.add_argument("--method", choices=["vanilla", "gawpolite"], default="vanilla")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--n_dev", type=int, default=50)
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_eval", type=int, default=80)
    ap.add_argument("--n_arc", type=int, default=200)
    ap.add_argument("--max_seq_len", type=int, default=common.MAX_SEQ_LEN)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model_revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    ap.add_argument("--dataset_id", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--dataset_revision", default="3949bf5f8c17c394422ccfab0c31ea9c20bdeb85")
    ap.add_argument("--arc_dataset_id", default="allenai/ai2_arc")
    ap.add_argument("--arc_revision", default="210d026faf9955653af8916fad021475a3f00453")
    ap.add_argument("--hf_home", default=str(Path.home() / "models" / "hf_cache"))
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    common.set_all_seeds(args.seed)
    t0 = time.time()
    timing = {}

    if args.mode == "smoke_test":
        import os

        os.environ["HF_HOME"] = args.hf_home
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, revision=args.model_revision, local_files_only=True
        )
        cfg = AutoConfig.from_pretrained(
            args.model_id, revision=args.model_revision, local_files_only=True
        )
        # shrink for speed; real tokenizer vocab/chat-template, random weights
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_hidden_layers = 2
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.dtype = torch.float32  # real Qwen2.5-0.5B config.json defaults to bf16;
        cfg.torch_dtype = torch.float32  # force fp32 for this offline smoke test
        common.set_all_seeds(args.seed)
        ref_model = AutoModelForCausalLM.from_config(cfg)
        ref_model.eval()
        common.set_all_seeds(args.seed)
        policy_model = AutoModelForCausalLM.from_config(cfg)
        policy_model.load_state_dict(ref_model.state_dict())
        policy_model.eval()
        dev_pairs, train_pairs, eval_pairs = (
            SMOKE_PAIRS[:2],
            SMOKE_PAIRS,
            SMOKE_PAIRS[2:],
        )
        arc_items = SMOKE_ARC
    else:
        policy_model, tokenizer = common.load_model_and_tokenizer(
            args.model_id, args.model_revision, args.hf_home
        )
        ref_model, _ = common.load_model_and_tokenizer(
            args.model_id, args.model_revision, args.hf_home
        )
        dev_pairs, train_pairs, eval_pairs = load_ultrafeedback_pairs(
            args.dataset_id, args.dataset_revision, args.hf_home,
            args.n_dev, args.n_train, args.n_eval,
        )
        arc_items = load_arc_items(args.arc_dataset_id, args.arc_revision, args.hf_home, args.n_arc)

    timing["load_s"] = time.time() - t0

    t1 = time.time()
    train_cache = precompute_reference(ref_model, tokenizer, train_pairs, args.max_seq_len)
    eval_cache = precompute_reference(ref_model, tokenizer, eval_pairs, args.max_seq_len)
    timing["precompute_ref_s"] = time.time() - t1

    t2 = time.time()
    losses = train_one_epoch(policy_model, train_cache, args.beta, args.method, args.lr, args.seed)
    timing["train_s"] = time.time() - t2

    t3 = time.time()
    metrics, _ = eval_metrics(policy_model, eval_cache, args.beta)
    metrics["kl_policy_ref_first_token_mean"] = eval_kl_first_token(policy_model, ref_model, eval_cache)
    timing["eval_metrics_s"] = time.time() - t3

    t4 = time.time()
    arc_acc = eval_arc_loglik(policy_model, tokenizer, arc_items, args.max_seq_len)
    metrics["arc_easy_loglik_acc"] = arc_acc
    metrics["n_arc"] = len(arc_items)
    timing["eval_arc_s"] = time.time() - t4

    timing["total_s"] = time.time() - t0

    result = {
        "args": vars(args),
        "n_train_truncated": sum(1 for x in train_cache if x["truncated"]),
        "n_eval_truncated": sum(1 for x in eval_cache if x["truncated"]),
        "final_train_loss": losses[-1] if losses else None,
        "mean_train_loss": sum(losses) / len(losses) if losses else None,
        "metrics": metrics,
        "timing": timing,
    }
    if args.mode == "smoke_test":
        # Explicit, always-present markers so this can never be mistaken for a
        # real result later: --n_* CLI args above are the REAL-RUN targets, not
        # what was actually used in smoke mode (random from-config weights,
        # synthetic hand-written pairs, tiny actual counts).
        result["PIPELINE_VALIDATION_ONLY_NOT_A_RESULT"] = True
        result["weights"] = "random_from_config"
        result["data"] = "synthetic_handwritten"
        result["actual_counts"] = {
            "n_dev": len(dev_pairs),
            "n_train": len(train_pairs),
            "n_eval": len(eval_pairs),
            "n_arc": len(arc_items),
        }
    print(json.dumps(result, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    main()
