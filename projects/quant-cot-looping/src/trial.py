"""Timed decode-throughput trial + fake-quant sanity check for Qwen3-0.6B on CPU.

Usage:
  python -I src/trial.py throughput --dtype bf16 --batch 12 --bits 4 --group 64 --max-new 256 --seed 0
  python -I src/trial.py sanity
Results are merged into results/trial.json under a key per run.
"""
import argparse, json, os, sys, time, platform
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/models/hf"))
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant import fake_quantize_model

torch.set_num_threads(4)
MODEL = "Qwen/Qwen3-0.6B"
REV = "c1899de289a04d12100db370d81485cdf75e47ca"
DS_REV = "740312add88f781978c0658806c59bc2815b9866"
OUT = os.path.join(HERE, "results", "trial.json")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load_questions(n, seed=0):
    from huggingface_hub import snapshot_download
    p = snapshot_download("openai/gsm8k", repo_type="dataset", revision=DS_REV, allow_patterns=["main/*.parquet"])
    df = pd.read_parquet(os.path.join(p, "main", "test-00000-of-00001.parquet"))
    idx = np.random.default_rng(seed).permutation(len(df))[:n]
    return [int(i) for i in idx], [df.iloc[int(i)]["question"] for i in idx]


def load(dtype):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=DTYPES[dtype],
                                                 use_safetensors=True, trust_remote_code=False)
    model.eval()
    return tok, model


def build(tok, qs):
    texts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=True) for q in qs]
    return tok(texts, return_tensors="pt", padding=True)


def save(key, rec):
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data[key] = rec
    tmp = OUT + ".tmp"
    json.dump(data, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


def env():
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "threads": torch.get_num_threads(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "model": MODEL, "revision": REV, "gsm8k_revision": DS_REV}


@torch.no_grad()
def throughput(a):
    torch.manual_seed(a.seed)
    tok, model = load(a.dtype)
    nq = fake_quantize_model(model, a.bits, a.group) if a.bits < 16 else 0
    idx, qs = load_questions(a.batch, seed=0)
    enc = build(tok, qs)
    # prefill timing (separate forward pass over prompt batch)
    t0 = time.perf_counter(); model(**enc); t_prefill = time.perf_counter() - t0
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    out = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                         max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
    t_total = time.perf_counter() - t0
    gen = out[:, enc["input_ids"].shape[1]:]
    eos_ids = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    lens = []
    for row in gen.tolist():
        L = len(row)
        for j, t in enumerate(row):
            if t in eos_ids:
                L = j + 1; break
        lens.append(L)
    steps = gen.shape[1]
    t_decode = max(t_total - t_prefill, 1e-9)
    rec = {"dtype": a.dtype, "batch": a.batch, "bits": a.bits, "group": a.group, "seed": a.seed,
           "max_new_tokens": a.max_new, "n_linear_quantized": nq, "gsm8k_test_idx": idx,
           "prompt_len_padded": int(enc["input_ids"].shape[1]), "decode_steps": int(steps),
           "t_prefill_s": t_prefill, "t_generate_total_s": t_total, "t_decode_est_s": t_decode,
           "gen_tokens_real": lens, "mean_gen_tokens": float(np.mean(lens)),
           "n_truncated": int(sum(l >= a.max_new for l in lens)),
           "agg_decode_tok_s_real": sum(lens) / t_decode,           # real (non-pad) tokens / decode time
           "agg_decode_tok_s_slots": steps * a.batch / t_decode,     # batch x steps / decode time
           "agg_total_tok_s_real": sum(lens) / t_total,
           "sample_output_0": tok.decode(gen[0], skip_special_tokens=False)[:600], "env": env()}
    key = f"throughput_{a.dtype}_b{a.batch}_w{a.bits}g{a.group}_n{a.max_new}_s{a.seed}"
    save(key, rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("gen_tokens_real", "sample_output_0", "gsm8k_test_idx", "env")}, indent=1))


@torch.no_grad()
def sanity(a):
    """Teacher-forced next-token agreement & KL vs unquantized bf16 on 4 prompts; plus short greedy text."""
    _, qs = load_questions(4, seed=0)
    tok, base = load("bf16")
    enc = build(tok, qs)
    ref = base.generate(**enc, do_sample=False, max_new_tokens=48, pad_token_id=tok.pad_token_id)
    mask = torch.ones_like(ref); mask[:, :enc["input_ids"].shape[1]] = enc["attention_mask"]
    ref_logits = base(input_ids=ref, attention_mask=mask).logits.float()
    P = enc["input_ids"].shape[1]
    sl = slice(P - 1, ref.shape[1] - 1)
    ref_lp = torch.log_softmax(ref_logits[:, sl], -1)
    rec = {"env": env(), "ref_text0": tok.decode(ref[0, P:], skip_special_tokens=True)}
    del base
    for bits in (8, 4, 3, 2):
        _, m = load("bf16")
        fake_quantize_model(m, bits, 64)
        lg = m(input_ids=ref, attention_mask=mask).logits.float()[:, sl]
        lp = torch.log_softmax(lg, -1)
        kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1).mean().item()
        agree = (lg.argmax(-1) == ref_logits[:, sl].argmax(-1)).float().mean().item()
        g = m.generate(**enc, do_sample=False, max_new_tokens=48, pad_token_id=tok.pad_token_id)
        greedy_match = (g[:, P:] == ref[:, P:]).float().mean().item()
        rec[f"w{bits}g64"] = {"kl_vs_bf16": kl, "top1_agree_teacher_forced": agree,
                              "greedy48_token_match": greedy_match,
                              "text0": tok.decode(g[0, P:], skip_special_tokens=True)}
        print(bits, f"KL={kl:.4f} top1={agree:.3f} greedy_match={greedy_match:.3f}", repr(rec[f'w{bits}g64']['text0'][:120]))
        del m
    print("ref", repr(rec["ref_text0"][:120]))
    save("sanity_bf16_g64", rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["throughput", "sanity"])
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    throughput(a) if a.mode == "throughput" else sanity(a)
