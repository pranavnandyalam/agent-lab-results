"""Batched CPU generation for the cue-verbalization grid. Resumable per (item, cell, seed).

Own sampling loop (KV cache) so each row has its own torch.Generator seeded from
(item, cell, seed): results do not depend on batch composition (up to float noise from
padding). Sampling: T=0.6, top_k=20, top_p=0.95 (Qwen3 thinking-mode defaults).
On truncation (max_new_tokens without EOS): append '\\n</think>\\n\\n' (if absent) +
'Answer:' and greedily generate a few tokens to get the answer.

Example: python -I src/run.py --model qwen3-0.6b --limit 60 --batch-size 8
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import re
import sys
import time
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import prompts as P  # noqa: E402
import verbalize as V  # noqa: E402

ANS_RE = re.compile(r"Answer\s*[:：]\s*\**\s*\(?\s*([AB])\b", re.IGNORECASE)


def gen_seed(item_id, cell, seed):
    return C.BASE_SEED + zlib.crc32(f"{item_id}|{cell}|{seed}".encode()) % 1_000_000


def sample_rows(logits, gens, greedy):
    if greedy:
        return logits.argmax(-1)
    logits = logits / C.TEMPERATURE
    topv, topi = logits.topk(C.TOP_K, dim=-1)
    probs = torch.softmax(topv, -1)
    cum = probs.cumsum(-1)
    probs = probs.masked_fill(cum - probs > C.TOP_P, 0.0)
    probs = probs / probs.sum(-1, keepdim=True)
    out = []
    for r in range(probs.shape[0]):
        j = torch.multinomial(probs[r], 1, generator=gens[r])
        out.append(topi[r, j])
    return torch.cat(out)


@torch.inference_mode()
def generate(model, tok, texts, seeds, max_new, greedy=False):
    """Left-padded batched generation. Returns (list of new-token-id lists, list hit_eos)."""
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    B = ids.shape[0]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()
    new = [[] for _ in range(B)]
    done = [False] * B
    hit_eos = [False] * B
    for _ in range(max_new):
        nxt = sample_rows(logits, gens, greedy)
        for r in range(B):
            if not done[r]:
                t = int(nxt[r])
                if t in eos:
                    done[r] = hit_eos[r] = True
                else:
                    new[r].append(t)
        if all(done):
            break
        nxt = torch.where(torch.tensor(done), torch.tensor(tok.pad_token_id), nxt)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype)], -1)
        pos = pos[:, -1:] + 1
        out = model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
    return new, hit_eos


def split(text):
    if "</think>" in text:
        th, ans = text.split("</think>", 1)
        return th.replace("<think>", "").strip(), ans.strip()
    return text.replace("<think>", "").strip(), ""


def parse_answer(ans_text):
    m = ANS_RE.findall(ans_text)
    return m[-1].upper() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b", choices=list(C.MODELS))
    ap.add_argument("--limit", type=int, default=60, help="first N items")
    ap.add_argument("--cells", default=",".join(P.CELLS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-jobs", type=int, default=0, help="stop after N new gens (0=all)")
    ap.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--tag", default="", help="output subdir suffix, e.g. _trial")
    a = ap.parse_args()

    torch.set_num_threads(C.NUM_THREADS)
    repo, rev = C.MODELS[a.model]
    outdir = C.ROOT / "results" / f"{a.model}{a.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    outf = outdir / "gens.jsonl"
    done = set()
    if outf.exists():
        for line in outf.read_text().splitlines():
            r = json.loads(line)
            done.add((r["item"], r["cell"], r["seed"]))

    items = P.load_items()[: a.limit]
    cells = [c for c in a.cells.split(",") if c]
    assert all(c in P.CELLS for c in cells), cells
    todo = [j for j in P.jobs(items, cells) if (j[0]["id"], j[1], j[2]) not in done]
    if a.max_jobs:
        todo = todo[: a.max_jobs]
    print(f"model={repo}@{rev} todo={len(todo)} already_done={len(done)} out={outf}", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, dtype=torch.float32, use_safetensors=True,
        trust_remote_code=False).eval()

    t_all = time.time()
    for b0 in range(0, len(todo), a.batch_size):
        batch = todo[b0: b0 + a.batch_size]
        texts = [P.render(tok, it, cell) for it, cell, _ in batch]
        seeds = [gen_seed(it["id"], cell, s) for it, cell, s in batch]
        t0 = time.time()
        new, hit_eos = generate(model, tok, texts, seeds, a.max_new_tokens)
        raws = [tok.decode(n, skip_special_tokens=False) for n in new]
        # forced answer for truncated rows
        trunc = [i for i in range(len(batch)) if not hit_eos[i]]
        forced = {}
        if trunc:
            ftexts = []
            for i in trunc:
                suffix = ("" if "</think>" in raws[i] else "\n</think>\n\n") + "Answer:"
                forced[i] = suffix
                ftexts.append(texts[i] + raws[i] + suffix)
            fnew, _ = generate(model, tok, ftexts, [0] * len(trunc), C.ANSWER_MAX_TOKENS, greedy=True)
            for k, i in enumerate(trunc):
                forced[i] += tok.decode(fnew[k], skip_special_tokens=False)
        dt = time.time() - t0
        with outf.open("a") as f:
            for i, (it, cell, s) in enumerate(batch):
                full = raws[i] + forced.get(i, "")
                think, ans = split(full)
                letter = parse_answer(ans)
                cued = "A" if cell.startswith("cueA") else "B" if cell.startswith("cueB") else None
                rec = {
                    "model": repo, "revision": rev, "item": it["id"], "cell": cell, "seed": s,
                    "gen_seed": seeds[i], "swapped": it["swapped"], "cued": cued,
                    "n_new_tokens": len(new[i]), "truncated": not hit_eos[i],
                    "think_closed_naturally": "</think>" in raws[i],
                    "forced_suffix": forced.get(i), "answer": letter, "parse_ok": letter is not None,
                    "followed": (letter == cued) if cued and letter else None,
                    **V.score(think, ans),
                    "batch_seconds": round(dt, 2), "batch_size": len(batch),
                    "thinking": think, "answer_text": ans,
                }
                f.write(json.dumps(rec) + "\n")
        print(f"batch {b0 // a.batch_size}: {len(batch)} gens in {dt:.1f}s "
              f"({dt / len(batch):.1f} s/gen), trunc={len(trunc)}", flush=True)
    print(f"total {time.time() - t_all:.1f}s for {len(todo)} gens", flush=True)


if __name__ == "__main__":
    main()
