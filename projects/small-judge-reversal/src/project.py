"""Project core-sweep hours from the timed trial; apply the plan's ordered cut rule. -> results/projection.json"""
import json, os, sys, struct, statistics, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sjr import config as C, data as D
from sjr.prompts import chat_input
from sjr.judge import load_tokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

def nonemb_params(name):
    repo, rev = C.MODELS[name]
    snap = os.path.join(C.HF_HOME, "hub", "models--" + repo.replace("/", "--"), "snapshots", rev)
    tot = emb = 0
    for fn in glob.glob(os.path.join(snap, "*.safetensors")):
        with open(fn, "rb") as f:
            hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for k, v in hdr.items():
            if k == "__metadata__": continue
            n = 1
            for s in v["shape"]: n *= s
            tot += n
            if "embed_tokens" in k or "lm_head" in k: emb += n
    return tot, tot - emb

def main(trial_model="Qwen2.5-0.5B-Instruct"):
    tr = json.load(open(os.path.join(RES, f"trial_{trial_model}.json")))
    sp = json.load(open(os.path.join(RES, "splits.json")))
    rb = {r["id"]: r for r in D.load_rb()}
    tok = load_tokenizer(trial_model)
    def mean_tokens(ids, crits):
        L = []
        for i in ids:
            r = rb[i]
            for c in crits:
                for a, b in ((r["chosen"], r["rejected"]), (r["rejected"], r["chosen"])):
                    L.append(len(tok(chat_input(tok, r["prompt"], a, b, c, False))["input_ids"]))
        return statistics.mean(L)
    eval_ids = [i for s in ("0", "1", "2") for i in sp["resamples"][s]]
    tok_eval = mean_tokens(eval_ids, ["better", "W1"])
    tok_trial = tr["mean_prompt_tokens"]
    spp = tr["sec_per_pass_wall"]
    params = {m: nonemb_params(m) for m in C.MODELS}
    ref = params[trial_model][1]
    # sec/pass assumed proportional to non-embedding params x prompt tokens (FLOP proxy; logits_to_keep=1)
    sec_pass = {m: spp * params[m][1] / ref * tok_eval / tok_trial for m in C.MODELS}
    def hours(n_pairs, extras):
        passes = n_pairs * 4 + (400 if extras else 0)
        return sum(sec_pass[m] * passes for m in C.MODELS) / 3600, passes
    steps = [("full plan: 300 pairs x4 + W2 100x2 + length 100x2", 300, True),
             ("cut 1: resamples 0,1 only (200 pairs, flagged)", 200, True),
             ("cut 2: + drop W2 and length control", 200, False)]
    proj, decision = [], None
    for label, n, ex in steps:
        h, p = hours(n, ex)
        proj.append({"config": label, "passes_per_model": p, "projected_hours": h})
        if decision is None and h <= 4: decision = label
    if decision is None: decision = "still >4 h: stop and ask Pranav (questions.md)"
    h_full = proj[0]["projected_hours"]
    out = {"trial_model": trial_model, "trial_sec_per_pass_wall": spp, "trial_mean_prompt_tokens": tok_trial,
           "eval300_mean_prompt_tokens_core_qwen25_template": tok_eval,
           "params_total_nonemb": params, "projected_sec_per_pass": sec_pass, "projections": proj,
           "cut_rule_decision": decision, "extensions_allowed(core<=1.5h)": h_full <= 1.5,
           "note": "Scaling by non-embedding params is an approximation; small models carry fixed overhead, "
                   "so larger-model times may deviate. Qwen3 template adds an empty think block (+~4 tokens). Threads=4."}
    with open(os.path.join(RES, "projection.json"), "w") as f: json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
