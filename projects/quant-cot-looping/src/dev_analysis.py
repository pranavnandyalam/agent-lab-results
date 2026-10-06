"""Dev analysis on the 8 GSM8K train problems (retokenized from saved text; core run uses real token ids).
Run from project dir: .venv/bin/python -I src/dev_analysis.py"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from detector import loop_mask
from timed_trial import load_train
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca')
idx, qs, _ = load_train(8, 0); Q = dict(zip(idx, qs))
FILES = {'w4': ['timed_trial_w4'], 'fp32': ['timed_trial_fp32', 'timed_trial_fp32_s12']}
for name, kw in [('frozen (A20 x4 + B)', {}), ('A16', dict(n_a=16)), ('A12', dict(n_a=12))]:
    for lvl, fs in FILES.items():
        fl, tot = {}, 0
        for f in fs:
            for r in json.load(open(f'results/{f}.json'))['rows']:
                tot += 1
                ids = tok.encode(r['text'], add_special_tokens=False)
                k = sum(loop_mask(ids, prompt_ids=tok.encode(Q[r['problem']], add_special_tokens=False), **kw))
                if k: fl[(r['problem'], r['seed'])] = k
        print(name, lvl, f'{len(fl)}/{tot} flagged', fl)
