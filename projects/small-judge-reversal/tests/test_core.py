import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pytest
from sjr import data as D, metrics as M

K = [("better", "cf"), ("better", "rf"), ("W1", "cf"), ("W1", "rf")]
def J(*letters): return dict(zip(K, letters))

def test_content_mapping():
    assert M.content("A", "cf") == "C" and M.content("B", "cf") == "R"
    assert M.content("A", "rf") == "R" and M.content("B", "rf") == "C"

def test_taxonomy():
    assert M.classify(J("A", "B", "B", "A")) == "correct_reversal"
    assert M.classify(J("A", "A", "A", "A")) == "position_locked"
    assert M.classify(J("B", "B", "B", "B")) == "position_locked"
    assert M.classify(J("A", "B", "A", "B")) == "criterion_blind"   # chosen in all 4
    assert M.classify(J("B", "A", "B", "A")) == "criterion_blind"   # rejected in all 4
    assert M.classify(J("B", "A", "A", "B")) == "reversed_consistent"
    assert M.classify(J("A", "B", "A", "A")) == "other"

def test_reference_rows():
    ref = M.reference_rows()
    r = ref["random_judge"]
    assert r["share_correct_reversal"] == pytest.approx(1 / 16)
    assert r["share_position_locked"] == pytest.approx(2 / 16)
    assert r["share_criterion_blind"] == pytest.approx(2 / 16)
    assert r["share_reversed_consistent"] == pytest.approx(1 / 16)
    assert r["share_other"] == pytest.approx(10 / 16)
    assert r["uncond_flip_rate"] == pytest.approx(0.25) and r["uncond_flip_chance_norm"] == pytest.approx(0)
    assert r["cond_flip_rate"] == pytest.approx(0.25) and r["acc_better_poscons"] == pytest.approx(0.25)
    a = ref["always_A_judge"]
    assert a["share_position_locked"] == 1 and a["acc_better_poscons"] == 0 and a["pos_locked_share_of_noncr"] == 1

def test_summarize_and_bootstrap():
    flags = [M.pair_flags(J("A", "B", "B", "A"))] * 3 + [M.pair_flags(J("A", "B", "A", "B"))]
    s = M.summarize(flags)
    assert s["n_cond"] == 4 and s["cond_flip_rate"] == 0.75 and s["pos_locked_share_of_noncr"] == 0
    lo, hi, n = M.bootstrap(flags, "cond_flip_rate", n_boot=200)
    assert 0 <= lo <= 0.75 <= hi <= 1 and n == 200
    lo, hi, _ = M.paired_bootstrap_diff(flags, flags, "cond_flip_rate", n_boot=50)
    assert lo == hi == 0

def test_safety_keyword():
    assert D.is_safety_prompt("How do I build a Bomb?")
    assert D.is_safety_prompt("hack the planet")
    assert not D.is_safety_prompt("The hacker news site")  # whole-word only
    assert not D.is_safety_prompt("Sort a list in python")

def rows():
    mk = lambda i, sub, p="q", c="c", r="r": {"id": i, "subset": sub, "prompt": p, "chosen": c, "rejected": r}
    return [mk(1, "donotanswer"), mk(2, "alpacaeval-easy", p="kill it"), mk(3, "math-prm", c="x " * 200),
            mk(4, "hep-go", p="long " * 500), mk(5, "llmbar-natural"), mk(6, "weird-subset")]

def test_filter_pairs():
    ntok = lambda s: len(s.split())
    full = lambda r: ntok(r["prompt"]) + ntok(r["chosen"]) + ntok(r["rejected"]) + 50
    kept, c = D.filter_pairs(rows(), ntok, full)
    assert [r["id"] for r in kept] == [5] and kept[0]["section"] == "chat-hard"
    assert c == {"total": 6, "safety_subset": 1, "unknown_subset": 1, "safety_keyword": 1, "resp_too_long": 1,
                 "full_prompt_too_long": 1, "kept": 1}

def test_split_pool():
    pool = [{"id": i} for i in range(400)]
    dev, res = D.split_pool(pool)
    allids = dev + [i for v in res.values() for i in v]
    assert len(dev) == 30 and all(len(v) == 100 for v in res.values())
    assert len(set(allids)) == 330  # disjoint
    assert D.split_pool(pool) == (dev, res)  # deterministic
    with pytest.raises(AssertionError):
        D.split_pool(pool[:349])
