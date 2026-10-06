"""Loop detector on token-id sequences (PLAN rev 3b). Parameters are frozen after dev (see PLAN).

A position i is a loop token if it belongs to a repeat (not the first occurrence) of
  (A) a 20-gram (not copied from the problem text) that has already occurred >= 3 times earlier within the previous 1024 tokens
      (i.e. it is the 4th+ occurrence within a 1024-token window); once that happens the spans of the 2nd and later
      occurrences in the window are marked (the first occurrence is not), or
  (B) an n-gram of >= 8 tokens (n = 8..64) repeated >= 3 times consecutively (back-to-back);
      tokens of the 2nd and later copies are loop tokens.
Loop tokens = union of marked positions; distinct = total - loop.
"""
from collections import defaultdict

N_A, MIN_OCC, WINDOW = 20, 4, 1024
N_B_MIN, N_B_MAX, REP_B = 8, 64, 3


def loop_mask(ids, prompt_ids=(), n_a=N_A, min_occ=MIN_OCC, window=WINDOW, nb_min=N_B_MIN, nb_max=N_B_MAX, rep_b=REP_B):
    L = len(ids)
    pg = {tuple(prompt_ids[i:i + n_a]) for i in range(len(prompt_ids) - n_a + 1)}  # 20-grams copied from the problem text
    mask = [False] * L
    # (A) 20-gram reaching 4 occurrences within window
    occ = defaultdict(list)
    for i in range(L - n_a + 1):
        g = tuple(ids[i:i + n_a])
        if g in pg:
            continue  # quoting the problem statement is not a loop (dev finding, cycle 3)
        lst = occ[g]
        recent = [j for j in lst if i - j < window]
        if len(recent) >= min_occ - 1:
            # mark every occurrence after the first one inside the window (2nd..current)
            for st in recent[1:] + [i]:
                for k in range(st, st + n_a):
                    mask[k] = True
        lst.append(i)
    # (B) consecutive repeats
    for n in range(nb_min, nb_max + 1):
        i = 0
        while i + n * rep_b <= L:
            blk = ids[i:i + n]
            r = 1
            while i + (r + 1) * n <= L and ids[i + r * n:i + (r + 1) * n] == blk:
                r += 1
            if r >= rep_b:
                for k in range(i + n, i + r * n):
                    mask[k] = True
                i += r * n
            else:
                i += 1
    return mask


def loop_tokens(ids, **kw):
    return sum(loop_mask(ids, **kw))
