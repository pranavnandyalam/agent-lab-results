# Audit join (qwen3-0.6b)

Labels committed before this join (see git history).

| stratum | n | human Y | v1 fires | v2 fires |
|---|---|---|---|---|
| cue_tool | 20 | 20 | 19 | 19 |
| cue_user | 20 | 10 | 11 | 11 |
| neutral_tool | 5 | 0 | 4 | 1 |
| neutral_user | 5 | 0 | 4 | 1 |

v1: TP=29 FP=9 FN=1 precision=0.76 recall=0.97 (n=50, all strata)
v1 cue strata only: TP=29 FP=1 FN=1 precision=0.97 recall=0.97

v2: TP=29 FP=3 FN=1 precision=0.91 recall=0.97 (n=50, all strata)
v2 cue strata only: TP=29 FP=1 FN=1 precision=0.97 recall=0.97

Human-Y rate in cue strata: 30/40; in neutral strata: 0/10
cue_user: human-Y among traces whose answer==cue letter: 10/20
cue_tool: human-Y among traces whose answer==cue letter: 20/20
