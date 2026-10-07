# Analysis: qwen3-0.6b

_Produced by an autonomous AI agent (Claude) on behalf of @pranavnandyalam. Not peer reviewed._

Rows parsed: 540 (unparseable lines skipped: 0); items with any row: 60. Bootstrap: item-level, 2000 resamples, seed 0, 95% percentile CI. v1 = frozen verbalize.py ('notices insertion'); v2 = verbalize_v2.py (source->answer link, post hoc). Neither is validated until the 50-trace audit.

## Per-cell quality

| cell | n | trunc | parse fail | P(A) | P(cued) |
|---|---|---|---|---|---|
| nocue | 180 | 0.028 | 0.000 | 0.594 | n/a |
| neutral_user | 60 | 0.017 | 0.000 | 0.633 | n/a |
| neutral_tool | 60 | 0.017 | 0.017 | 0.746 | n/a |
| cueA_user | 60 | 0.000 | 0.000 | 0.950 | 0.950 |
| cueB_user | 60 | 0.017 | 0.000 | 0.133 | 0.867 |
| cueA_tool | 60 | 0.000 | 0.000 | 0.983 | 0.983 |
| cueB_tool | 60 | 0.000 | 0.000 | 0.017 | 0.983 |

## Positional bias

P(A) all valid = 0.583; P(A) nocue = 0.594 (swapped 0.644, unswapped 0.544); items with all nocue answers A = 0.417 of 60.

## Channel: user

- Net follow (items 60, traces 120): P(cued|cue) 0.908 - P(cued|nocue) 0.500 = 0.408 CI [0.350, 0.458]
- Net follow vs channel neutral: 0.408 CI [0.350, 0.458]
- Sensitivity (parse fail = not followed): 0.408 CI [0.350, 0.458]
- Switch items: 50/60 = 0.833 CI [0.733, 0.917]
- Chance switch rate mean p(1-p): 0.082 CI [0.057, 0.108] -> floor verdict: **NOT FLOOR (above chance interval)**
- Neutral flip rate (vs nocue majority): 0.167 CI [0.083, 0.267] (n=60)
- Secondary followers: 51/120 traces; VCR v1 0.706, v2 0.686
- VCR among switch traces (n=100): v1 0.610, v2 0.610
- VCR excl. truncated (n=100): v1 0.610, v2 0.610
- VCR item-level (either trace, n=50): v1 0.740, v2 0.760
- Neutral-cell mention rate (n=60): v1 0.800, v2 0.150

## Channel: tool

- Net follow (items 60, traces 120): P(cued|cue) 0.983 - P(cued|nocue) 0.500 = 0.483 CI [0.458, 0.500]
- Net follow vs channel neutral: 0.483 CI [0.458, 0.500]
- Sensitivity (parse fail = not followed): 0.483 CI [0.458, 0.500]
- Switch items: 58/60 = 0.967 CI [0.917, 1.000]
- Chance switch rate mean p(1-p): 0.120 CI [0.094, 0.147] -> floor verdict: **NOT FLOOR (above chance interval)**
- Neutral flip rate (vs nocue majority): 0.407 CI [0.288, 0.525] (n=59)
- Secondary followers: 58/120 traces; VCR v1 0.897, v2 0.948
- VCR among switch traces (n=116): v1 0.879, v2 0.914
- VCR excl. truncated (n=116): v1 0.879, v2 0.914
- VCR item-level (either trace, n=58): v1 0.948, v2 0.983
- Neutral-cell mention rate (n=60): v1 0.950, v2 0.167

## Paired subset (switch in both channels)

Items: 48
- user: traces 96, VCR v1 0.635, v2 0.635
- tool: traces 96, VCR v1 0.906, v2 0.938

v2 neutral mention rate >10% in any channel: True (if True, VCR is audit-based only per PLAN rev 4).
