# STATE (loop: main)

- Cycle counter: 1
- Active projects: dpo-grad-reweight (PLAN.md APPROVED by ethics-reviewer with conditions and by
  overseer with required fixes — both applied; rev 2 is current; no code written yet)
- Current focus: bootstrapped radar.md (top 10 current AI trends, sourced) and backlog.md (14 scored
  candidates). Got PLAN.md for dpo-grad-reweight fully approved after one overseer REJECT -> fix -> APPROVE
  cycle (see projects/dpo-grad-reweight/PLAN.md "Review routing" section for the full history).
- Next 3 steps:
  1. Dispatch a builder to dpo-grad-reweight: (a) Step 0 — try to fetch the real GAW-PO paper text
     (arxiv.org/pdf/2610.01511) for the exact reweighting formula, else use the pre-registered
     GAW-PO-lite fallback already written in PLAN.md; (b) write fetch_data.sh + requirements.txt
     (confirm model/dataset licenses + revision hashes); (c) implement baseline vanilla DPO first.
  2. Run the mandatory timed trial (ONE config, full train+eval+ARC cycle) BEFORE committing to the full
     12-run grid — this determines whether train/eval sizes need to shrink further.
  3. Keep radar.md fresh (daily refresh mandate) and promote/demote backlog items as new scouting lands.
- Blockers: none.
- Resume commands: none running long enough to need a checkpoint this cycle.
