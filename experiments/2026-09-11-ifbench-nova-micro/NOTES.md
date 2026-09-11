# IFBench on Nova Micro — first live search (2026-09-11)

Command (two invocations: the first hit the call cap during the held-out pass, the second resumed):

    python -m tasks.ifbench.run --provider bedrock --n-train 40 --cheap-n 10 --rounds 2 --holdout 40 --out runs/ifbench_nova

Model `us.amazon.nova-micro-v1:0`, max_tokens 1024, no schema. Schedule: round 0 `random(3)` on root then
`guided(2)` on each random child; round 1 same on `top_k(2, among=unexpanded)`; successive halving 10 -> 40.
Selection was greedy `top_k` (no BOSelector). Seed 0 split: 40 train / 260 held-out (40 scored).

| | train (40) | held-out (40) |
|---|---|---|
| root `strict` | 0.425 | 0.175 |
| best `strict` | 0.450 | 0.250 |
| root `inst_strict` | 0.4625 | 0.237 |
| best `inst_strict` | 0.4875 | 0.300 |

Cost: 802 calls (722 search + 80 held-out), ~110k input / ~105k output tokens, ~2-3 cents.
Tree: 26 nodes, depth 4, 15 nodes evaluated on the full 40, 11 dropped at the 10-example rung.

Observations
- Best node is a depth-1 `random` restatement (97 vs 104 template tokens). Depths 2-4 (guided rewrites,
  ~500 calls) produced nothing better: flat landscape at this model size.
- +7.5 pts held-out is inside noise (SE of a proportion at n=40, p~0.2 is +-6.3 pts per number).
- Root train/held-out gap (0.425 vs 0.175) is sample composition (constraint-type mix), not overfitting.
- Bug found: `Budget` overshot under concurrency (722 calls with max_calls=500) - fixed in c0956ce.
- Call count for these flags ~ (1 + 9 * expanded_parents) * cheap_n + survivors * n_train; `guided(n)`
  runs on *each* random child so one expansion is n_random * (1 + n_guided) nodes.

Files: report.txt (final report), tree.txt (tree with scores), evaluations.jsonl (one row per evaluation
event with metrics), tree.png. Full cache/tree.json/events are in runs/ (not committed).
