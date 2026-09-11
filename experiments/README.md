# Experiments and findings

One directory per run (`YYYY-MM-DD-<task>-<model>/`) with NOTES.md, the final report, the scored tree and
a per-evaluation JSONL. Heavy artifacts (completion cache, tree.json checkpoints, event logs) stay in
`runs/` (gitignored). Datasets live with their task (`tasks/<name>/data/`).

## Research question

Compare **Bayesian optimization over a prompt tree** (bpto: embeddings + GPR + acquisition selecting
which node to expand, with ancestor attribution as the training target) against **GEPA-style search**
(evolutionary dynamics + a per-example Pareto pool as the sample-efficiency mechanism), with the same
mutation operators available to both - including feedback-aware rewrites, which are a user-space pattern
on top of bpto (see BACKLOG.md). The comparison is *selection strategy at equal rollout budget*.

## Findings so far

| date | what | result |
|---|---|---|
| 2026-09-10 | Synthetic BO benchmark (`tests/test_bo.py` setup, 20 seeds, HashEmbedder + GPR + EI) | BO finds the planted optimum in 1.45 rounds vs 7.35 for random selection. Synthetic landscape only. |
| 2026-09-11 | Bedrock Nova Micro smoke (`examples/smoke.py`, 10 calls, names task) | Provider path works; Nova echoes JSON schemas -> client-side balanced-brace extraction. |
| 2026-09-11 | [IFBench, Nova Micro, greedy top_k, 802 calls](2026-09-11-ifbench-nova-micro/) | strict 0.175 -> 0.250 held-out (n=40, inside noise). Best node is a depth-1 restatement; deeper expansion added nothing. |

## Standing conclusions

- IFBench is a flat landscape for prompt search at every model size published (GEPA: +1.7 on Qwen3-8B,
  +8 on GPT-4.1 Mini; MIPROv2 ~0). With 300 rows, a +2 pt effect is undetectable. Use it as a plumbing /
  flat-landscape stress test, not as the benchmark that decides between BO and GEPA-style selection.
- No live run has yet exercised `BOSelector` or a real embedder; all BO evidence is synthetic.
- Gains reported on low-baseline models (Nova Micro) do not transfer proportionally to stronger models:
  a restatement fixes "cheap" failures that an 8B model has already absorbed.
- Reference points from the GEPA paper (Agrawal et al. 2025), Qwen3-8B, test accuracy %:
  HotpotQA 42.3 -> 62.3, IFBench 36.9 -> 38.6, HoVer 35.3 -> 52.3, PUPA 80.8 -> 91.9 (GEPA, <= 7k rollouts);
  MIPROv2 (BO-family): 55.3 / 36.2 / 47.3 / 81.6. Tasks with headroom: HotpotQA, HoVer, PUPA.
