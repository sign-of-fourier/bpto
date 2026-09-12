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

**Rollout** = one unique (prompt, example) model call, i.e. `client.usage.calls` with the cache on. The GEPA
arm reads its reflection minibatch from the parent's cached evaluation, so it is not charged twice for it.

## Findings so far

| date | what | result |
|---|---|---|
| 2026-09-10 | Synthetic BO benchmark (`tests/test_bo.py` setup, 20 seeds, HashEmbedder + GPR + EI) | BO finds the planted optimum in 1.45 rounds vs 7.35 for random selection. Synthetic landscape only. |
| 2026-09-11 | Bedrock Nova Micro smoke (`examples/smoke.py`, 10 calls, names task) | Provider path works; Nova echoes JSON schemas -> client-side balanced-brace extraction. |
| 2026-09-11 | [IFBench, Nova Micro, greedy top_k, 802 calls](2026-09-11-ifbench-nova-micro/) | strict 0.175 -> 0.250 held-out (n=40, inside noise). Best node is a depth-1 restatement; deeper expansion added nothing. |
| 2026-09-11 | [Synthetic ladder, 8 arms x 40 seeds x 6 conditions, $0](2026-09-11-synthetic-ladder/) | Pareto pool matters, stochastic sampling does not: greedy argmax within the pool beats GEPA's sampler everywhere except a deceptive landscape, where all tie. BO beats GEPA's sampler under noise (0.97 vs 0.86) and surrogate child pre-screening helps most with a weak mutator (0.91 vs 0.83); nothing beats greedy on additive landscapes. |

| 2026-09-11 | [IFBench, GEPA vs BO, seed 0, 1,200 rollouts each, $0.16](2026-09-11-ifbench-gepa-vs-bo/) | Tie by construction: neither arm found a child beating the root on train (Lite reflections are paraphrases; Micro has no headroom on IFBench). Stopped before seeds 1-4. Several expander/budget bugs fixed. |
| 2026-09-11 | [Compression, GEPA vs BO, annealed F1 floor, 5 seeds x 600 rollouts, $0.065](2026-09-11-compression-gepa-vs-bo/) | Both arms compress 97 -> 5-26 tokens within 0.05 train F1 of the root. BO shorter on average (11.4 vs 17.2 tokens, paired +5.8 ± 5.1, 2/5 wins) but with lower held-out F1 (0.936 vs 0.958): not significant, and the shortest-feasible readout rewards overfitting a 30-example floor. Mechanics (floor moves, re-scoring, "new best" when the floor crosses) verified live. |
| 2026-09-12 | [Compression v2, 12 seeds x 2,000 rollouts, train 100, minibatch 5, $0.41](2026-09-12-compression-v2-gepa-vs-bo/) | **Readout is the front graph.** BO's pooled (tokens, F1) front lies left of GEPA's at every accuracy level; per seed, at the accuracy the floor targeted (root - 0.05..0.10) BO is 3.5 tokens shorter (±2.0, 8-9/12 seeds). At stricter bars GEPA's pool hedges better because BO's value ignores accurate-but-not-shorter children. BO wasted 35% of rollouts on non-advancing full evals vs GEPA's 49%. |

## Standing conclusions

- IFBench is a flat landscape for prompt search at every model size published (GEPA: +1.7 on Qwen3-8B,
  +8 on GPT-4.1 Mini; MIPROv2 ~0). With 300 rows, a +2 pt effect is undetectable. Use it as a plumbing /
  flat-landscape stress test, not as the benchmark that decides between BO and GEPA-style selection.
- No live run has yet exercised `BOSelector` or a real embedder; all BO evidence is synthetic.
- Live: the reflector's willingness to make substantive edits is the binding constraint on every selection
  strategy. Nova Lite paraphrases; test a forced-structure prompt / Pro before comparing selectors again.
- Synthetic ladder: the *weighting* in GEPA's sampler is what works; its randomness is not. BO's clearest
  contributions are robustness to evaluation noise and child pre-screening when the mutator is weak.
- Compression maiden run: with a 30-example train set and a permissive 3-example gate, ~95% of rollouts go to
  full evaluations of accepted children, leaving ~20 parent choices per run - too few for selection strategy
  to show. Make selection expensive (bigger train set or stricter gate) before comparing selectors again, and
  compare (tokens, held-out F1) fronts rather than the shortest train-feasible prompt.
- Compression v2 (first result where selection had real decisions): acquisition + surrogate child screen beat
  Pareto-weighted sampling at the targeted accuracy; a threshold-tied scalar value leaves the rest of the
  front to chance, which the Pareto pool covers for free. Next BO value: front/hypervolume gain. Report
  constrained comparisons as the full front graph, never one row. Caveat: BO makes ~3x the reflector
  calls (3 children/round vs 1); the `gepa + 3 children, random keep-1` control is not yet run.
- Gains reported on low-baseline models (Nova Micro) do not transfer proportionally to stronger models:
  a restatement fixes "cheap" failures that an 8B model has already absorbed.
- Reference points from the GEPA paper (Agrawal et al. 2025), Qwen3-8B, test accuracy %:
  HotpotQA 42.3 -> 62.3, IFBench 36.9 -> 38.6, HoVer 35.3 -> 52.3, PUPA 80.8 -> 91.9 (GEPA, <= 7k rollouts);
  MIPROv2 (BO-family): 55.3 / 36.2 / 47.3 / 81.6. Tasks with headroom: HotpotQA, HoVer, PUPA.
