# Experiments and findings

One directory per run (`YYYY-MM-DD-<task>-<model>/`) with NOTES.md, the final report, the scored tree and
a per-evaluation JSONL. Heavy artifacts (completion cache, tree.json checkpoints, event logs) stay in
`runs/` (gitignored). Datasets live with their task (`tasks/<name>/data/`).

## Research question

Compare **Bayesian optimization over a prompt tree** (bpto: embeddings + GPR + acquisition selecting
which node to expand, with ancestor attribution as the training target) against **GEPA-style search**
(evolutionary dynamics + a per-example Pareto pool as the sample-efficiency mechanism), with the same
mutation operators available to both - including feedback-aware rewrites, which are a user-space pattern
on top of bpto. The comparison is *selection strategy at equal rollout budget*.

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

| 2026-09-15/16 | [HotpotQA-distractor, GEPA vs BO (+ 0-shot MIPRO), 3 seeds x 3,000 rollouts, $6.5](2026-09-15-hotpotqa-phase1/) | **Flat.** After fixing a glued-template artefact (a doubled `{context}` was worth +2-9 pts held-out) and pinning evaluation to temperature 0: gepa Δheld -0.004 ± 0.003, bo -0.003 ± 0.003, best = root in 3/6 runs; 0-shot MIPRO 0/16 above root; hand variants within 3 pts of root. Nova Micro is nondeterministic even at temperature 0 (F1 differs on 6-11% of rows between identical evaluations). Stopped at the 25% pause both times. |

| 2026-09-16 | [HotpotQA two-stage program (searched selector, fixed answerer), GEPA vs BO, 3 seeds x 4,000 calls, $1.9](2026-09-16-hotpotqa-program/) | **The mechanism moves when the prompt controls a decision.** gepa raised held-out selection recall +6 pts in 2/3 seeds (first out-of-sample movement on HotpotQA); held-out F1 +.010 ± .019 (one +4.6, one 0, one −1.8). bo −.014 ± .006, 0/3, no smaller train-held gap (H3 falsified). In every run, the F1-chosen incumbent lost held-out iff its recall fell: selection ran on the noisier of two signals. Open: recall in the objective, program-valued nodes, surrogate noise. |

| 2026-09-16 | [Two-module synthetic ladder, $0](2026-09-16-synthetic-ladder-2mod/) | Gate for program-valued BO: `AdditiveGPR` (RBF per module, summed) + per-node noise + full-posterior EI reaches the target in 241 rollouts vs 340 for GEPA round-robin (both 100%), concat-RBF 0.854; holds with an interaction term (224 vs 249). **EI on the module component alone is a dead end (0.64)** - it ignores the modules the child keeps. Per-node noise helps the additive GP and hurts the single RBF. |
| 2026-09-16 | [HotpotQA two-module program (selector + answerer searched), GEPA round-robin vs additive-kernel BO, 3 seeds x 4,000 calls, $1.38](2026-09-16-hotpotqa-two-module/) | **Searching the answerer on Micro is wasted budget.** gepa held ΔF1 −.009 ± .009 (worse than its selector-only +.010; recall no longer moved), bo +.001 ± .012 with one clean win (seed 1: +2.4 F1, +4.7 recall). 0/6 incumbents carry a substantive answerer rewrite. Recombination = best-found in 1/3. Diagnostics exposed a bo-arm defect present in every earlier live bo arm: EI's incumbent was the max raw target, so one 5-row 1.0 froze parent selection (seed 0: one parent 291/345 times). Fixed after the run; bo rows are pre-fix. |

| 2026-09-16 | [HotpotQA selector program, BO without the minibatch, 3 seeds, $0.80](2026-09-16-hotpotqa-bo-pure/) | **First bo arm not below zero.** GP trains on full evaluations only, PIT targets, EI incumbent = best posterior expectation, parent value = own score (the descendant target locked EI onto one parent: ladder 0.50 vs 0.995). Held ΔF1 +.019 ± .016 (gepa same seeds +.010 ± .019), held selection recall up in 3/3 seeds (+.029 ± .008 vs gepa ± .035). Ladder: beats gepa-weighted at baseline (450 vs 483 rollouts), under noise (0.887 vs 0.836) and with a weak mutator (1.000 vs 0.836). |

| 2026-09-17 | [HotpotQA selector program, GEPA vs bo-pure, 12 seeds x 3,000 rollouts, $5.33](2026-09-17-hotpotqa-12-seeds/) | **The 3-seed bo-pure lead does not replicate.** gepa held +.020 ± .004, bo-pure +.004 ± .004; paired −.016 ± .007, p = .04, gepa ahead 9/12. The surrogate child screen picks children that beat the parent 27% of the time = the mutator's base rate; GEPA's 5-row gate lets through children that beat the parent on the full set 51% of the time and sees twice the proposals. |
| 2026-09-17 | [Heritability of the landscape, parent-offspring regression over 3 tree families, $0](2026-09-17-heritability/) | HotpotQA F1 pooled r ≈ .8 is "broken begets broken": among parents ≥ root, r ≈ .05 (gated) / .5 (bo-pure), P(child ≥ root) .5-.8, mean child 1 pt below its parent, no lineage climbing beyond depth 2. **Level-heritable, stacking not demonstrated.** Compression F1 not heritable (r .3 / .06). Noise floor .01 at 200 rows. |
| 2026-09-17 | [All-evaluated tree (`botree`), 4 runs, 15 seeds, $4.5](2026-09-17-hotpotqa-botree/) | Best-child target locks on (fit frozen at 9 obs). Every-child-as-observation (`value` → list) breaks the lock; EI then flat because 256-d Titan distances concentrate (nearest/median .40, ℓ ≈ 10 vs distances ≤ 1.4). **`BOSelector(pca=4)` restores contrast** (ℓ → 1-2, EI spread 10×). At 25 rows held −.014/−.026/−.007; at **50 rows, 6 seeds: +.010 ± .011**, 5/6 up, winners are mostly root's best-of-n children. Sibling ICC .0-.58 by seed. |
| 2026-09-17 | [`gepaei`: GEPA with EI in the expand seat only, 6 + 3 seeds paired with gepa, $1.4; ladder control $0](2026-09-17-hotpotqa-gepaei/) | Gain-over-parent target expands the weakest node (v1 −.017 ± .010 vs gepa); estimated-child-F1 target fixes that, still −.009 ± .005 (n = 3). EI ≈ 0 everywhere when no parent's expected child beats the incumbent. **Ladder: gepaei .947 vs gepa-weighted .962 vs uniform .830; under noise .901 vs .867** - the arm works where selection is known to matter; on HotpotQA the sampler is already ≈ uniform-optimal. |

## Standing conclusions

- **Two selections, two seats (2026-09-17).** "Select to expand" (which pool member the reflector works on) and
  "select to survive" (which proposal earns a full evaluation) are different decisions. A surrogate in the survive seat
  with no observations at the proposals' inputs is at the mutator's base rate; GEPA's paired 5-row gate is the better
  cheap filter there (51% vs 27%). BO's seat is expand. On HotpotQA GEPA's expand sampler is ≈ uniform (equal per-row
  wins across a ~11-node pool) and three EI variants tie or trail it; on the ladder, where the pool spans 0-1, EI ≈
  the weighted sampler and is nominally ahead under noise. The selector is only as good as the signal per observation.
- **Embedding dimensionality is a signal-to-noise problem at small n.** Raw 256-d Titan vectors: every prompt about
  equally far from every other, a single-lengthscale GP goes flat with tens of inputs. `BOSelector(pca=k)`, k ≈ 4-8,
  refit per call on fit ∪ candidate rows, restores contrast; ARD when rows ≫ dims; an isotropic kernel on raw
  embeddings is never sufficient on its own. Product rule (BACKLOG "Product (2.0)").
- **Repeated observations at one input** (`value` returning a list: every child's score at the parent's embedding)
  is the supported way to learn a node's expected yield and spread; a scalar `max` over descendants locks the search
  onto one parent. Rows per evaluation: 25 cannot rank the top on HotpotQA (train winners lose held-out); 50 is
  where the pick starts to transfer (+.010 ± .011, 6 seeds).

- **Heritability is a property of the landscape, not of optimisers.** Genetic search (Pareto pool + mutate the best,
  GEPA) wins where a good node's children are good - the synthetic ladder is built that way, and greedy "expand the best"
  is optimal there. That property is an assumption, not a given, for prompt and program optimisation, and may be the
  uncommon case; it is also easy to measure on any run (parent-offspring regression of full-evaluation scores). The
  descendant-value BO target exists for landscapes where it fails, and the ladder cannot test it (2026-09-16).
- **Batch size is the design decision; "minibatch" is not a free lunch.** The full train set is not automatically the
  right evaluation size when the goal is fewest calls; the right size is found by heuristics for now (downsample if 200
  is too expensive per node, and evaluate *everything* at that size). A minibatch that is too small (5 rows on F1:
  SE ≈ .2) causes problems rather than solving them: near-random accept/reject in GEPA's gate, and a surrogate that must
  never see such scores as targets. GEPA pays ~20% of its budget on 5-row coin flips; the bo arm now has no minibatch.

- IFBench is a flat landscape for prompt search at every model size published (GEPA: +1.7 on Qwen3-8B,
  +8 on GPT-4.1 Mini; MIPROv2 ~0). With 300 rows, a +2 pt effect is undetectable. Use it as a plumbing /
  flat-landscape stress test, not as the benchmark that decides between BO and GEPA-style selection.
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
- HotpotQA-distractor on Nova Micro is flat *as a single prompt*: root F1 ~0.72, six hand-written variants within 3 pts,
  reflective search (gepa, bo) and 0-shot MIPRO all tie it on held-out. As a two-stage program (searched paragraph
  selector, fixed answerer) the same reflector moves held-out selection recall +6 pts: the flatness was the task, not
  the mutator. Pick tasks where the searched prompt controls a decision a later stage consumes, or a tradeoff. Always run the $0.15 hand-variant pilot
  (at temperature 0) and the 25% pause before a full comparison.
- All live runs before 2026-09-16 evaluated at Nova's default temperature 0.7 (nothing was sent). Now pinned to 0 -
  but Bedrock Nova Micro is still nondeterministic at 0 (F1 differs on 6-11% of rows between identical evaluations,
  up to 3.7 pts on 200 rows). One measurement per candidate cannot resolve effects below ~3 pts: re-measure
  incumbents or give the surrogate per-node SE.
- Mutator hygiene matters more than selection on these tasks: a template that rendered `{context}` twice (Nova Lite
  glued two templates) beat every genuine instruction change. The expander now requires each placeholder exactly once.
- Reference points from the GEPA paper (Agrawal et al. 2025), Qwen3-8B, test accuracy %:
  HotpotQA 42.3 -> 62.3, IFBench 36.9 -> 38.6, HoVer 35.3 -> 52.3, PUPA 80.8 -> 91.9 (GEPA, <= 7k rollouts);
  MIPROv2 (BO-family): 55.3 / 36.2 / 47.3 / 81.6. Tasks with headroom: HotpotQA, HoVer, PUPA.
- BO has not yet shown noise-robustness live (program run: 0/3, gap no smaller than gepa's). Its GP is fit on single
  noisy measurements with `noise=None`; the synthetic-ladder noise result assumed a known noise level. Fix before the next
  BO vs GEPA claim on a noisy metric. bpto nodes are single prompts; GEPA's are programs - a multi-module comparison needs
  program-valued nodes and a per-module (concatenated / additive-kernel) embedding, untested.
