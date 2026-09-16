# HotpotQA two-module program (selector + answerer both searched): GEPA round-robin vs BO with an additive kernel

3 seeds x 2 arms x 4,000 task-model calls, **$1.38** (estimate was $2.2). Plan: `PLAN.md` at commit time; gate:
`../2026-09-16-synthetic-ladder-2mod/`. Follows `../2026-09-16-hotpotqa-program/` (selector searched, answerer fixed),
where the fixed answerer looked like the bottleneck once selection recall rose.

**What is new.** Every node is a `Program` {selector, answerer}. Mutation is GEPA's: round-robin (module r mod 2),
the reflector rewrites one module from that stage's own traces (`ExampleResult.trace["answerer"]` holds what the
answerer saw and said), child = parent with one module swapped - identical in both arms. The bo arm's surrogate is
`AdditiveGPR`: one Titan V2 embedding per module, kernel = RBF(selector) + RBF(answerer), per-node noise = F1 SE
(minibatch-5 nodes ~.2, full-200 nodes ~.03), EI on the full posterior for parents (3 warm-up parents from the Pareto
sampler), surrogate screen 3 -> 1 for children. At the end the bo arm evaluates the additive model's *recombination*
(per-module argmax of posterior mean) on train and held-out.

Same seeds, splits (train 200 / held 300), models (Micro at temperature 0 for rollouts, Lite at 1.0 for reflection),
minibatch 5, gate `beats_parent`, per-run `Budget(max_calls=6,650)` as the previous run plus the recombination readout.

Pre-registered: **H1** gepa-2mod held ΔF1 > gepa-1mod (+.046/−.018/+.001) in ≥ 2/3 seeds; **H2** bo held ΔF1 ≥ 0 in ≥ 2/3
and mean ≥ gepa-2mod; **H3** recombination ≥ best-found on train in ≥ 2/3; oracle 0.780 stays the ceiling.

## Result

| arm | seed | train F1 | train recall | held F1 | held sel-recall | train-held gap | accepted (sel/ans) | incumbent's lineage |
|---|---|---|---|---|---|---|---|---|
| gepa | 0 | .743→.762 | .858→.877 | .696→.698 (+.002) | .832→.848 (+.017) | +.017 | 15/96 (8/7) | selector, answerer |
| gepa | 1 | .704→.717 | .850→.828 | .704→.676 (**−.028**) | .837→.820 (−.017) | +.041 | 15/66 (8/7) | selector |
| gepa | 2 | .645→.668 | .800→.785 | .739→.737 (−.002) | .857→.847 (−.010) | +.025 | 16/50 (8/8) | selector |
| bo | 0 | .756→.758 | .882→.882 | .702→.692 (−.009) | .835→.835 (+.000) | +.011 | 15/116 (4/11) | answerer |
| bo | 1 | .705→.744 | .840→.870 | .697→.721 (**+.024**) | .827→.873 (**+.047**) | +.015 | 14/83 (8/6) | selector, selector |
| bo | 2 | .643→.710 | .802→.830 | .731→.720 (−.011) | .862→.882 (+.020) | **+.078** | 14/74 (7/7) | selector |

gepa: held ΔF1 **−.009 ± .009**, held Δrecall −.003 ± .010. bo: held ΔF1 **+.001 ± .012**, held Δrecall +.022 ± .014.
Previous run (selector only): gepa +.010 ± .019, bo −.014 ± .006. `trajectory_f1.png`, `trajectory_recall.png`,
`best_prompts.md` (both modules of every incumbent, plus bo's recombinations).

- **H1: no (0/3).** Letting GEPA also rewrite the answerer did not convert recall into F1; it did worse than the
  selector-only run in all three seeds, and held-out recall no longer moved (+.017 / −.017 / −.010 vs +.063 / −.045 / +.058).
  With half the rounds spent on the answerer, the selector got ~50 reflections instead of ~100; the accepted answerer
  children (7-8 per seed) rarely survive into the incumbent - the winning answerer is the root answerer verbatim in 4/6
  runs and a paraphrase in the other two (`best_prompts.md`). On Micro the answerer prompt has no headroom the
  reflector can find in 200 chars; the answerer being "the bottleneck" was the wrong reading of seed 2 last time.
- **H2: no (1/3), mean tie.** bo seed 1 is the one clean win of the six runs (held F1 +2.4, recall +4.7, both
  selector rewrites); seeds 0 and 2 lost held-out. bo's train gain (+.036) is twice gepa's, but seed 2's train-held gap of
  +.078 says most of it was fitting the 200-row noise (Micro is ±3 pts per evaluation even at temperature 0).
- **H3: 1/3.** Seed 1's recombination *is* the best-found program (the model's per-module argmaxes coincide with it);
  in seeds 0 and 2 the recombination scored .665 / .648 on train against best-found .758 / .710 - the additive model
  predicted a better program than the one it found, and was wrong. Either the modules interact (the selector's output
  is the answerer's input, so this is plausible) or the surrogate was not fit well enough to attribute - see next point.
- **Oracle holds.** Best held-out .737 < .780. No placeholder / tag artefacts in any incumbent.

## A defect in the bo arm, found in the diagnostics

`tree.meta["bo_fits"]` shows seed 0's parent surrogate trained on **4 nodes for all 113 fits**, with a selector lengthscale
of 1e-8: BO picked the same parent 291 times out of 345 proposals. Seeds 1-2 grew to 12-13 training nodes but were also
concentrated (124/243 and 102/220 proposals from one parent). Cause: EI's incumbent was `max(y)` over the training
targets, and the targets include minibatch-5 children - one child scoring 1.0 on five rows makes every EI ≈ 0, ties go
to the first candidate, and the parent set never changes. Per-node noise enters the GP but not the incumbent, so it did
not help; the 1e-8 lengthscale is the median-distance heuristic on four nodes that share one selector prompt.

Fixed after this run (`bpto/bo/__init__.py`, `gpr.py`, test `test_incumbent_is_posterior_mean_not_a_lucky_observation`):
the incumbent is now the best *posterior mean* at the training points (the standard plug-in for noisy BO) and a
degenerate median falls back to unit scale. The ladder gate was re-run with the fix (see the ladder notes). **The bo
numbers above are with the defect**; the same rule was in force in every earlier live bo arm (`bo_fits` was not
inspected before). The gepa rows are unaffected.

## What to take from it

1. Searching the answerer on Micro is wasted budget: 0/6 incumbents got a substantive answerer rewrite, and the rounds it
   took cost GEPA its recall gains. If a second module is to be searched here it has to be one with headroom - or the
   round-robin has to be replaced by adaptive module choice (EI over modules), which the additive model makes cheap.
2. The two-module machinery works end to end (program nodes, per-module traces and feedback, additive surrogate,
   recombination) and is a $0 change for single-module tasks (default `--modules selector`).
3. No BO-vs-GEPA claim from this run: bo's arm ran with the incumbent defect, and 3 seeds cannot resolve a 1-pt mean on a
   metric with 3-pt evaluation noise. A re-run of the bo arm alone with the fix costs ~$1.4 (3 × $0.46); worth it only
   with a module pair that has headroom in both modules.

## Files

`results.jsonl` (per arm-seed: curve, all candidates, both modules of the best program, `accepted_by_module`,
`recombination`, `bo_fits`), `trajectory_f1.png`, `trajectory_recall.png`, `best_prompts.md`. Heavy artefacts in
`runs/hotpot_program2/` (gitignored). Harness: `experiments/live_compare/hotpot.py --program --modules selector,answerer`.
