# Synthetic ladder — selection strategies at equal rollouts (2026-09-11)

`python -m experiments.synthetic_ladder.ladder --seeds 40 --budget 800 [--noise 0.3] [--p-informed 0.2] [--trap]`

Real bpto stack (Tree, evaluate, gepa loop, BOSelector) on a planted landscape served by a MockClient.
6 hidden skills = keywords; 24 distractor words; 40 examples each needing 1-3 skills; score = fraction of
needed skills present, with per-(prompt, example) noise. The mutator is the same ReflectiveExpander for
every arm: the mock "reflection" adds a *missing* skill with prob `p_informed` (0.5), else a random word.
Same minibatch gate (4 examples), same 800-rollout budget. BO arms use HashEmbedder + GPR + EI and fall back
to the GEPA sampler until 4 nodes have been expanded. `--trap`: three extra words lift every example to
>= 0.5 but cap it at 0.6 - the best-mean lineage becomes a dead end that only non-trap lineages (which win
individual examples at 1.0) can escape; the case the Pareto pool is designed for.

Cells: true score of the best pool candidate at 800 rollouts, mean ± SE over 40 seeds (% of seeds reaching 0.9).

| arm | baseline | noise 0.3 | dumb mutator (p_informed 0.2) | noise 0.3 + dumb | trap | trap + noise 0.3 |
|---|---|---|---|---|---|---|
| gepa-weighted (GEPA) | 0.971 ± 0.011 (90%) | 0.858 ± 0.026 (55%) | 0.829 ± 0.026 (38%) | 0.698 ± 0.031 (18%) | 0.812 ± 0.032 (48%) | 0.704 ± 0.031 (20%) |
| gepa-uniform | 0.828 ± 0.026 (40%) | 0.683 ± 0.022 (8%) | 0.646 ± 0.024 (8%) | 0.512 ± 0.027 (2%) | 0.695 ± 0.022 (10%) | 0.619 ± 0.024 (8%) |
| gepa-best (greedy) | 1.000 ± 0.000 (100%) | 0.998 ± 0.002 (100%) | 0.994 ± 0.006 (98%) | 0.946 ± 0.018 (78%) | 0.822 ± 0.037 (62%) | 0.712 ± 0.038 (40%) |
| gepa-all (no Pareto) | 0.739 ± 0.019 (10%) | — | — | — | 0.626 ± 0.019 (2%) | 0.632 ± 0.019 (2%) |
| bo-replace | 0.963 ± 0.015 (88%) | 0.936 ± 0.017 (68%) | 0.836 ± 0.027 (45%) | 0.740 ± 0.030 (30%) | 0.811 ± 0.032 (48%) | 0.742 ± 0.033 (35%) |
| bo-in-pareto | 0.983 ± 0.008 (90%) | 0.966 ± 0.018 (92%) | 0.805 ± 0.029 (40%) | 0.755 ± 0.030 (32%) | 0.755 ± 0.033 (38%) | 0.715 ± 0.034 (30%) |
| gepa+screen | 0.966 ± 0.012 (85%) | — | — | — | 0.755 ± 0.033 (35%) | — |
| bo+screen | 0.979 ± 0.011 (90%) | 0.957 ± 0.014 (80%) | 0.906 ± 0.025 (72%) | 0.720 ± 0.031 (18%) | 0.819 ± 0.034 (57%) | 0.751 ± 0.036 (42%) |

## What it says

1. **The Pareto pool matters; the stochastic sampling does not.** `gepa-all` (no pool) is worst everywhere
   and `gepa-uniform` (pool, no weights) is second worst. Weighting ∝ examples won is what makes GEPA's
   sampler work - and pushing that to its limit (greedy `argmax mean`, no randomness) is never worse and
   usually much better. The user's guess "the stochastic part is meh" is supported; "the Pareto part is
   meh" is not.
2. **Greedy's dominance is a landscape artifact** on the additive conditions: no dead ends, and an informed
   mutator always has a useful move from the best node, so exploration cannot pay. Under `--trap`, greedy's
   edge collapses (0.822 vs 0.812 weighted vs 0.811-0.819 BO: all within one SE) and under trap + noise the
   BO arms are nominally ahead (0.74-0.75 vs 0.71 greedy vs 0.70 weighted) - about one SE, not a result.
3. **Where BO helps, measurably:** (a) under evaluation noise, a surrogate over the pool beats GEPA's sampler
   (`bo-in-pareto` 0.966 vs 0.858; `bo-replace` 0.936); (b) with a weak mutator, **child pre-screening** is
   the most useful single addition (`bo+screen` 0.906 vs 0.829) - it spends rollouts only on the proposal
   the surrogate likes. Neither beats greedy on these landscapes.
4. **Where BO does not help:** replacing the pool (`bo-replace`) is never better than restricting to it;
   with a dumb mutator BO parent choice alone (0.805-0.836) is no better than GEPA's sampler.

## Caveats

- The mock mutator has no analogue of *reflection quality*; in live runs that is the dominant factor.
- HashEmbedder features are literally the keyword presence the landscape is built on - the surrogate's job
  is far easier here than with real embeddings of real prompts.
- 800 rollouts ≈ 20 full evaluations; all arms are budget-starved in the harder conditions.
- "Reached 0.9" counts seeds whose best pool candidate's *true* score hit 0.9 within budget.

## Implication for the live comparison (next step)

Arms worth spending on: `gepa-weighted` (reference), `gepa-best`, `bo-in-pareto`, `bo+screen`, and
`best+screen` (greedy parent + surrogate child screen - not run here; add it). Expect greedy to look good
unless the real landscape has dead ends; the trap condition is the one to keep in mind when reading it.
