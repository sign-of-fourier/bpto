# Two-module synthetic ladder: additive kernel vs concatenation vs GEPA round-robin ($0)

Gate for the two-module HotpotQA run (`PLAN.md`): does a GP with an additive per-module kernel select at least as well
as GEPA's Pareto sampler when the node is a two-module program, and better than the naive single RBF over the
concatenated embeddings?

`python -m experiments.synthetic_ladder.ladder --modules 2 [--interaction] [--no-noise] --seeds 20 --budget 800`.
Node = Program {A, B}; skills 0-2 count only in A, 3-5 only in B; round-robin mutation (module r mod 2, one module per
child, mock reflector adds a missing skill with p 0.5); per-module feedback names only that module's misses. All arms:
minibatch-4 gate, 800 rollouts, HashEmbedder(128) per module. BO arms: propose 3 / surrogate keeps 1, GEPA sampler
until 4 expanded nodes, and per-node noise (`metrics_std / sqrt(n)`) unless `--no-noise`.
`--interaction`: +0.25 when A has skill0 and B has skill3 (a term the additive model cannot represent).

## Result (n = 20 seeds; true score of the best pool candidate at 800 rollouts)

| condition | arm | final (mean ± SE) | reached 0.9 | median rollouts to 0.9 |
|---|---|---|---|---|
| plain / noise | gepa-rr | 1.000 ± 0.000 | 100% | 340 |
| plain / noise | **bo-additive+screen** | **1.000 ± 0.000** | 100% | **241** |
| plain / noise | bo-concat+screen | 0.854 ± 0.049 | 65% | 279 |
| plain / noise | bo-additive-comp+screen | 0.641 ± 0.032 | 5% | 495 |
| plain / no-noise | bo-additive+screen | 0.967 ± 0.021 | 85% | 292 |
| plain / no-noise | bo-concat+screen | 1.000 ± 0.000 | 100% | 331 |
| plain / no-noise | bo-additive-comp+screen | 0.645 ± 0.035 | 10% | 370 |
| interaction / noise | gepa-rr | 1.000 ± 0.000 | 100% | 249 |
| interaction / noise | **bo-additive+screen** | **1.000 ± 0.000** | 100% | **224** |
| interaction / noise | bo-concat+screen | 0.824 ± 0.046 | 50% | 236 |

`curves.png`: anytime curves (left: plain, with the no-noise BO arms dashed; right: interaction).

Re-run after the incumbent fix found in the live run (EI incumbent = best posterior mean at the training points instead
of the best raw observation; degenerate median lengthscale guarded), plain / noise, 20 seeds:

| arm | final | reached 0.9 | median rollouts |
|---|---|---|---|
| bo-additive+screen (fixed) | 0.996 ± 0.004 | 100% | 290 |
| bo-concat+screen (fixed) | 0.688 ± 0.055 | 30% | 226 |
| single-module ladder, bo+screen (fixed; 2026-09-11 baseline 0.979, 441) | 0.964 ± 0.020 | 85% | 429 |

Gate still passes (additive ≥ gepa-rr's 1.000 / 340 within SE and well above concat); the fix changes nothing on this
landscape because its minibatch scores rarely produce a fluke incumbent - the live run's did.

1. **Gate passed** for the live configuration (additive kernel, full-posterior EI, per-node noise): it reaches the
   target 30% sooner than gepa-rr (241 vs 340 rollouts) with every seed at 1.0, and beats the concatenated-RBF ablation
   (0.854, 65%). The interaction term does not break it (224 vs 249; concat 0.824).
2. **Parent EI on the module component alone is a dead end** (0.64 with or without noise). The plan's argument - "the
   other modules' terms cancel between parent and child" - is about the *improvement*, but selection has to maximise the
   *child's value*, which includes the modules the child keeps. Component-only EI happily picks a parent whose module m
   looks promising while the rest of its program is bad. Full-posterior EI (as in the single-prompt arm) is now the
   default; `--parent-rank module` / arm `-comp` remain for the record.
3. **Noise × kernel interaction.** Per-node noise helps the additive GP (0.967 -> 1.000, 292 -> 241) and hurts the
   concatenated one (1.000 -> 0.854). With 128-d hash embeddings × 2 modules the single RBF has one lengthscale for a
   space where only a few coordinates matter; adding the (large, minibatch-4) per-point variances leaves it close to its
   prior, and EI over a flat posterior is roughly uniform selection - which the single-module ladder already showed is
   worse than the Pareto sampler (gepa-all 0.74). The additive kernel has one lengthscale per module and evidently
   keeps enough signal. n = 20 per cell; treat the no-noise additive shortfall (3 seeds < 0.9) as within noise.
4. This ladder is easy for GEPA (100% in every condition): each module has three skills and the feedback names the
   missing one. It cannot show a *final-score* advantage for anything; the readout is rollouts-to-target.

## Files

`table.md`, `curves.png`, `curves.json` (anytime curves per condition/arm; pre-fix runs). Code: `experiments/synthetic_ladder/ladder.py`
(`--modules 2`, `--interaction`, `--no-noise`; arms `gepa-rr`, `bo-concat+screen`, `bo-additive+screen`,
`bo-additive-comp+screen`).
