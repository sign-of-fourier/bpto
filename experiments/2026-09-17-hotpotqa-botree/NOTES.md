# The all-evaluated tree (`botree`): BO chooses which node to expand, every child is measured ($4.5 over 4 runs)

User design (2026-09-17). After a cold start, every fully evaluated node - root, expanded parents, leaves - is a
candidate; a GP over the candidate's embedding predicts what its children will score; EI picks one; the reflector
writes 3 children and **all 3 get the full train evaluation**. No minibatch, no child screen, no gate. One
expansion = one GP update. BO only, no GEPA arm: the readout is held-out best-vs-root, the convergence trace
(`bo_fits`: parent, its score, μ/σ/EI, the children's scores, best so far) and the sibling ICC of child scores.

    python -m experiments.live_compare.hotpot --arms botree --seeds 3 --rollouts 3000 --n-train 25 --holdout 300 --program [--botree-target best] [--pca 4]
    python -m experiments.live_compare.hotpot --arms botree --seeds 6 --rollouts 6000 --n-train 50 --holdout 300 --program --pca 4

Rows per evaluation was deliberately cut to 25 (then 50): the batch size is the design choice, and many cheap
observations were the point. ~28 rollouts per child at 25 rows, ~55 at 50; ~36 expansions and ~115 evaluated
nodes per seed in every run. Warmup = GEPA's sampler for the first 4 expansions.

## Results (`table.md`, `curves.png`, `evaluations_*.jsonl`)

| variant | rows | seeds | train gain | **held gain** | ℓ | EI spread | top-node share | GP obs / inputs | sibling ICC |
|---|---|---|---|---|---|---|---|---|---|
| target = best child | 25 | 3 | +.068 | **−.014 ± .008** | 9.3 | .002 | 74% | 10 / 10 | .39 .29 .58 |
| target = every child (repeated obs.) | 25 | 3 | +.061 | **−.026 ± .018** | 10.2 | .006 | 28% | 111 / 14 | .33 .35 .11 |
| every child + PCA-4 | 25 | 3 | +.063 | **−.007 ± .017** | 2.9 | .051 | 57% | 110 / 8 | .00 .04 .00 |
| every child + PCA-4 | **50** | **6** | +.056 | **+.010 ± .011** | 7.9 | .010 | 70% | 112 / 6 | .58 .00 .46 .06 .14 .33 |

(ℓ = median fitted lengthscale after warmup; EI spread = max − min EI over the candidates; top-node share = fraction
of post-warmup expansions spent on the single most-expanded node; ICC = between-parent / total variance of child F1.)

## What each run taught

1. **`best`-child target locks on.** From round 18 EI picked the same depth-5 node 25 times in a row (69 children,
   none beating the .834 found at round 9). Its target - max over its children - cannot change unless a child
   beats the old max, and leaves have no target, so the training set froze at 9 observations and the argmax never
   moved. This is the descendant-value lock-on from the CLAUDE.md caveat, reproduced without a child screen.
2. **Every child as its own observation fixes the lock mechanically** (`BOSelector.value` may return a list;
   `DescendantValue(1, agg=list)`): 111 observations at 14 inputs, the GP updates every round, the search walks a
   chain (4-6 expansions per node, then moves). But EI was flat: fitted ℓ ≈ 8-13 against pairwise embedding
   distances ≤ 1.4, so k(x, x′) ≥ .994 for every pair and every candidate got the global mean.
3. **The flatness is geometry, not the target.** Titan V2 vectors are unit-norm in 256-d; the nearest neighbour of a
   prompt is 40% as far as the median prompt (distance concentration), and a single lengthscale cannot separate
   near from far. `BOSelector(pca=k)` - PCA fit per call on the pooled fit ∪ candidate rows - restores contrast
   (nearest/median .09 at k = 4, .15 at k = 8); the $0 refit of the saved trees took ℓ from 8.6 to 2.2 and EI
   spread up 10× (seed 1), and live it did the same in 2 of 3 seeds. It cannot manufacture a parent effect that
   is not there (seed 2 stayed flat; ICC .00 in all three PCA seeds at 25 rows).
4. **50 rows is where the pick starts to transfer.** 5 of 6 seeds up on held-out (+1.6 to +3.6 pts), one split
   (seed 1: root .663 train / .710 held) overfit the sample for −4.2. Within-parent SD fell from ~.05 to ~.025;
   ICC up to .58 on some seeds. The mechanism is mostly **best-of-n from root**: 4 of 6 winners are depth-1
   children, and the model spent 41-88% of expansions re-drawing from the node with the best child yield -
   which is what EI over repeated observations should do when no descendant beats the incumbent.

## Why the mutator's children sit below the parent

Root vs its worst and best child (seed 0, 25 rows): the reflector rewrites ~80% of the prompt per child
(`diff_ratio` ≈ .2) and its "fix" from 5 failure traces is usually "be precise, no extraneous paragraphs" -
which drops the two-hop hint that carries the tuned root and lowers selection recall (.82 → .66). F1 here is
recall-driven, so the reflector's reasoning pushes the wrong way. Every child is a draw from a distribution
centred ~1-5 pts below its parent with a tail above it; the tail is what best-of-n harvests.

## Comparison with GEPA (not paired)

12-seed gepa at 200 rows / 3,000 rollouts: held +.020 ± .004. botree at 50 rows / 6,000 rollouts: +.010 ± .011.
Welch t = −0.80, p = 0.45 - not distinguishable, and the old bo-pure arm (−.016 vs gepa, p = .01) it is not. A
paired gepa arm at 50 rows / 6,000 rollouts is the fair comparison and has not been run.

## Conclusions

- Selection-side machinery is now in a state that *can* use a signal: repeated observations per input, no lock-on,
  non-flat acquisition after PCA. What it lacks on this task is signal per observation: at 25 rows the parent-to-
  parent differences (~.02-.03) are below the measurement noise (~.03); at 50 rows they are about equal.
- **Dimensionality is a signal-to-noise problem at small n, not an expressiveness problem** (user). Few distinct
  inputs → PCA-k; rows ≫ dims → ARD; an isotropic kernel on raw 256-d embeddings is never sufficient by itself.
  Rule recorded in CLAUDE.md and the product backlog.
- Nova Micro at temperature 0: the same root on the same 25 rows scored .762 and .754 in two runs (2 rows flipped).
  At 25 rows one flipped row is 4 pts of F1.
