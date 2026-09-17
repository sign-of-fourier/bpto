# `gepaei`: GEPA with BO in the "select to expand" seat only ($1.4 live + $0 ladder)

Two decisions in every loop are both called "selection" (user, 2026-09-17), and the old bo arm did both with two GPs:
1. **select to expand** - which pool member the reflector works on next (GEPA: Pareto-weighted sample; BO: EI);
2. **select to survive** - which proposal earns a full evaluation and joins the pool (GEPA: 5-row paired gate;
   old bo: a surrogate screen with no observations at the proposals' inputs, measured at the base rate).

`gepaei` is GEPA verbatim - 1 reflected child, 5-row minibatch, gate (child > parent on the same rows), full
200-row evaluation on pass - with **only (1) swapped for EI**. Same rows, rollouts, seeds and held-out sets as the
12-seed gepa run (`2026-09-17-hotpotqa-12-seeds/`), so the comparison is paired.

On HotpotQA the Pareto pool is the whole set of fully evaluated nodes (~11 per seed; with 200 per-row scores nobody
dominates anybody) and the per-row-win weights are nearly equal (162, 157, 157, 155, ...), so GEPA's sampler is
close to uniform there. Any selector with real signal should beat it.

**What the GP trains on.** The user dropped the 2026-09-16 rule that the surrogate never sees minibatch scores:
every proposal, gate pass or fail, gives one observation at the parent's input. Two targets were tried:

- **v1: the child's paired gain** over its parent on the child's rows - the gate's own number. Wrong: a weak parent
  is the easiest to improve on, so its children post the highest gains and EI expands the weakest node. Seed 5:
  the pool's lowest node (.622 own F1) expanded 41 times, 39 children, mean gain −.032, while the best (.704) got 2.
- **v2: the child's estimated full-train F1** = parent's full F1 + (child − parent on the child's rows). The paired
  difference cancels row difficulty (a gate failure is only ever scored on its 5 rows); adding the parent's full
  score makes it absolute and comparable to everything else.

Both with PIT targets, per-observation F1 SE as noise, `pca=4`, warmup = GEPA's sampler until every current
candidate (up to 4) has been expanded once.

    python -m experiments.live_compare.hotpot --arms gepaei --seeds 6 --program --pca 4 --out runs/hotpot_gepaei    # v1
    python -m experiments.live_compare.hotpot --arms gepaei --seeds 3 --program --pca 4 --out runs/hotpot_gepaei2   # v2

## Live results (paired with gepa, 200 rows, 3,000 rollouts)

| seed | gepa Δheld | v1 Δheld | v2 Δheld | gepa Δtrain | v1 Δtrain | v2 Δtrain | v2 most-expanded node's F1 rank in pool |
|---|---|---|---|---|---|---|---|
| 0 | .000 | .000 | .000 | .000 | .000 | .000 | 1/12 |
| 1 | +.016 | +.019 | +.006 | +.034 | +.022 | +.011 | 11/12 |
| 2 | +.017 | −.042 | −.001 | +.066 | +.021 | +.056 | 1/10 |
| 3 | +.018 | +.016 | — | +.052 | +.005 | — | |
| 4 | +.036 | +.008 | — | +.029 | +.010 | — | |
| 5 | +.033 | +.016 | — | +.038 | +.034 | — | |

- **v1 vs gepa: −.017 ± .010, paired t = −1.78, p = 0.14 (n = 6).** Behind on train in 5/6 seeds at an equal
  gate pass rate (~11 of 60-70 proposals): EI spent 26-76% of rounds on one node (uniform ≈ 9%), and on the wrong
  one for the reason above.
- **v2 vs gepa: −.009 ± .005, t = −1.85, p = 0.21 (n = 3).** The target fix removed the weak-parent bias (seed 2's
  most-expanded node is now the pool's best; −.042 → −.001) but not the deficit. EI spread was .000-.007 in every
  seed even with PCA and 50-130 observations: the incumbent is the pool's best node, every candidate's expected
  child is below it (the mutator beats its parent 13-16% of the time), so EI ≈ 0 everywhere and the ranking is
  on crumbs. Seed 1 spent 26% of its rounds on the 11th-best of 12.

Three versions of "BO picks the node to expand" on HotpotQA now sit at −.017, −.009 and −.010 (botree, unpaired)
against gepa; same sign each time, none significant.

## The $0 control: does the arm work where selection is known to matter?

Synthetic ladder (`experiments/synthetic_ladder/ladder.py --arms gepa-weighted gepa-uniform bo-replace gepaei`,
30 seeds, 800 rollouts; `ladder_tables.md`, `ladder_*_curves.png`). Here nodes span 0-1, the per-row-win weights
are far from uniform, and gepa-weighted beats gepa-uniform by 13 pts - the non-uniform case.

| arm | baseline (noise .1) | noise .3 |
|---|---|---|
| gepa-weighted | .962 ± .015 (80% reach .9) | .867 ± .024 (53%) |
| gepa-uniform | .830 ± .028 (33%) | .653 ± .020 (0%) |
| bo-replace (old design) | .939 ± .019 (73%) | .888 ± .025 (53%) |
| **gepaei v2** | **.947 ± .021 (80%)** | **.901 ± .026 (60%)** |

The arm is not broken: far above uniform, within one SE of GEPA's weighted sampler on the clean ladder, nominally
ahead under evaluation noise (+.034, ~1 SE; ~100 seeds would decide it at $0). Noise hurts gepa-weighted because
per-row wins go to whoever was lucky on that row; the GP averages observations and carries a noise term.

## Conclusions

- With the flatness fix in and 5× the observations, EI over the pool is still not ahead of near-uniform sampling
  on HotpotQA. The reason is now specific: when no parent's expected child beats the incumbent - the normal state
  with a 13-16% mutator - EI has nothing to rank. GEPA's sampler does not try to rank and spreads its proposals,
  which is the better policy for harvesting the mutator's tail.
- The selector is only as good as the signal it is given (user): 5-row gains (SD ≈ .2) and 25-50-row scores are
  the size of the parent-to-parent differences. The levers are all "buy more signal": rows per probe, children
  per parent, PCA k, seeds. None is a design change.
- The right live test is a landscape where the expand decision is *known* to be non-uniform - the ladder says the
  arm behaves there. On HotpotQA that means a weak root (10-15 pts of headroom, so pool members sit at different
  levels; the heritability analysis showed r ≈ .8 in that regime). Backlog.
- Vocabulary fixed in CLAUDE.md: "select to expand" vs "select to survive". Every BO arm since 2026-09-16 is in
  seat (1) only.
