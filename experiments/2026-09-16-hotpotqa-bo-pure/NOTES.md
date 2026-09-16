# HotpotQA selector program, BO without the minibatch: own-score EI, PIT targets, full evaluations only

bo arm only, 3 seeds x 4,000 task-model calls, **$0.80**, ~6.5 min per seed. Compared against the gepa rows of
`../2026-09-16-hotpotqa-program/` (same seeds, train/held splits, models, temperatures, budget). Follows the diagnosis in
`../2026-09-16-hotpotqa-two-module/NOTES.md` (parent selection frozen by a minibatch fluke) and the user's decision:
the surrogate must never train on minibatch scores, and the bo arm should not run GEPA's 5-row gate as a second filter.

**The arm.** Each round: parent by EI over the Pareto candidates (GEPA sampler for the first 4 rounds); reflector proposes
3 selectors from the parent's own traces; the child surrogate keeps 1; that child is evaluated on the full 200-row train
set. No minibatch anywhere. Both GPs train on full evaluations only. Targets are PIT-transformed (rank -> Φ⁻¹, `PIT` in
`bpto/bo/gpr.py`), per-node noise = F1 SE carried into the transformed scale, EI incumbent = best posterior expectation
at the training inputs. `--bo-gate none --bo-transform pit` (defaults now).

**Parent value = the candidate's own full-train F1**, not its best child's. On the ladder (below) the descendant target
locked EI onto one parent (its posterior σ made its own EI larger than any untried node's): 0.50 vs 0.995. This departs
from the "ancestor attribution is the BO target" invariant in CLAUDE.md for parent selection in this schedule; the
invariant still describes `DescendantValue`/`SubtreeValue`, which remain available - it just is not what the bo arm uses.

## Ladder gate (`experiments/synthetic_ladder/ladder.py --arms bo-pure-own-ei`, 20 seeds x 800 rollouts)

| condition | bo-pure (own, EI) | bo-pure (best child, EI) | gepa-weighted |
|---|---|---|---|
| baseline | **0.995**, median 450 | 0.499, 0% | 0.964, 483 |
| noise 0.3 | **0.887** | - | 0.836 |
| weak mutator (p_informed 0.2) | **1.000**, 512 | - | 0.836 |

UCB and Thompson with the own-score value also reach 1.000 (492); Thompson with the descendant value only 0.848.

## Live result

| arm | seed | train F1 | held F1 | held sel-recall | train-held gap | full evals | best depth |
|---|---|---|---|---|---|---|---|
| bo-pure | 0 | .745→.762 | .683→.727 (**+.044**) | .830→.872 (**+.042**) | **−.026** | 17 | 5 |
| bo-pure | 1 | .708→.731 | .705→.694 (−.011) | .825→.838 (+.013) | +.034 | 16 | 3 |
| bo-pure | 2 | .653→.702 | .721→.746 (**+.025**) | .853→.885 (**+.032**) | +.025 | 17 | 6 |
| gepa | 0 | .732→.746 | .694→.740 (+.046) | .832→.895 (+.063) | −.032 | 15 | - |
| gepa | 1 | .713→.736 | .692→.674 (−.018) | .827→.782 (−.045) | +.041 | 14 | - |
| gepa | 2 | .662→.699 | .741→.742 (+.001) | .863→.922 (+.058) | +.035 | 15 | - |
| bo (old, minibatch-trained GP) | 0-2 | | −.003 / −.017 / −.022 | +.007 / −.020 / −.047 | | 14-15 | |

| | held ΔF1 | held Δsel-recall | held ΔEM |
|---|---|---|---|
| **bo-pure** | **+.019 ± .016** | **+.029 ± .008** | +.014 |
| gepa | +.010 ± .019 | +.026 ± .035 | +.004 |
| bo (old) | −.014 ± .006 | −.020 ± .015 | −.008 |

`trajectory_f1.png`, `best_prompts.md`. Root held-out F1 differs from the previous run's by 1-2 pts on the same rows
(.683 vs .699, .721 vs .727): Micro's nondeterminism at temperature 0; every Δ above is against its own run's root.

- Held-out F1 up in 2/3 seeds, mean +1.9 pts; the same seeds' gepa gives +1.0 with one large win and one loss. Not
  separable at n = 3 with ±3-pt evaluation noise, but this is the first bo arm that is not below zero.
- **Selection recall moved out of sample in all three seeds** (+4.2 / +1.3 / +3.2, SE ≈ .015 each), where gepa's recall
  swung −4.5 to +6.3. That is the lever the feedback is about, and bo-pure moved it consistently.
- The surrogate behaved: `n_train` grew every round (to 16-17), 8-9 distinct parents per seed, incumbents at depth 3-6,
  no flat-acquisition rounds after warm-up. Rollout accounting: 16-17 full evaluations per 4,000 calls, the same count
  gepa reached through 100 minibatches + 15 accepted - the gate's savings were spent on the 85 rejects.
- Cost per seed $0.27 vs $0.46 for the old bo arm and $0.22 for gepa: 17 reflector calls instead of ~80.

## What to take from it

1. The bo arm's problems were design, not tuning: minibatch scores as GP targets, a fluke incumbent, and a parent
   value that rewards having been expanded. Removing all three gives an arm that beats GEPA's sampler on every ladder
   condition and is at least level live. Every earlier live bo number (compression v2 included) predates this and should
   be re-run before being cited.
2. A 12-seed run (~$3 bo + ~$2.7 gepa, or gepa reused for the 3 existing seeds) is what it takes to separate +1.9 from
   +1.0 on held-out F1; recall, with SE ≈ .015, already separates at 3 seeds.

## Files

`results.jsonl` (per seed: curve, candidates, best selector, `bo_fits` with best_y / best_obs / flat / ell / noise),
`trajectory_f1.png`, `best_prompts.md`. Heavy artefacts in `runs/hotpot_program3/` (gitignored).
