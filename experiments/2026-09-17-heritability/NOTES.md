# Is the landscape heritable? Parent-offspring regression over existing trees ($0)

Backlog −1(2): before assuming the tree can stack improvements, measure whether a good node's children are good.
Every full-evaluation (parent, child) pair in the saved trees of three families, scores centred on each run's root.
Scripts: `herit.py` (pooled regression, noise floor, sibling ICC, best-of-children), `herit_top.py` (split by
parent above / below root). Full output in `output.txt`, scatter in `parent_child.png`.

Two things "heritable" can mean, and they come apart here:
- **Level inheritance** - child score tracks parent score (slope ≈ 1, r high). Necessary for any tree search.
- **Stacking** - children of above-root parents are themselves above root, so improvements accumulate with
  depth (the synthetic ladder is built this way). This is what makes "expand the best" the right strategy.

## Noise floor

Same root prompt, same rows, evaluated independently in different run directories (temperature 0):

| family | rows | repeats | pooled SD of root F1 | rows that differ between identical evaluations |
|---|---|---|---|---|
| hotpot_program | 200 | 3 seeds x 2 | .010 | 8.5% |
| compress_v2 | 100 | 12 seeds x 2 | .009 | 2.3% |

## Pooled (all pairs)

| family | pairs | slope child-on-parent | r | sibling ICC | P(child > parent) | best-of-children slope |
|---|---|---|---|---|---|---|
| hotpot_program (gepa + bo, gated) | 88 | +.84 ± .08 | .77 | .72 | .35 | +.90 |
| hotpot_program3 (bo-pure, no gate) | 50 | +1.07 ± .12 | .80 | .68 | .26 | +1.07 |
| compress_v2 (gepa + bo, gated) | 467 | +.41 ± .06 | .30 | .29 | .22 | +.71 |

HotpotQA F1 looks strongly heritable pooled; compression F1 does not (the compressor trades F1 for tokens, so
children scatter).

## Split by where the parent sits (the number that matters)

| family, metric | parents ≥ root | slope | r | P(child ≥ root) | parents < root | slope | r |
|---|---|---|---|---|---|---|---|
| hotpot_program F1 | 47 | +.38 | **.09** | .53 | 41 | +.84 | .83 |
| hotpot_program3 F1 | 40 | +1.05 | **.49** | .78 | 10 | +1.29 | .79 |
| compress_v2 F1 | 288 | +.42 | **.06** | .23 | 179 | +.57 | .49 |

The pooled r ≈ .8 on HotpotQA is **"broken begets broken"**: parents well below root (a lost hint, a damaged
template) have children that are also below root, and that spans a wide range. Among parents at or above root -
the only ones a search will actually expand - the correlation is r ≈ .05 in the gated trees and ≈ .5 in the
bo-pure trees. Children of the best parents are not reliably above root (P = .53 gated, .78 bo-pure), and the
mean child sits *below* its parent by 1 pt. Pairs by depth: 61 of 88 are depth ≤ 2 and the best nodes are at
depth 1-2 in every run; no run shows a lineage climbing beyond depth 2. **Stacking is not demonstrated.**

Selection bias: in the gated trees only gate-passers get full evaluations, so the pairs are the survivors; the
bo-pure trees (every pick fully evaluated) are the cleaner sample and show the higher top-of-tree r.

## Conclusions

- HotpotQA on Micro is level-heritable but not shown to stack: the tuned root is ~2-3 pts from the model's ceiling,
  every child is a draw from a distribution centred ~1 pt below its parent with a tail above it, and gains are
  harvested from the tail (best-of-n), not accumulated down a lineage. Compression F1 is not heritable in either sense.
- At the top of the tree the parent-to-parent signal (~.02-.03) is the size of the measurement noise (.01 at 200
  rows; ~.03 at 25 rows). Any selector that ranks parents needs either more rows or more children per parent
  before it can see it - measured directly in `2026-09-17-hotpotqa-botree/` (sibling ICC by run).
- A task with real headroom (a weak root, or a harder task) is where stacking could show; that is the next
  live test (backlog).
