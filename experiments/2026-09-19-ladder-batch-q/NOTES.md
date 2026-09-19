# q = 2 as a batch acquisition: GEPA's second draw is blind, the surrogate's is not ($0)

**Question.** The user wants a *speed* result, not an accuracy one: at the same rollouts, does drawing two parents
per round (reflect + minibatch-gate both in parallel) reach the same score in fewer rounds? And is GEPA's way of
getting two parents - sampling twice from the Pareto pool - as good as a surrogate's joint pick?

**Design.** Arms `gepa` (Pareto-weighted sampler) and `gepa-ei` (same loop, `BOSelector` in the expand seat, PIT
targets, `pca=4`, child-estimated-full-score observations; `gepaei` v2 renamed). `--q 2` engages once the pool has
more than one member; during warmup both arms draw q via the sampler. `gepa q=2` draws two without replacement;
`gepa-ei q=2` asks `BOSelector.top(2)` with `batch=` one of `QEI` (local Monte Carlo q-EI, greedy), `QuantecarloQEI`
(hosted exact q-EI on a lognormal objective) or `None` (independent top-2, the control). Full evaluations stay gated
by the 5-row paired minibatch; q only parallelises reflect + gate. Three ladders (`experiments/synthetic_ladder/ladder.py`):
6 skills easy (budget 600 and 2000, 100 seeds), 6 skills hard (noise .3, p_informed .25, 100 seeds), 12 skills
(`--skills 12`, 30 seeds, q = 1, 2 and 4). The mock has no latency, so rounds stand in for wall time. Tables in `table.md`, curves in
`curves.png` (`plot.py`), HotpotQA replay rows in `replay_hotpot_gepaei2_q2.jsonl` (`live_compare/replay_batch.py`).

## Results

| ladder | gepa q2 − gepa q1 | gepa-ei q2 − gepa q2 | rounds to 0.9, q2 vs q1 |
|---|---|---|---|
| 6 skills easy, b600 | −.046 ± .020 | +.030 ± .017 | 13 vs 21–24 |
| 6 skills hard | −.018 ± .012 | +.013 ± .016 | 32–36 vs 61–71 |
| 12 skills | −.090 ± .032 | +.120 ± .034 (local), +.156 ± .034 (hosted) | 54 vs 103 |

12 skills, q = 4 (paired, 30 seeds): gepa q4 − gepa q1 = **−.173 ± .029** (0% reach 0.9); gepa-ei q4 (q-EI) − gepa q1 =
**+.012 ± .031 at 31 rounds vs 103**; gepa-ei q4 − gepa-ei q2 = −.018 ± .037; q-EI − top-4 = +.015 ± .031;
gepa-ei q1 − gepa q1 = −.105 ± .033; gepa-ei q2/q4 − gepa-ei q1 = +.135 ± .038 / +.117 ± .034. Full table in `table.md`.

1. **GEPA pays for its blind draws, monotonically in q.** The extra parents are chosen as if the first pick had not
   happened - no pool update between picks and no posterior to hedge with. −.046/−.018/−.090 at q = 2 across the
   ladders, −.173 at q = 4 on 12 skills. This is the information penalty of a batch (BO_ACQUISITION.md §3) landing
   on a method with nothing to pay it with.
2. **The surrogate's batch is not free - it is a gain.** Sequential EI (`gepa-ei q1`) trails gepa q1 on every
   ladder (−.024, −.042, −.105): with ξ = 0 it locks onto one parent, the 26–76%-of-rounds-on-one-node pattern seen
   live. q-EI's hedged batch forces the bets apart, and the arm jumps by +.135 (q = 2) / +.117 (q = 4) over its own
   q = 1, to parity with gepa q1 (+.012 at q = 4) or above it (+.031 at q = 2). **q > 1 is doing the job of the
   exploration dial**, and it is what makes EI competitive with the Pareto sampler in the expand seat at all.
3. **The speed result.** At equal rollouts, gepa-ei q = 4 (q-EI) reaches GEPA q = 1's accuracy in 31 rounds instead
   of 103; q = 2 in 54. Going from q = 2 to 4 costs −.018 ± .037 - inside the noise.
4. **Top-q vs q-EI is not separable on the ladder at q = 2 or q = 4** (+.009, +.022 for top-2; +.015 ± .031 for q-EI
   at q = 4). q-EI does pair siblings less (51% vs 65% of rounds at q = 4) but the ladder's siblings differ by a skill
   edit and their hash embeddings are not near-duplicates, so the clone-taking failure top-k exists to prevent barely
   occurs here. Real prompts are where siblings embed within ε of each other; the joint criterion stays the production
   choice on theory (BO_ACQUISITION.md §5.3), and the ladder is not the instrument to falsify it.
5. **Hosted vs local q-EI: +.036 ± .041 at q = 2, −.071 ± .031 at q = 4** (hosted q4 − hosted q2 = −.125 ± .040).
   The hosted lognormal criterion is the best arm at q = 2 and the worst surrogate arm at q = 4. Either exp-weighting
   over-exploits once four picks are on the table, or the orthant approximation error the service owner flagged for
   ρ ∈ (0.97, 0.99) compounds with q. Reported to the service owner; until resolved, local `QEI` at q ≥ 4.

## The hosted service: two defects found, fixed the same night

The first 12-skill run of `QuantecarloQEI` scored .604 ± .020 with 0% reaching 0.9, filling the second slot with
far-away low-mean nodes. A six-candidate posterior (means [1.7, 1.6, 1.2, 0, −1, −1.1], σ = .3 independent,
y⁺ = 1.7) reproduced it: at q = 3 the service preferred {best, 2nd, worst}; reported q-EI 3–5× too large; all-equal
means → HTTP 500. Root causes (quantecarlo a947405): an absolute 0.01 jitter on the orthant covariance that
penalised close-mean pairs by ~13% vs ~1% for far pairs, and a reported `qei` missing the −e^{y⁺}·P(max f > y⁺)
strike term. The 500 was the q = 1 path. A second 500 surfaced on the rerun: sibling candidates with identical
embeddings give a covariance with an exactly corr = 1 pair (captured request in the log), whose f_a − f_j coordinate
has variance 0 and standardised to NaN - masked before by the jitter. Fixed (f01d37c + `dup_corr`): the coordinate
folds, a batch holding a duplicate scores exactly the batch without it (verified 0.404668 both ways), no nugget.
`bpto.bo.QEI` had handled the singular case from the start (eigendecomposition, eigenvalues clamped at 0). The
service's remaining approximation risk is the ρ ∈ (0.97, 0.99) band - near but not identical unobserved candidates.

After the fix: the six-candidate cases are order-invariant and agree with local `QEI` on the chosen set (12/12);
the 12-skill rerun is the .900 above; the HotpotQA replay's hosted picks differ from top-2 in 21/72 rounds (was
61/72), between KB (15) and local q-EI (32).

## Live parallelism (arithmetic, no run)

q parallelises reflect + minibatch gate; full evaluations are downstream and only on a gate pass (~15% on HotpotQA).
At q = 2 with the same concurrency the round is one reflect + two 5-row gates in the time of one, and the expected
full-eval load per round rises from .15 to .30 of one evaluation - about 30% less wall time per rollout at
HotpotQA-like pass rates, with no concurrency change. A live q = 2 run on `gepa-ei` (≈$1.4, 3 seeds, seconds
readout) is the next step and is pending the user's go; q = 4 after that if q = 2 transfers.
