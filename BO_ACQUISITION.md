# Acquisition functions for Bayesian optimisation over a discrete candidate pool, sequential and batch

*A short survey written for `bpto`, 2026-09-19. Scope: what an acquisition function is, the standard sequential ones,
why a batch of q > 1 is a different problem, the batch acquisitions that exist, and what our own experiments on prompt
trees say about them. Not covered: kernels, hyperparameter fitting, multi-fidelity, constraints, or anything continuous.
Where a claim comes from our runs it says so; everything else is the literature.*

---

## 1. Setting: why there is an acquisition function at all

Bayesian optimisation (BO) is a strategy for maximising a function f that is expensive to evaluate and has no usable
gradient. It keeps a probabilistic *surrogate* of f — almost always a Gaussian process (GP) — fitted to every
evaluation so far, and chooses the next evaluation by maximising an *acquisition function* α(x) computed from the
surrogate's posterior. The surrogate says what we believe; the acquisition says what that belief is worth testing.
The loop is: fit → argmax α → evaluate → repeat.

In `bpto` the objects are prompts. Each candidate is a node in a tree of prompt variants; its input x is an embedding
(Titan V2, 256-d, usually projected to a few PCA directions because a single-lengthscale kernel cannot separate near
from far in 256 dimensions at tens of observations); f(x) is the prompt's score on a training set. Two things make
this different from textbook BO and matter for everything below:

1. **The candidate set is discrete and small.** We never optimise α over a continuous domain; we score a pool of
   existing nodes (10–100) and pick some of them. Argmax is a table lookup. This makes exact batch criteria that
   would be intractable over a continuum perfectly feasible over subsets of the pool — and makes "diversity" a
   question of which existing siblings to bet on, not where to place a new point.
2. **There are two selection seats, and BO sits in one.** *Select to expand* — which pool member the reflector
   rewrites next — is where the surrogate ranks candidates. *Select to survive* — whether a proposal earns a full
   evaluation — is GEPA's paired minibatch gate. The acquisition therefore chooses which parent gets a *cheap*
   probe (one reflection plus a 5-row gate), and the expensive full evaluation only happens on a pass. That
   asymmetry decides what parallelism is worth (section 4).

The surrogate gives, for any set of candidates, a posterior mean vector μ, a variance vector σ², and — needed for
batches — the full posterior covariance Σ. An *incumbent* y⁺ (the best value found so far, or in our code the best
posterior mean at a training input, which is more robust under noise) anchors "improvement". Targets are PIT- or
rank-normalised so the posterior is on a normal scale; that matters for any criterion that takes exp of it.

## 2. Sequential acquisitions: choosing one point

All of these are functions of the marginal posterior (μ, σ) at a candidate and the incumbent y⁺. Write
z = (μ − y⁺ − ξ)/σ, with Φ and φ the standard normal CDF and PDF.

**Probability of improvement (PI, "MPI").** Kushner 1964. α = Φ(z). The oldest; rewards any improvement, however
small, so it is greedy — it hugs the incumbent and needs ξ > 0 to explore at all. Its one modern use is as a
*floor*: a candidate whose PI is below some threshold can be screened out of an expensive batch search
(`pi_floor` in quantecarlo).

**Expected improvement (EI).** Mockus 1978; Jones, Schonlau & Welch 1998 made it the default (EGO).
α = (μ − y⁺ − ξ)Φ(z) + σφ(z). Weighs the *size* of the improvement by its probability, so a high-variance
candidate with a mediocre mean can beat a certain small gain. ξ ≥ 0 shifts the incumbent upward and is the
exploration dial: ξ = 0 is standard, ξ ≈ 0.01 (on a normalised scale) is a common nudge. Closed form, cheap,
no free parameter that needs tuning in practice. It is what `bpto.bo.EI` computes and what every ranking in this
document starts from. Known weaknesses: under observation noise the "incumbent" is ill-defined (hence our
posterior-mean incumbent), and it is myopic — it values only the next observation.

**Upper confidence bound (UCB / GP-UCB).** Srinivas, Krause, Kakade & Seeger 2010. α = μ + κσ. The optimist's
rule; κ trades off directly and has a regret theory behind it (κ growing like √log t gives sublinear cumulative
regret). In practice κ ≈ 2 is used and tuned. It is the acquisition with the most transparent exploration knob,
and the one whose batch version (section 3) has the cleanest theory.

**Thompson sampling.** Draw one sample f̃ from the posterior and take its argmax. Randomised, parameter-free,
naturally exploratory, and — importantly for batches — trivially parallel: q draws give q picks. Over a discrete
pool a *joint* draw (from Σ) is exact; drawing independently per candidate (as `bpto.bo.Thompson` does) ignores
correlation and is only a ranking heuristic.

**Posterior mean.** α = μ. Pure exploitation. Useful as the final "recommend the best" step and as a control.

**Beyond the myopic family (mentioned, not used).** *Knowledge gradient* (Frazier, Powell & Dayanik 2009) values a
point by how much the *posterior maximum* would rise after observing it — correct under noise and when the final
recommendation may be an unevaluated point, at the cost of an inner optimisation. *Entropy search* (Hennig &
Schuler 2012), *predictive entropy search* (Hernández-Lobato et al. 2014) and *max-value entropy search* (Wang &
Jegelka 2017) choose the point that most reduces uncertainty about the location (or value) of the optimum. All are
better-founded than EI and all are more expensive; none has been necessary at our pool sizes.

**On the exploration dial.** Every acquisition has one — ξ, κ, the Thompson temperature — and the choice of
acquisition is itself a coarser version of the same dial (PI < EI < UCB in exploration, roughly). Which setting is
right is a property of the landscape, not the method: our hard ladder rewarded exploitation (section 5), the easy one
did not care. The methodological point (user, 2026-09-19) is that this is why a *single* landscape can never
"falsify" a criterion; it can only say where its default sits.

## 3. Why a batch is a different problem

A batch acquisition chooses q candidates *before any of them is evaluated*. There are two reasons to want that:

- **Parallelism.** q evaluations in flight at once; wall time per observation falls by up to q.
- **Batch quality.** q points chosen jointly can cover the posterior better than q chosen one at a time by a rule
  that does not know the other picks exist.

And one reason it is hard: the second pick is made blind to the first's outcome. Sequential BO gets q updates for
q evaluations; batch BO gets one. Any batch method pays an *information penalty* relative to sequential, and the
literature is largely about keeping that penalty small. Bounds exist (Desautels et al. 2014 show the regret of
GP-BUCB is within a constant factor of sequential GP-UCB for bounded batch sizes), but the practical question is
always empirical: at equal evaluations, how much does the batch lose?

**The wrong answer: independent top-q.** Rank every candidate by a sequential acquisition and take the top q. This
ignores Σ entirely. When candidates are correlated — and in a prompt tree siblings are near-identical embeddings
with near-identical posteriors — the top q are q copies of the same bet. It is the natural first implementation
and it is what `BOSelector.top(k)` did before 2026-09-18. It is *not* useless: when the pool is small relative to q,
or when the landscape rewards repeatedly hammering the best region, it is fine and can even win (section 5). But it
has no mechanism to do anything else, which is why it is a control and not a method.

**What q buys in a GEPA-style loop.** In `bpto`'s gepa/gepa-ei schedule a round is reflect → 5-row minibatch gate
→ (full evaluation, only on a pass). Passes are the minority (8–20% of rounds on HotpotQA), so the typical round is
one reflection call and one wave of the concurrency semaphore. Those are exactly the serial parts a second chain
overlaps, and two 5-row gates fit inside 16 slots. So q = 2 halves the round count and most of the wall time
without touching the concurrency limit; the full evaluations, which are the rollout cost, happen at the same rate
either way. The batch acquisition is therefore deciding which second parent gets a *5-rollout* probe. That is the
sense in which "q-EI just gets you to the minibatch" (user, 2026-09-18): the expensive step is downstream of the
acquisition and gated by evidence, not by the surrogate.

## 4. Batch acquisitions

Ordered from heuristic to exact. Every one of them takes (μ, Σ, y⁺, q) over the pool and returns q indices; that is
the `BatchAcquisition` protocol in `bpto.bo.acquisition`.

### 4.1 Fantasies: kriging believer and constant liar

Ginsbourger, Le Riche & Carraro 2010 ("Kriging is well-suited to parallelize optimization"). Pick the argmax of a
sequential acquisition; *pretend* you observed it; condition the GP on that fantasy; re-rank; repeat q times.
**Kriging believer** (KB) uses the posterior mean as the fantasy; **constant liar** uses a fixed value (the current
min, mean, or max of y, giving progressively more exploration). Conditioning on a fantasy does not move the mean
(if the fantasy *is* the mean) but it removes the variance the picked point explained from everything correlated
with it — which is precisely what stops the second pick being a sibling of the first.

Properties. Deterministic and cheap in arithmetic: q rank-one updates on the n×n posterior covariance. But:

- **The fantasies are optimistic and the error compounds.** Each pick was chosen because it looked good; believing
  its mean is a systematically favourable observation. By the eighth pick the posterior has been conditioned on
  seven invented observations and no longer resembles the real one. KB drifts toward the incumbent as q grows;
  constant-liar-max over-explores. This is the "noisier as q grows" objection (user, 2026-09-19) and the reason KB
  was always presented as an approximation to q-EI, not an alternative.
- **It is inherently serial.** Pick i cannot start until update i−1 is in. There is nothing to hand a GPU; q = 8
  is eight sequential solves regardless of hardware. q-EI's Monte Carlo (4.3) is the opposite shape.
- **Where it is fine.** A pool of 10–15 candidates and q ≤ 3 — the regime every GEPA-style loop starts in — where
  the chain is microseconds and the pool barely contains q distinct bets anyway. Our replay found KB and local
  q-EI agreeing in 49/72 rounds at q = 2 on pools ≤ 11.

Snoek, Larochelle & Adams 2012 generalise the idea by *integrating* over fantasies (Monte Carlo over possible
outcomes of the pending points) rather than picking one; that is the bridge to 4.3.

### 4.2 Exact multi-point EI

Chevalier & Ginsbourger 2013 ("Fast computation of the multi-points expected improvement"). q-EI is the expected
improvement of the *best of the batch*:

    α_q(X) = E[ (max_{i≤q} f(x_i) − y⁺)⁺ ]   with f(X) ~ N(μ_X, Σ_X).

Ginsbourger et al. 2010 gave the definition; Chevalier & Ginsbourger showed it has a closed form as a sum of q
terms, each a q-dimensional multivariate normal CDF plus (q−1)-dimensional ones — O(q²) CDF evaluations per batch.
Exact and deterministic, but multivariate normal CDFs are themselves numerically integrated and the cost is steep
beyond q ≈ 10. Over a discrete pool of n the *search* is the other cost: C(n, q) subsets. At n = 45, q = 8 that
is ~2×10⁸ — the reason a hosted implementation has "exact" and "screened" regimes and a PI floor to prune.

### 4.3 Monte Carlo q-EI and greedy construction

The practical form (Snoek et al. 2012 for fantasies; Wilson, Hutter & Deisenroth 2018 for the theory; BoTorch,
Balandat et al. 2020, for the standard implementation). Draw S joint samples f_s ~ N(μ, Σ) over the whole pool
once — by the reparameterisation f_s = μ + L ε_s with L a square root of Σ — and estimate

    α_q(X) ≈ (1/S) Σ_s (max_{i∈X} f_s,i − y⁺)⁺.

Two things make this the workhorse:

- **Submodularity → greedy is near-optimal.** Wilson et al. 2018 show that q-EI (and q-PI, q-UCB) as a function of
  the *set* X is submodular, so building the batch greedily — add the candidate that most raises the batch's
  q-EI, q times, on the same shared samples — is within (1 − 1/e) of the optimal batch. That replaces C(n, q) with
  n·q evaluations of a cheap max over samples. This is exactly `bpto.bo.QEI`. The shared samples make each greedy
  step a common-random-numbers comparison, so the noise of the estimate does not swamp the small differences
  between candidates.
- **It is embarrassingly parallel.** Samples × candidates is a matrix; scoring many subsets is independent work.
  Where KB is a chain, q-EI is a GEMM. That is what makes a GPU service sensible and is the structural argument
  for q-EI over fantasies that does not depend on any accuracy result.

The greedy first pick is always the sequential-EI argmax (the max over a one-element batch *is* EI), so q-EI never
loses the sequential decision; it only changes what goes in the remaining q−1 slots. Every batch method in this
document has that property, which is why in our runs the q = 1 chain is identical across them and only the second
parent differs.

**Scale of the objective.** q-EI on f and q-EI on exp(f) are different criteria. A lognormal objective
(improvement measured on exp of a normal-scale posterior) weights the top of the distribution more heavily — it is
more exploitative in the upper tail and makes candidates far below the incumbent contribute exactly zero. The
hosted quantecarlo service computes the lognormal form and therefore requires a normal-scale posterior (rank-normal
or PIT) and an incumbent on the same scale. `QEI` in `bpto` computes the plain form. Their picks are supposed to
differ by design; see 5.4 for what actually happened.

### 4.4 Batch UCB and hallucinated observations

Desautels, Krause & Burdick 2014 (GP-BUCB) and the related "hallucinated" UCB: run UCB sequentially but, for the
pending points, update only the *variance* (which does not depend on the observed value) and leave the mean alone.
This is the one batch method with a regret bound comparable to its sequential version, and it is what KB is when
the sequential acquisition is UCB. The same serial-chain shape as KB.

### 4.5 Local penalisation

González, Dai, Hennig & Lawrence 2016. Pick the sequential argmax, then multiply the acquisition by a penalty
function that is ~0 near the pick and ~1 far away (its radius derived from a Lipschitz estimate of f), and repeat.
A diversity mechanism that does not need Σ at all, useful with non-GP surrogates. In a discrete pool with
embeddings the natural analogue is a kernel-distance penalty; we have not needed it because Σ is available.

### 4.6 Determinantal point processes and other diversity priors

Kathuria, Deshpande & Kohli 2016 sample the batch from a DPP whose kernel is the posterior covariance restricted to
a high-UCB region, so diverse batches are exponentially more likely. Elegant; the diversity is explicit rather than
a side-effect of the improvement calculation. Related work uses uncertainty sampling or k-means over the posterior.
Worth knowing as the "diversity for its own sake" family, which is not what q-EI does — q-EI diversifies only as
far as diversity raises the expected best of the batch, and will happily double up when that is the better bet.

### 4.7 Parallel Thompson sampling

Kandasamy, Krishnamurthy, Schneider & Póczos 2018. Draw q joint posterior samples, take the argmax of each. No
tuning, no subset search, natural batch diversity from the sampling itself, regret bounds under both synchronous
and asynchronous evaluation. The cheapest way to get a *defensible* batch, and the obvious fallback when q is large
relative to the pool. Over a discrete pool of n it is q draws from an n-dimensional Gaussian — trivial.

### 4.8 Parallel knowledge gradient (mentioned)

Wu & Frazier 2016. The batch version of KG: value a batch by the expected rise in the posterior maximum after
observing all q. The right criterion when observations are noisy and the recommendation may be unevaluated, and
correspondingly the most expensive. Not needed at our scale.

### 4.9 Summary table

| method | uses Σ | parallel in q | exploration control | cost over a pool of n | failure mode |
|---|---|---|---|---|---|
| independent top-q | no | trivially | ξ / κ | n | q copies of one bet |
| kriging believer / constant liar | yes | **no** (serial chain) | fantasy value | q rank-one updates | optimistic fantasies compound with q |
| GP-BUCB | yes (variance) | no | κ | q updates | as KB |
| exact q-EI | yes | subset search is | ξ | O(q²) MVN CDFs × C(n, q) | intractable q, n |
| MC q-EI, greedy | yes | yes (samples × candidates) | ξ, sample count | S·n·q | MC noise at tiny differences |
| local penalisation | no | no | penalty radius | q re-rankings | needs a Lipschitz guess |
| DPP | yes | sampling is | kernel scaling | n³ per sample | diversity even when not wanted |
| parallel Thompson | yes | yes | none (temperature) | q draws | high variance of picks |

## 5. What our experiments say (bpto, 2026-09-17 → 19)

All offline unless stated; scripts in `experiments/synthetic_ladder/ladder.py` (`--q`, `--batch`, `--skills`) and
`experiments/live_compare/replay_batch.py`; harness flags `--q`, `--batch` on the `gepa` and `gepa-ei` arms of
`experiments/live_compare/hotpot.py`.

### 5.1 Replay on recorded HotpotQA trees ($0)

Refit the surrogate at every post-warmup round of the three `gepa-ei` v2 runs (pools of 3–11) and ask each batch
method for q = 2. All methods kept EI's argmax in 72/72 rounds. Second pick differed from top-2 in 15/72 (KB),
32/72 (MC q-EI), 21/72 (hosted, after the fixes in 5.4; 61/72 before them). The hosted criterion sits between KB and
local MC q-EI in how often it departs from the independent ranking, and no method paired siblings more than top-2
did (3/72).

### 5.2 The batch penalty is real for GEPA, and for the surrogate the batch is a gain

Synthetic ladder, GEPA's own loop (1 child, paired minibatch gate, full evaluation on pass), 100 seeds unless noted.
`gepa` draws its q parents from the Pareto pool ∝ examples won, without replacement; `gepa-ei` is the same loop
with the expand seat replaced by EI over a GP on the paired-difference target (v2), q parents by the batch
acquisition. Both switch to q > 1 at the same point (pool > 1). Paired differences on the final true score:

| landscape | pool | gepa q2 − gepa q1 | gepa-ei q1 − gepa q1 | gepa-ei q2 − gepa-ei q1 | gepa-ei q2 − gepa q2 | rounds to 0.9, q2 vs q1 |
|---|---|---|---|---|---|---|
| 6 skills, easy (budget 600) | ~13 | **−.046 ± .020** | −.024 | +.009 ± .023 | +.030 | 13 vs 21–24 |
| 6 skills, hard (noise .3, p .25) | ~45 | **−.018 ± .012** | −.042 | +.036 ± .018 | +.013 | 32–36 vs 61–71 |
| 12 skills (30 seeds) | ~45 | **−.090 ± .032** | −.105 ± .033 | **+.135 ± .038** | **+.120 ± .034** | 54 vs 103–120 |

And at q = 4 on 12 skills: gepa q4 − gepa q1 = **−.173 ± .029** (no seed reaches 0.9); gepa-ei q4 (q-EI) − gepa-ei
q1 = +.117 ± .034; gepa-ei q4 − gepa-ei q2 = −.018 ± .037; gepa-ei q4 − gepa q1 = **+.012 ± .031 at 31 rounds
versus 103**.

Three readings, in order of how much they change what we do:

1. **The blind draw costs GEPA, monotonically in q.** Picks 2..q are made as if pick 1 had not happened: no pool
   update in between, no posterior to hedge with. The penalty grows with the pool (−.046 → −.090 at q = 2) and with
   q (−.090 → −.173). This is §3's information penalty landing on a method with nothing to pay it with.
2. **For EI the batch is not free; it is the missing exploration.** Sequential EI trails the Pareto sampler on every
   ladder: with ξ = 0 it locks onto the parent with the best expected child (the 26–76%-of-rounds-on-one-node pattern
   in the live runs). q-EI's second and third picks are made *conditional on the first being unobserved*, which pushes
   them to other parents; the arm jumps by .12–.14 over its own q = 1 and lands at or above gepa q1. The joint
   criterion is doing the job of the exploration dial (§2.5), and it is what makes EI competitive in the expand seat
   at all. A ξ > 0 or UCB at q = 1 would presumably buy some of the same; q > 1 buys it and the parallelism together.
3. **Rounds fall as 1/q with no accuracy cost between q = 2 and q = 4.** That is the speed result: **a posterior is
   what makes a batch free** - and here, better than free.

### 5.3 Independent top-2 versus q-EI: the theory, and what the ladder can and cannot say about it

The theory (4.3) is settled: q-EI is the expected best-of-batch under the joint posterior; greedy Monte Carlo
construction is within (1 − 1/e) of its optimum; independent top-q is q-EI with the covariance deleted and is
correct only when the candidates are uncorrelated. The ordering exact ≥ greedy MC ≥ fantasies ≥ independent is not
something an experiment on one landscape can overturn, and we did not run one to try. What a ladder *can* show is
two mechanical facts and one caution:

| landscape | q-EI − top-2 (paired) | sibling pairs, top-2 vs q-EI |
|---|---|---|
| 6 skills, easy | −.009 ± .020 | 24% vs 13% |
| 6 skills, hard | −.022 ± .014 | 27% vs 13% |

- **The mechanism works as stated.** q-EI halves the sibling-pair rate: the covariance term does exactly what it is
  there to do, and independent ranking does double up on near-copies.
- **At q = 2 and q = 4 the two are not separable on the ladder** (+.009, +.022 for top-2; q-EI − top-4 = +.015 ±
  .031 on 12 skills). q-EI does pair siblings less (51% vs 65% of rounds at q = 4), but the ladder's siblings differ
  by a skill edit and their hash embeddings are not near-duplicates, so the failure top-q is built to prevent -
  filling the batch with copies of one bet - barely occurs there. Real prompts, where siblings embed within ε of each
  other (§3), are where it must diverge; the ladder is not the instrument that can falsify the criterion.
- **The caution: the sign of a small difference is landscape-dependent.** On the hard ladder the more exploitative
  (theoretically wrong) rule was nominally ahead, at 1.6 SE, because a mutation operator that is informative a
  quarter of the time rewards two tries at the best parent. That is the exploitation dial (section 2) showing
  through, and the response is to set ξ with more than one landscape in view — not to pick the rule that has no
  dial because it happened to sit where one landscape wanted it. A method that is put into production on tasks it
  has not been tested on has to be the principled one.

### 5.4 The hosted q-EI service: two defects found and fixed (2026-09-19)

The first 12-skill run of `QuantecarloQEI` scored .604 ± .020 - below gepa q2 - while keeping EI's argmax in the
first slot (42/51 rounds) and filling the second with far-away, low-mean candidates. A six-candidate posterior
(means [1.7, 1.6, 1.2, 0, −1, −1.1], independent σ = .3, y⁺ = 1.7) made it reportable: at q = 3 the service
preferred {best, 2nd, worst} to {best, 2nd, 3rd}, the reported q-EI was 3–5× the true
E[(exp max f − exp y⁺)⁺], and all-equal means returned HTTP 500. The service's owner traced it (quantecarlo
a947405, f01d37c) to three things, none of them the subset search:

1. **An absolute 0.01 jitter on the orthant covariance.** Exact q-EI (§4.2) needs orthant probabilities of the
   transformed vector (f_a, f_a − f_j); on a σ = .3 posterior the jitter moved that correlation from .71 to .65,
   which penalised *close-mean* pairs by ~13% and far pairs by ~1%. That is a systematic bias against exactly the
   pairs a good batch contains, and it is what the ladder saw. Now 0.
2. **The reported `qei` omitted the strike term** −e^{y⁺}·P(max f > y⁺): the score was the tilted sum alone. Fixed;
   the response now agrees with local Monte Carlo to ~1e−3 on the test cases.
3. **Singular covariance.** Sibling prompts often share an embedding, so two candidates are the same point
   (correlation exactly 1). The coordinate f_a − f_j then has variance 0, standardises to NaN and tripped
   `Normal.cdf` - the old jitter had hidden this too. Fixed inside the orthant problem: a zero-variance coordinate
   is a deterministic constraint, a corr = 1 pair folds into one coordinate, and a batch holding both copies scores
   exactly the batch with one (q-EI({a, a′}) = EI(a) to 1e−9). `bpto.bo.QEI` had handled the same case from the
   start by eigendecomposition with eigenvalues clamped at 0.

After the fix the six-candidate cases are order-invariant and monotone in the partner's mean (12/12 agree with
local `QEI` on the chosen set), and the 12-skill rerun scores .900 ± .027 (67% of seeds reach 0.9) against local `QEI`’s .864 ± .025 (53%): +.036 ± .041 paired, not separable at 30 seeds, and both well clear of gepa q2 (.744) and gepa q1 (.833). At q = 4 the picture flips: hosted .775 ± .023 against local
.845 ± .021 (−.071 ± .031 paired; hosted q4 − hosted q2 = −.125 ± .040) while local q-EI loses nothing from q = 2
to 4. Either the exp-weighting over-exploits once four picks are on the table, or the ρ ∈ (0.97, 0.99) approximation
error below compounds with q; reported to the service owner, and until it is understood the guidance in §6 is local
`QEI` at q ≥ 4. A related property matters for the expand
seat specifically: pool members that have already been observed have a collapsed posterior (sd ≈ .004), and any
batch containing one is nearly perfectly correlated with the batch without it (ρ ≈ 0.99997). The service folds that
coordinate (`dup_corr`, default 0.01), so such a batch scores as EI of the other member to ~0.2% - exact in the
limit, since a collapsed point adds nothing to improvement. The remaining approximation risk is the 0.97–0.99 band:
*unobserved* candidates that are near-duplicates of each other, which sibling prompts produce routinely; the
service's owner is treating that separately. `pi_floor` drops observed points from the batch search outright if
wanted (their marginal PI ≈ 0).

### 5.5 Parallelism in the live loop

The ladder can only count rounds (a mock model has no latency). Seconds exist only in the live harness. The
arithmetic for a HotpotQA gepa-ei run (51 rounds, 9 gate passes, reflection ≈ 3 wave-equivalents of Micro latency,
full evaluation 13 waves at concurrency 16): sequential ≈ 51·4 + 9·13 = 321 units; q = 2 ≈ 26·4 + 117 = 221, about
30% less, approaching 2× as the pass rate falls. No concurrency change is needed because the parallelised part is
the reflection and the 5-row gate. Whether that materialises is the one thing only a live run answers (~$1.4 for two
arms × three seeds, paired with the existing gepa-ei runs).

## 6. Practical guidance for `bpto`

- **Sequential acquisition: EI**, ξ small, PIT targets, per-observation noise, posterior-mean incumbent. UCB is the
  alternative when an explicit exploration knob is wanted; Thompson when a randomised policy is.
- **q = 1 is the sequential path, unchanged.** `BOSelector(batch=None)`.
- **q > 1: a joint criterion, never independent top-q as the default.** Top-q stays as a control.
  - Small pool relative to q (≲ 5q candidates): `KrigingBeliever()` is adequate and free; the pool does not contain
    enough distinct bets for the criterion to matter.
  - Wider pool, or q ≥ 4: `QEI()` (local Monte Carlo, greedy). Its picks are the settled criterion within a
    (1 − 1/e) factor and it scales as S·n·q.
  - The hosted `QuantecarloQEI` (fixed 2026-09-19, 5.4) when n·q makes the exact criterion worth the round trip; on
    the 12-skill ladder it is the best arm at q = 2 and behind local `QEI` at q = 4 (5.4) - use it at q ≤ 2 until
    that is understood. PIT posterior and same-scale incumbent are required by its lognormal contract; expect a
    more exploitative batch than local `QEI` (exp weights high means).
- **Exploitation/exploration is a setting, not a verdict.** Choose ξ (or the criterion) with more than one landscape
  in view; do not re-tune per dataset — the point of the method is to run it on tasks it has not been tested on.
- **What q parallelises** is the reflection and the gate. The full evaluation is downstream of the acquisition and
  gated by evidence; do not size q against the evaluation concurrency.

## References

- Kushner 1964. A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. *J. Basic Eng.*
- Mockus, Tiesis & Zilinskas 1978. The application of Bayesian methods for seeking the extremum. *Towards Global Optimization 2.*
- Jones, Schonlau & Welch 1998. Efficient global optimization of expensive black-box functions. *J. Global Optim.*
- Srinivas, Krause, Kakade & Seeger 2010. Gaussian process optimization in the bandit setting: no regret and experimental design. *ICML.*
- Ginsbourger, Le Riche & Carraro 2010. Kriging is well-suited to parallelize optimization. In *Computational Intelligence in Expensive Optimization Problems.*
- Snoek, Larochelle & Adams 2012. Practical Bayesian optimization of machine learning algorithms. *NeurIPS.*
- Hennig & Schuler 2012. Entropy search for information-efficient global optimization. *JMLR.*
- Chevalier & Ginsbourger 2013. Fast computation of the multi-points expected improvement with applications in batch selection. *LION 7.*
- Desautels, Krause & Burdick 2014. Parallelizing exploration–exploitation tradeoffs in Gaussian process bandit optimization. *JMLR.*
- Hernández-Lobato, Hoffman & Ghahramani 2014. Predictive entropy search for efficient global optimization of black-box functions. *NeurIPS.*
- González, Dai, Hennig & Lawrence 2016. Batch Bayesian optimization via local penalization. *AISTATS.*
- Kathuria, Deshpande & Kohli 2016. Batched Gaussian process bandit optimization via determinantal point processes. *NeurIPS.*
- Wu & Frazier 2016. The parallel knowledge gradient method for batch Bayesian optimization. *NeurIPS.*
- Shahriari, Swersky, Wang, Adams & de Freitas 2016. Taking the human out of the loop: a review of Bayesian optimization. *Proc. IEEE.*
- Wang & Jegelka 2017. Max-value entropy search for efficient Bayesian optimization. *ICML.*
- Frazier 2018. A tutorial on Bayesian optimization. arXiv:1807.02811.
- Wilson, Hutter & Deisenroth 2018. Maximizing acquisition functions for Bayesian optimization. *NeurIPS.*
- Kandasamy, Krishnamurthy, Schneider & Póczos 2018. Parallelised Bayesian optimisation via Thompson sampling. *AISTATS.*
- Balandat, Karrer, Jiang, Daulton, Letham, Wilson & Bakshy 2020. BoTorch: a framework for efficient Monte-Carlo Bayesian optimization. *NeurIPS.*
