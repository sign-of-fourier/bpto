# Compression: GEPA vs BO under an annealed accuracy floor (Nova Micro, 5 seeds, $0.065)

Maiden run of the constrained/continuation design (BACKLOG 4b). Harness: `experiments/live_compare/compress.py`,
commit after `3782c2f`. Raw: `results.jsonl` (per run: anytime curve, floor trace, final front, held-out
metrics, best prompt); `summary.md`; `curves.png`.

## Setup

- Task `tasks/compression` (synthetic name-extraction passages, `generate_dataset(170, seed=0)`): 30 train,
  100 held-out per seed (seeded split). Root prompt = 97 template tokens (chars/4 estimate), train F1 0.956-1.0.
- Objective: **minimise `template_tokens` s.t. train F1 >= floor(rollouts)**, floor = root_f1 - 0.15 at the
  start, +0.02 every 100 rollouts, capped at root_f1 - 0.05. Same schedule in both arms, keyed on the task
  client's call count. Every stored evaluation is re-scored when the floor moves (no calls). No penalty:
  infeasible nodes stay in the tree and the pool and can be parents; only the "best" readout (shortest
  feasible candidate) needs feasibility.
- Mutator (both arms): `ReflectiveExpander` with `COMPRESS_REFLECT_PROMPT` ("shorter, keep the rule that
  fixes the shown misses/spurious names"), 3-example minibatch of traces with missed/spurious feedback,
  failures first (`passed()` = exact set match), Nova Lite at temperature 1.0.
- Gate (both arms): child on the 3-example minibatch vs parent on the same ids, constrained order -
  feasible beats infeasible; both feasible -> shorter wins; both infeasible -> higher F1 wins. Accepted
  children get the full 30-example evaluation and join the candidate set.
- **gepa**: Pareto pool on per-example F1 (`pareto_sample(metric="f1")`), sample ∝ examples won, 1 parent,
  1 child per round. **bo**: GEPA sampler for the first 4 expansions, then EI (Titan V2 embeddings, GPR)
  over all candidates with value = best child's *gain* (tokens saved vs root if feasible, else 0);
  3 children, surrogate (own gain) keeps 1 for the minibatch.
- 600 rollouts per arm-seed; caches per arm-seed (a shared cache gave the second arm its root for free).

## Result

| seed | root tok | root held F1 | gepa tok / held F1 / train F1 | bo tok / held F1 / train F1 | final floor |
|---|---|---|---|---|---|
| 0 | 97 | 0.990 | 26 / 0.908 / 0.960 | 5 / 0.875 / 0.917 | 0.917 |
| 1 | 97 | 0.997 | 17 / 0.990 / 1.000 | 19 / 0.910 / 0.967 | 0.950 |
| 2 | 97 | 0.990 | 6 / 0.942 / 0.933 | 7 / 0.932 / 0.933 | 0.872 |
| 3 | 97 | 0.997 | 21 / 0.966 / 1.000 | 6 / 0.963 / 1.000 | 0.950 |
| 4 | 97 | 0.973 | 16 / 0.983 / 0.989 | 20 / 1.000 / 1.000 | 0.950 |

- gepa: **17.2 ± 3.3** tokens, held-out F1 0.958. bo: **11.4 ± 3.3** tokens, held-out F1 0.936.
  Paired gepa-bo: +5.8 ± 5.1 tokens, BO shorter in 2/5. **Not significant**, and BO's extra compression
  is paid for in held-out accuracy (mean drop vs root: gepa -0.032, bo -0.041).
- Both arms compress 4-19x (97 -> 5-26 tokens) while staying within 0.05 train F1 of the root. Typical
  winners: `Extract full names from: {text}`, `Extract names of people from this text, excluding titles,
  organizations, and places: {text}`.
- The "find a new best" event happened in 5 of 10 runs (`floor_trace` in results.jsonl): e.g. bo s3 best
  8 -> 17 tokens at rollout 216 when the floor crossed the 8-token prompt's F1, then back down to 6 by the
  end; gepa s3 15 -> 24 at rollout 425. Infeasible candidates accumulate as the floor rises (3-12 of ~20 at
  the end). Mechanically the design does what was intended.

## What the trace says (why this is not yet a selection test)

1. **Selection barely had a say.** 600 rollouts / 30 train examples = 20 full evaluations, and the gate
   accepted 18-20 children in every run (acceptance 55-90%): a shorter prompt almost always beats its
   parent on a 3-example minibatch. So ~570 of 600 rollouts went to full evaluations of accepted
   children, and each arm made only ~20 parent choices. BO's surrogate pre-screen threw away 50-100
   children per run, but since the gate lets nearly everything through, screening is not where the
   rollouts were going.
2. **The floor on 30 examples is a loose proxy.** Prompts at the train floor lose 3-10 pts held-out
   (bo s0: train 0.917 = floor, held 0.875). BO exploits the floor harder, which is what an optimizer
   should do; the comparison at "shortest feasible" therefore rewards overfitting the floor. Compare
   fronts at equal *held-out* F1 instead (right panel of curves.png shows train fronts; both arms' fronts
   interleave).
3. **Variance is dominated by the reflector's dice.** Same seed, same split, the earlier smoke run
   (`runs/compress`, shared cache) gave gepa 9 / bo 7 tokens; this run gave 26 / 5. Five seeds cannot
   resolve a difference of a few tokens under that noise.
4. Nova Micro on this synthetic task is nearly saturated (root train F1 0.96-1.0); a 5-token prompt gets
   0.9. There is headroom in tokens, little in accuracy - the constraint only starts to bind below ~10
   tokens, i.e. in the last 100-200 rollouts.

## Follow-ups (in order)

- **Score every candidate on held-out** (20 candidates x 100 examples x 10 runs = 20k Micro calls,
  ~$0.25) and compare (tokens, held-out F1) fronts / hypervolume. Free of search noise, and the honest
  readout for a constrained problem.
- Make selection matter: raise the price of a bad parent choice - larger train set (100) with the same
  600 rollouts (6 full evals) or a stricter gate (minibatch 6, require shorter *and* no F1 loss), so that
  which parent is expanded decides the outcome. Then 10 seeds.
- Penalty term (`ConstrainedObjective(penalty=...)`) so that infeasible-but-close prompts carry graded
  value for the surrogate, instead of gain = 0.
- Reflector: temperature 1.0 + the compression prompt produced genuinely different rewrites
  (diff ratio to parent 0.34-0.65), unlike the IFBench run - keep it. Nova Pro not needed here.
- Harness: `rollouts` now excludes held-out calls; add the same fix to `compare.py`.
