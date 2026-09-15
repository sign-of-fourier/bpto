# HotpotQA (distractor) phase 1: GEPA vs BO vs 0-shot MIPRO, 3 seeds x 3,000 rollouts, train 200 (Nova Micro / Lite reflector, $2.56 + $0.15 pilot)

First 25% of the planned 12-seed comparison (BACKLOG 4c), run to the pause point to answer "is there anything to
find?" before spending the rest. **Preliminary: the gepa/bo rows are contaminated by a mutator artefact (below)
and will be re-run; the headroom conclusion and the MIPRO rows stand.**

## Headroom pilot (`pilot.md`, `pilot_reasoning.md`, $0.15)

Root + 5 hand-written variants x 300 rows. Root F1 0.732 (EM 0.587); every hand variant is *worse* by 5-8 pts
(paired SE 0.016, all significant). With a `reasoning` field in the answer schema the root is unchanged (0.737)
and the two-hop variant gains 5 pts but stays below root. Landscape has real structure (8-pt spread) and the root
is a strong local optimum - a fair test of "can an optimizer find anything above a good root". Run with the
reasoning schema.

## Setup

- `tasks/hotpotqa`: 600-row seeded sample of the distractor dev set (10 paragraphs, ~1.5k tokens/example);
  per seed 200 train / 300 held-out. Objective: maximise train F1 (official normalisation). No constraint.
- Arms at 3,000 rollouts each (root eval included), same root, split, reflector (Lite, temperature 1.0), minibatch 5:
  gepa (Pareto pool, sample ∝ wins, 1 child/round, gate = child beats parent on the minibatch); bo (EI over
  candidates, value = best child F1, 3 children, surrogate keeps 1 for the minibatch); mipro (16 grounded
  candidates proposed once from the root, categorical TPE trials on minibatches of 10, full eval of the
  best-by-mean every 10 trials; no feedback, no tree growth). Harness `experiments/live_compare/hotpot.py`.

## Result (`summary.md`, `trajectory.png`)

| arm | train gain over root | held-out gain over root | best held F1 (mean ± SE, same rows per seed) |
|---|---|---|---|
| gepa | +0.040 ± 0.007 | +0.048 ± 0.021 | 0.730 ± 0.024 |
| bo | +0.035 ± 0.003 | +0.038 ± 0.007 | 0.737 ± 0.005 |
| mipro | +0.013 ± 0.013 | +0.002 ± 0.002 | 0.698 ± 0.016 |

- **Not bombing.** Both reflective arms find prompts above the root on train *and* held-out in 3/3 seeds
  (held +2.5 to +9 pts). This is the first live task where selection has both headroom and real decisions.
- **0-shot MIPRO finds nothing** in 2/3 seeds (0 of 16 grounded candidates beat the root; the +0.039 train gain
  on seed 2 is held +0.006). Consistent with the paper's MIPROv2 ~0 on HotpotQA and with the pilot: the root
  is already better than what a proposer without execution feedback writes.
- gepa vs bo: 1 seed each way and a tie; paired best-held gepa-bo +0.027 / -0.042 / -0.005. Nothing to
  conclude at n=3 - and see the artefact below.
- **Noise floor:** the same root on the same 200 train rows scored 0.713 / 0.710 / 0.749 across the three arms
  of seed 0 (Nova Micro is nondeterministic), and 0.676 / 0.699 / 0.710 on the same 300 held-out rows. ~2-4 pts
  of any per-seed number is model sampling noise; "gain over root" should be read against that, and best-held
  compared within a seed across arms is the cleaner readout.

![trajectory](trajectory.png)

## Artefact found (fixed in `bpto/ops.py`, tests added)

Nova Lite sometimes returns *two templates glued into one JSON string* with `<prompt>` tags between them. The
expander only stripped leading/trailing tags, so the glued string became one template that renders `{context}`
(1.5k tokens) and `{question}` twice. 5 of the 6 gepa/bo "best" prompts are such doubles, and they beat the best
clean prompt in the same run by 1-2.5 pts train F1 - a doubled-context effect, not a better instruction. The
clean bests still beat the root (gepa +2.7 / +2.6 / +3.9, bo +2.1 / +0.8 / +3.0 train), so the headroom
finding holds, but the gepa/bo magnitudes and their comparison need the re-run. Fix: split returned strings on
`<prompt>` tags and require each placeholder exactly once. Also fixed: Lite's raw newlines inside JSON strings
(`json.loads(strict=False)` fallback in the Bedrock parser) which had been discarding ~8% of reflections.

## Next

Re-run gepa and bo for seeds 0-2 with the fixed expander (~$1.7; mipro rows are clean and kept), then seeds 3-11
if the picture holds (~$7.5). Add the `gepa3` control (3 children, random keep-1) when scaling.
