# HotpotQA two-stage program: GEPA vs BO on the paragraph selector (3 seeds x 4,000 calls, $1.90)

**Question.** The single-prompt HotpotQA task was flat (see `2026-09-15-hotpotqa-phase1`). Is that because the
mutator is useless on this model, or because a single prompt over ten given paragraphs has nothing to decide?
Following the GEPA paper's multi-hop formulation, make the searched prompt control a *decision*: stage A
(searched) selects the paragraphs needed, stage B (fixed answerer prompt) answers from only those. Feedback
to the reflector is about the selection ("MISSED [title]"), not the answer. `tasks/hotpotqa/program.py`,
`tasks/hotpotqa/feedback.py::program_feedback`, harness `experiments/live_compare/hotpot.py --program`.

Pre-registered (before the run):
- **H1** GEPA's mechanism works when the prompt controls a decision: held-out ΔF1 > 0 in ≥ 2/3 seeds, selection recall rising.
- **H2** the oracle (answerer on the gold paragraphs, 0.780 on the pilot rows) is the ceiling; anything above it is an artefact.
- **H3** BO's edge is noise-robustness: bo's train-held gap smaller than gepa's in ≥ 2/3 seeds.

## Pilot ($0.12, 300 rows, temperature 0) - `pilot.md`

Oracle selection 0.780 F1 vs all-ten single prompt 0.715; five hand-written selectors 0.625-0.742, tracking selection
recall (0.64 -> 0.89). A 12-pt spread where the single-prompt task had 3 - the headroom exists. Root selector: recall 0.863,
precision 0.899.

## Setup

Nova Micro, temperature 0, `max_tokens 512`; reflector Nova Lite, temperature 1.0. Train 200 / held-out 300 per seed
(disjoint, seeded). Objective: answer F1 (LinearObjective f1=1.0); `sel_recall`, `sel_precision`, `n_selected`, `em` logged.
Rollouts = task-model calls; the program makes two per example, so 4,000 calls ≈ 2,000 example rollouts ≈ 10 full-train
evaluations. gepa: Pareto-weighted parent, 1 reflective child, minibatch-5 gate `beats_parent`. bo: `BOSelector(Titan V2,
GPR, EI)` over parents after 4 warm-ups, 3 children per round screened to 1 by a second surrogate. Per-run
`Budget(max_calls=5,650)`; three seed processes in parallel, ~11 min each.

## Result

| arm | seed | train F1 | train recall | held F1 | held sel-recall | train-held gap | accepted |
|---|---|---|---|---|---|---|---|
| gepa | 0 | .732→.746 | .873→.907 | .694→.740 (**+.046**) | .832→.895 (**+.063**) | −.032 | 15/97 |
| gepa | 1 | .713→.736 | .838→.800 | .692→.674 (−.018) | .827→.782 (−.045) | +.041 | 14/112 |
| gepa | 2 | .662→.699 | .805→.840 | .741→.742 (+.001) | .863→.922 (**+.058**) | +.035 | 15/99 |
| bo | 0 | .756→.761 | .868→.897 | .699→.697 (−.003) | .837→.843 (+.007) | +.008 | 15/67 |
| bo | 1 | .712→.716 | .848→.833 | .700→.683 (−.017) | .835→.815 (−.020) | +.021 | 14/123 |
| bo | 2 | .657→.672 | .805→.757 | .727→.705 (−.022) | .862→.815 (−.047) | +.037 | 15/75 |

gepa: held ΔF1 **+.010 ± .019**, held Δrecall **+.026 ± .035**, train-held gap +.015.
bo: held ΔF1 **−.014 ± .006**, held Δrecall −.020 ± .015, train-held gap +.022.

`trajectory_f1.png`: anytime best train F1. `trajectory_recall.png`: selection recall *of the F1-chosen incumbent* - it
falls in three of six runs, i.e. the incumbent was chosen for an F1 gain that did not come from better selection.
`best_prompts.md`: the six winning selectors.

- **H1: partly.** In 2/3 gepa seeds the reflector produced selectors with held-out recall +6 pts (recall on 300 rows has
  SE ≈ 0.015, so both are real). This is the first time on HotpotQA that anything moved out of sample, and it moved the
  thing the feedback was about. But held-out F1 met the pre-registered 2/3 bar only if +.001 counts: one clear win (+4.6), one
  zero, one loss.
- **H2: holds.** Best held-out F1 0.742 < oracle 0.780.
- **H3: falsified.** bo's train-held gap is not smaller (+.022 vs +.015) and it lost held-out in 3/3. In seed 2 it picked a
  candidate with recall 5 pts *below* root because its train F1 was 1.5 pts higher.
- The rule across all six runs: **when the F1-best candidate's recall was below root's, it lost held-out; when recall rose,
  held-out did not fall.** F1 is the objective but recall is the lever; 200-row F1 is ±3 pts of noise
  (Micro nondeterminism, see phase-1 notes), 200-row recall about half that. Selection ran on the noisier signal.
- Seed 2 (gepa): recall +5.8 on held-out, F1 +0.1. The answerer did not convert better paragraphs into better answers -
  the fixed stage became the bottleneck, which is the paper's argument for searching every module.
- Reflector JSON failures: 9 of ~770 proposals (1.2%), mostly `Invalid \escape` (Nova Lite writes markdown `\*`, `\_`
  inside JSON strings). Recorded, skipped, not retried.

## What to take from it

1. The "decision vs description" hypothesis survives: same model, same reflector, same dataset - the selector prompt
   moved held-out recall where the single prompt moved nothing. The single-prompt flatness was the task, not the mutator.
2. The comparison between selection strategies is still not decided by this run: at 3 seeds, held-out F1 cannot resolve
   a +1 pt mean, and both arms were selecting on the noisier of two available signals.
3. BO did not show noise-robustness here. Its surrogate is fit on single noisy F1 measurements with no noise term
   (`BOSelector(noise=)` unset) - the ladder result that BO beats the sampler under noise assumed the GP knows the noise.

## Open questions (not resolved here; user's call whether to spend on them)

- **Objective.** The paper scores its multi-hop modules on retrieval recall as well as answer F1. With a fixed answerer the
  selector's honest objective is recall (or F1 + recall). Re-running the 25% with `LinearObjective(f1=0.5, sel_recall=0.5)`
  (~$1.9) is the obvious next test: it should stop both arms from accepting recall-losing candidates. Held-out F1 stays the
  final readout; the oracle stays the ceiling.
- **Is BO built for programs?** Not yet. A bpto node is one `Prompt`; GEPA's node is a whole program (all module prompts)
  and its mutation is "one module, round-robin". Here both arms searched one module, so the comparison was fair, but a
  multi-module run would need (a) `Node.prompt` to become a program, (b) the reflector to take a `module=`, (c) an embedding
  of a program. Proposed: concatenate per-module embeddings (equivalently an additive kernel, sum of per-module RBFs), never
  pool them - pooling destroys attribution. The user's concern is dimension (1,024 × modules). A GP's cost is in the
  number of nodes, not the dimension, and the kernel only sees per-module distances, but whether a GP can *attribute*
  across modules with ~100 nodes is untested. Test on the synthetic ladder before spending on it live.
- **Surrogate noise.** Give `BOSelector` a per-node SE (from `metrics_std`/n) or re-measure incumbents before any further BO
  vs GEPA claim on a noisy metric. Every live BO result so far is with `noise=None`.
- **Second module.** Searching the answerer too (round-robin) is what would let seed-2-style recall gains turn into F1.
  It needs the program-node change above.

## Files

`results.jsonl` (one row per arm-seed incl. curve, all candidates, best prompt), `trajectory_f1.png`, `trajectory_recall.png`,
`best_prompts.md`, `pilot.md`. Heavy artefacts (caches, trees, events) in `runs/hotpot_program/` (gitignored).
Code: commit `64b9f6d` (program task, feedback, `--program`).
