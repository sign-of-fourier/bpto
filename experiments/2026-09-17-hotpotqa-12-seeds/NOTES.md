# HotpotQA selector program, GEPA vs bo-pure, 12 seeds x 3,000 rollouts ($5.33)

The 3-seed bo-pure result of 2026-09-16 (held +.019 ± .016 vs gepa +.010 ± .019) extended to 12 paired seeds
to separate the means (backlog 0(a)). Same harness, same arms, same design rules as
`2026-09-16-hotpotqa-bo-pure/`; run in two parts (seeds 0-2, then 3-11) with no code or state change between
them (`bos` is created per run; `git diff` empty). `--rollouts` was the harness default **3,000** (the 2026-09-16
runs used 4,000), the same for both arms.

    python -m experiments.live_compare.hotpot --arms gepa bo --seeds 12 --program --out runs/hotpot_program4

- task: `tasks.hotpotqa.program`, selector under search, answerer fixed; Nova Micro at temperature 0, Nova Lite reflector.
- train 200 rows (seeded split), held-out 300 rows from the remainder, per seed.
- gepa: Pareto-weighted sample -> 1 reflected child -> 5-row minibatch gate -> full 200-row evaluation on pass.
- bo (pure): EI over candidates (own full-train F1, PIT, F1 SE as noise) -> 3 reflected children -> a second GP
  ranks the 3 and the top one gets the full 200-row evaluation; no minibatch, no gate.

## Result

| arm | train gain | **held-out gain** | proposals / seed | full evals / seed | pass or pick rate |
|---|---|---|---|---|---|
| gepa | +.033 ± .006 | **+.020 ± .004** | 67 | 10.7 | 16% pass the 5-row gate; 51% of passers beat the parent on the full set |
| bo-pure | +.013 ± .004 | **+.004 ± .004** | 35 | 12 | the screen's pick beats the parent on the full set 27% of the time |

**Paired bo − gepa on held-out: −.016 ± .007, t = −2.37, p = 0.04 (n = 12).** gepa is ahead in 9/12 seeds.
Per-seed table in `table.md`; anytime curves and per-seed held-out bars in `gains.png`; full rows in
`evaluations.jsonl` (curve, all candidates, bo fits, best prompt).

The 2026-09-16 3-seed picture (bo ahead) was two bo-friendly splits: seeds 0 and 2 are reproducibly the two
splits where bo beats gepa (in both the 2026-09-16 run and this one); over 12 seeds they are the exception.

## Why bo-pure loses

1. **The surrogate child screen adds nothing.** The GP that picks 1 of 3 fresh proposals has no observations at
   those inputs; its pick beats the parent 27% of the time, which is the mutator's base rate for a random child.
   GEPA's 5-row gate, by contrast, doubles the hit rate of what it lets through (51%) for ~5 calls per proposal.
2. **bo sees half the proposals.** At equal rollouts, gepa spends ~20% of its budget on 67 cheap probes and the
   rest on 11 full evaluations; bo spends everything on 12 full evaluations of screened picks. With a mutator whose
   children beat the parent one time in four, seeing more draws matters more than measuring each draw better.
3. bo's train gain (+.013) is below gepa's (+.033) too - it is not a transfer problem.

## Conclusions

- Under the 2026-09-16 design rules the bo arm is behind GEPA at 12 seeds: −.016 ± .007 held. The 3-seed +.019 does
  not replicate. Every "bo ≥ gepa" number on HotpotQA before this run was 3 seeds or fewer.
- "Select to survive" by a surrogate with no observations at the candidates' inputs is at the base rate; GEPA's
  paired minibatch gate is the better cheap filter. BO's remaining seat is "select to expand"
  (`2026-09-17-hotpotqa-botree/`, `2026-09-17-hotpotqa-gepaei/`).
- Nova Micro nondeterminism at temperature 0 is visible in every table: root held-out differs by ~1 pt between
  arms on the same rows.
