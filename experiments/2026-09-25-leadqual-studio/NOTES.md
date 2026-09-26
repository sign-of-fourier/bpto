# Lead qualification through the Impromptune studio: 8 discovery runs, $3.22 (+ $1.14 data)

2026-09-25/26. Copied from the downstream studio project (`~/projects/prompt_optimization/lead_qual_synth/`, left in
place there). The runs used bpto via the studio (pinned first at `bb2313a`, then at the gate-rung commits
`5161860`/`4e87136` on `gepa-gate-rungs`, bpto PR #1). They are here because they are what motivated those commits,
and because they are the first bpto result on a classification task with a known generating rule.

**Discovery grade, not a benchmark:** one run per setting, seed 0, hold-out comparisons unpaired (the studio stores
only aggregate hold-out metrics).

## Setup

- **Task:** five-way routing of 3,000 synthetic inbound B2B leads for a fictional vendor (Tallyforge): `accepted`,
  `rejected_fit`, `rejected_no_intent`, `existing_customer`, `partner_route`. Each lead has a latent fit and intent;
  the published SDR labels carry ~8% rater noise (mostly accepted <-> rejected_no_intent). Messages written by Claude
  Haiku 4.5 on Bedrock ($1.14). Dataset, baselines and the four planted traps (redundant field, leakage field,
  frozen-vs-live enrichment, selection bias): `dataset_report.md`, `TRAPS.md`.
- **Reference points** (5-fold CV vs SDR labels): TF-IDF + structured LR 0.725; generating rule on the noise-free
  latents 0.794; majority class 0.411.
- **Seed prompt** (`baseline_prompt.txt`): hand-written, rules applied in order, first match wins. Rule 3
  (`rejected_fit`) is a strict AND of industry, 50-5,000 employees, Salesforce/HubSpot, listed departments, manager+.
  A sealed answer key (kept out of every prompt and run) lists where the seed departs from the generating rule.
- **Program:** one node (`qualify`), no edges: a single classify call. No trace, which matters below.
- **How runs were made:** `reference_code/run_usecase.py` calls the studio's own validation, pilot and run loop with
  `STUDIO_DATA` pointed at a scratch dir (it imports the studio app, so it doesn't run from this repo). GEPA-style
  loop: reflector Nova Pro (temperature 1.0), task model Nova Micro or Lite, 2,400 train / 600 hold-out rows, exact
  match on the label.

## Results

Hold-out SE ≈ 0.020 on accuracy, so a gain needs roughly 4 points to mean anything.

| run | task model | reflector | gate rows | change | rewrites passed | train root -> best | hold-out root -> best | cost |
|---|---|---|---|---|---|---|---|---|
| 1 | Nova Micro | Nova Lite | 5 | none | 1 of 5 | 0.537 -> 0.558 | 0.542 -> 0.575 | $0.13 |
| 2 | Nova Micro | Nova Lite | 15 | none | 2 of 9 | 0.537 -> 0.545 | 0.540 -> 0.565 | $0.19 |
| 3 | Nova Micro | Nova Pro | 15 | none | 0 of 8 | 0.537 -> 0.537 | 0.548 -> 0.548 | $0.18 |
| 4 | Nova Lite | Nova Pro | 15 | none | 1 of 11 | 0.530 -> 0.530 | 0.515 -> 0.513 | $0.38 |
| 5 | Nova Lite | Nova Pro | 15 | critic fix + editing reflect prompt | 4 of 9 | 0.531 -> 0.555 | **0.515 -> 0.575** | $0.58 |
| 6 | Nova Lite | Nova Pro | 15 | critic fix | 3 of 8 | 0.532 -> 0.597 | **0.515 -> 0.568** | $0.55 |
| 7 | Nova Lite | Nova Pro | 15 -> 60 -> 120 | critic fix + gate rungs | 4 of 7 | 0.532 -> 0.602 | 0.515 -> 0.590 | $0.66 |
| 8 | Nova Lite | Nova Pro | 15 -> 60 -> 120 | as 7, objective = balanced accuracy | 3 of 9 | bal. 0.683 -> 0.715 | bal. 0.704 -> 0.740 (SE 0.046) | $0.55 |

Winners compared on the 2,400 training rows (recall against the true labels):

| winner | acc (SDR) | balanced (SDR) | balanced (true) | macro-F1 (true) | recall accepted | recall rej_fit | recall rej_no_intent |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed | 0.532 | 0.683 | 0.714 | 0.687 | 0.73 | 0.23 | 0.82 |
| run 6 (accuracy) | 0.597 | 0.700 | 0.725 | **0.733** | 0.66 | 0.56 | 0.61 |
| run 7 (accuracy + gate rungs) | **0.602** | 0.667 | 0.687 | 0.708 | 0.32 | 0.80 | 0.45 |
| run 8 (balanced + gate rungs) | 0.559 | **0.715** | **0.747** | 0.722 | **0.73** | 0.32 | 0.75 |

Prompts: `best_prompts.md`. Per-node summaries with gate outcomes: `evaluations.jsonl`. Logs: `logs/`.

## Findings

1. **Runs 1-4 failed for three stacked reasons, one of them bpto's.**
   - *Blind critic (studio).* The studio's critic prompt had no slot for the prompt under review; a single-step
     program writes no trace, so the critic saw "(single step)". Run 4: 0 of 161 critic notes named a rule. With the
     fix (critic sees the prompt, quotes the failing rule), run 6: 97 of 100 do.
   - *Additive reflection (bpto `REFLECT_PROMPT`).* It asks for "concrete rules, a step-by-step strategy, or brief
     examples" and to keep "what already works" (the 2026-09-11 fix for a paraphrasing reflector). On a
     first-match-wins rule list that can't work: 33 of 33 rewrites in runs 1-4 left rule 3 unchanged and appended
     advice that rule 3 pre-empts. An editing reflect prompt (run 5, `EDIT_REFLECT_PROMPT` in `run_usecase.py`) made
     8 of 9 rewrites edit in place, 0 append. `REFLECT_PROMPT` stays as is (every committed result used it); tasks
     pass their own `meta_prompt`.
   - *A 15-row strict gate can't see a 3-5 point gain* (about one row). Run 4: 9 of 10 rejections were ties, mostly
     fixes and breaks cancelling. Run 6: 2 of 3 passed rewrites were no better on the full set (~$0.20 wasted).
     Calibration: a rewrite that changed only line breaks flipped 225 of 2,400 Lite answers (83 fixed, 90 broken).
2. **Gate rungs work** (run 7, `gepa(extend=...)`): all 4 passed rewrites beat their parent on the full set; 6 of 7
   children were settled on 60 or 120 rows. In run 8 the gate caught 3 rewrites as inert (no answer changed) at 15 rows.
3. **With a good gate, the objective becomes the limit.** Run 7's winner leads with the strict ICP checklist:
   accuracy wins because `rejected_fit` is the largest class, but accepted recall falls 0.73 -> 0.32. Run 6's prompt is
   the better router. Balanced accuracy (run 8; each row weighted 1/class share, so it stays linear in the rows) stops
   the collapse, but on 600 rows its SE is 0.046 (the weights magnify the ~30-40-row classes), so the hold-out can't
   confirm the gain. Deciding on balanced accuracy needs a bigger hold-out.
4. **The rewriter is still the limit on the product side.** Run 8's winner appends a 4-step "STRATEGY" that restates
   the rule order (additive, no edit to rule 3). Run 6's winner appended four concrete rules; against the answer key
   three are right (exclude Manufacturing/Logistics/Higher Ed/Government on fit; exploration without a project ->
   no intent; evaluating alternatives next quarter -> accepted), one wrong ("staying in the loop" -> accepted).
5. **Accuracy hid the model difference.** With the seed prompt Micro and Lite tie on accuracy (0.537 / 0.530), but Lite's
   macro-F1 is 5-6 points higher (0.687 vs 0.625 against the true labels). Micro reads rule 3 as a hard checklist
   (predicts `rejected_fit` 76% of the time vs a true 40%, never finds `rejected_no_intent`); Lite reads it leniently
   (12%) and ignores the partner flag about a fifth of the time. For routing, per-class recall is the number to watch.

## For bpto

- `gepa(extend=...)`, `reflect_rows=` and `context=` (commits `5161860`, `4e87136`) are the library side of findings
  1-2. Still open: a confusion-table `context=` in the studio (fix #3), and an editing `meta_prompt` as a task option.
- Classification objectives: weigh per-class recall / balanced accuracy, with the hold-out sized for its larger SE.
- Paired hold-out comparison needs per-example hold-out results kept (the studio keeps aggregates only).
- Next: balanced objective + editing reflect prompt in one run (~$0.60); anything published needs more than one seed.

## Files

Committed here: `NOTES.md`, `dataset_report.md`, `TRAPS.md`, `baseline_prompt.txt`, `best_prompts.md`,
`evaluations.jsonl` (74 nodes over 8 runs: op, gate outcome, mean metrics; no per-example rows), `logs/` (run 1-8
driver logs + pilot), `reference_code/` (generator, analysis and studio driver, as used; not runnable from this repo).

Gitignored, in `runs/leadqual_studio/`: the per-run `tree.json` (with per-example evaluations), `events.jsonl`,
`cache.jsonl`, `status.json`, the studio sqlite, the CSVs (`leads.csv`, `impromptune_ready.csv`, `labels_routed.csv`,
`live_enrichment.csv`) and `sealed/` (true labels and the answer key; kept out of the repo on purpose). Run ids 1-8:
`8c22489edeb0 b9d908d085d2 26ebbdc9b43c 7da99b05d90e 53f44c8428d1 cb6efb73aec3 0da81d534e4b ea5ab38c06b8`.
