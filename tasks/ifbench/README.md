# IFBench task

Optimise a *system prompt* so that a model obeys unusual, programmatically-verifiable constraints
("every sentence must contain the keyword", "no words with three consonants in a row", "use exactly
N sub-bullets"...). The benchmark is [AllenAI IFBench](https://github.com/allenai/IFBench)
(Pyatkin et al. 2025): 300 prompts, 1-2 constraints each, 58 constraint types, no released train split.

This directory is a *user* of bpto, not part of it.

## Setup

```bash
pip install datasets                       # to fetch allenai/IFBench_test (300 rows, ~150 KB)
git clone https://github.com/allenai/IFBench.git vendor/IFBench && pip install -e vendor/IFBench   # the checkers (not on PyPI)
```

`tasks/ifbench/data/ifbench_test.jsonl` is a saved copy in bpto's record format so runs are reproducible
without the Hub; `load_or_fetch()` creates it on first use.

## What is optimised

```
ROOT_PROMPT = "You are a careful assistant... Follow every constraint exactly and literally...\n\nRequest:\n{instruction}"
```

The tree searches over that template. Each `Example` carries `meta.instruction_id_list` + `meta.kwargs`;
the scorer runs the official checkers on the raw completion (no schema, no reference answer, no judge):

| metric | meaning |
|---|---|
| `strict` / `loose` | 1 if *all* constraints pass on the raw / relaxed response (IFBench's prompt-level accuracy) |
| `inst_strict` / `inst_loose` | fraction of constraints passed (instruction-level accuracy) |
| `template_tokens` | tokens in the prompt template (fixed cost per request) |
| `output_tokens` | completion length |

Objectives: `strict_objective()` (default), `loose_objective()`, `compact_objective(w)` = strict − w·template_tokens.

## Running

```bash
python -m tasks.ifbench.run --mock --rounds 2                                  # offline plumbing check
python -m tasks.ifbench.run --provider bedrock --n-train 40 --cheap-n 10 --rounds 2 --holdout 40 --max-calls 400
```

The 300 rows are split by `--n-train`/`--seed` into a train set (the search only ever sees this) and a
held-out remainder; with `--holdout N` the root and the final best prompt are each scored once on N
held-out examples. Those two numbers are the result — everything on train is selection-biased.

Call count ≈ nodes × examples. Successive halving (`--cheap-n` examples first, survivors on the full
train set) keeps it near `nodes × cheap_n + survivors × n_train`; `--max-calls` is a hard `Budget` stop
and completions are cached in `<out>/cache.jsonl`, so `--resume` after a budget stop costs nothing extra.

## Caveats

- With 58 constraint types in 300 rows, any train/held-out split has types the search never saw.
  That is the interesting question (does a better system prompt generalise across constraint *types*?),
  but expect small, noisy deltas; use `--holdout` ≥ 100 before believing a gain.
- Small models (Nova Micro) score low on IFBench regardless of prompt; the checkers are unforgiving.
- GEPA and MIPROv2 papers train on the *test* set. We don't; numbers are not comparable to theirs.
