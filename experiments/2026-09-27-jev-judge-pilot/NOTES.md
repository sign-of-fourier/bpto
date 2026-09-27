# Jev as a scorer: pilot vs token F1, string fuzzy match and the Nova Micro `llm_judge` (2026-09-27)

Question: when the terminal step's output can't be checked exactly, does Jev (TypeSafe's typed-answer model,
`jev-1.13-20260917` via OpenRouter, `typesafe-sdk` 0.7.2) score it better than what bpto has today?

## Setup

- **43 items, hand-labelled by one person (Claude) — labels are the ground truth and the main caveat.**
  - 29 real Nova Micro HotpotQA answers from `runs/hotpot*/` tree.json, sampled where token F1 is uninformative:
    24 with 0 < F1 < 1, 8 with F1 = 0 (seed 0); 3 dropped as ambiguous (hedged answer, surname missing, gold typo
    "19408"). 20 correct / 9 wrong.
  - 14 proper-name pairs, same person or not (nicknames, transliterations, stage names vs Jr./Sr., siblings,
    regnal numbers). 8 same / 6 different. No context passed, so this tests world knowledge.
- Scorers: token F1 (the current HotpotQA metric); `difflib` ratio (string fuzzy match); bpto's `llm_judge` on Nova
  Micro, temperature 0, a fresh client per repeat (the default in-memory cache otherwise replays rep 0); Jev Noul
  ("same answer as the reference?" / "same real person?") and, for QA, a 4-level Jev Score (reported /3).
- Every item scored twice. 86 Jev calls (hard cap 90), 86 Bedrock calls (`Budget(max_calls=90)`), ~$0.001 Nova;
  Jev ~400 input / 29 output tokens per call (OpenRouter price not checked).

## Result (`report.txt`, `python analyze.py`)

| | AUC | acc@.5 | Brier | distinct values | repeat flips |
|---|---|---|---|---|---|
| QA: token F1 | .95 | .62 | .233 | 18 | – |
| QA: Nova `llm_judge` | .89 | .86 | .086 | 3 | 0 |
| QA: Jev Noul | 1.00 | 1.00 | .014 | 15 | 0 |
| QA: Jev Score/3 | 1.00 | 1.00 | .040 | 27 | 0 |
| Names: difflib | .17 | .43 | .421 | 14 | – |
| Names: Nova `llm_judge` | .81 | .79 | .214 | 2 | 1 |
| Names: Jev Noul | 1.00 | 1.00 | .002 | 7 | 0 |

Median latency 0.16 s (Jev) vs 2.5 s (Nova). Nova's misses: gave 1.0 to two wrong answers (a family instead of
"flowers"; a name instead of "quarterback"), 0.0 to three same-person pairs (Ike, Gaahl, Peter Tchaikovsky).
Jev's closest calls were exactly the ambiguous-looking ones (partial "film director" 0.09; "motorcycle road race"
0.30; Gaahl 0.86).

## Reading

- On this set Jev separates right from wrong perfectly, is repeatable (|Δ| ≤ .01 between repeats) and its values are
  graded rather than 0/0.5/1, which is what a GP target and GEPA's paired gate want (fewer exact ties).
- String fuzzy matching is worse than useless for names (AUC .17: siblings and Jr./Sr. look closer than nicknames).
- Token F1 ranks well (AUC .95) but its level is wrong: correct verbose answers score .15-.4. It penalises format, which
  is a legitimate objective, but a separate one — with Jev they can be two metrics in the vector.
- Not shown: anything beyond 43 easy-to-label items, a second labeller, domain data (the studio's proper-name task),
  or cost at scale. Jev returns no reason text, so GEPA feedback has to be written from the probabilities.

## Client smoke (`smoke_client.py`, 2 × 43 calls through `bpto.JevClient`, pinned `jev-1.13`)

First pass disagreed with the SDK on 5 QA items by up to .31 (all still on the right side of .5): the client had
sorted the state's keys, so Jev read `model_answer` before `question`. **Jev is sensitive to state key order**; the
client now sends it as given (and it is part of the cache key). Second pass: |client − SDK| max .02, mean .003 -
the same as Jev's own repeat spread. Rerun with the cache file (`runs/jev_smoke/`) makes 0 calls.
