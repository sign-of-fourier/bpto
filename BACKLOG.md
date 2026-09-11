# Backlog (in order)

Goal: compare BO-over-a-prompt-tree against GEPA-style evolutionary + Pareto-pool selection at equal rollout
budget, with the same mutation operators available to both. Findings go in `experiments/`.

1. **Feedback-aware `guided` for IFBench (user-space, `tasks/ifbench/`).** Subclass `LLMExpander`; build
   the directive from `node.evaluation.per_example` (failed constraint ids, a few failing outputs). For
   unevaluated children inside a `Pipeline`, take feedback from `tree.ancestor(node, 1)`. Record what was
   fed in via `origin.params`. Note: feedback text is in the meta-prompt, so it is part of the cache key.
2. **GEPA-style selection in user-space.** A selector that keeps the per-example Pareto pool (a candidate
   survives if it is best on at least one train example) and samples parents from it; the evolutionary loop
   is just a `run()` schedule. Needs per-example scores, which `Evaluation.per_example` already has.
3. **Live embedder smoke.** `BedrockEmbedder` (Titan V2) on <= 10 texts, so `BOSelector` can run with real
   embeddings. Then one BO-selected round on IFBench with a small cap, to exercise `BOSelector` live.
4. **Equal-budget comparison harness.** One script, arms: (a) root only; (b) best-of-N `random`;
   (c) tree + greedy `top_k`; (d) tree + `BOSelector`; (e) tree + feedback-aware guided; (f) GEPA-style
   Pareto pool + feedback; (g) BO + feedback. Same `Budget` per arm, 3+ seeds, held-out >= 150,
   report mean +- SE. Output a table into `experiments/<date>-<task>-compare/`.
5. **A task with headroom.** HotpotQA distractor setting as a single-prompt task (`tasks/hotpotqa/`,
   `{question, context}` -> answer, EM + F1). GEPA/MIPROv2 show +13-20 pts there; IFBench cannot separate
   selection strategies. Loader from HF, ~1.5-2k input tokens per example - budget accordingly.
6. **Stronger models.** Nova Lite / Pro via the same `--provider bedrock --model ...`; Anthropic when a key
   exists. Check whether prompts found on Micro transfer (GEPA reports cross-model transfer).
7. **Multi-prompt programs** (only if HoVer / PUPA become targets): `Node.prompt` as a mapping of named
   templates, expanders mutate one slot, `Task.run(prompts, example)` supplied by the user.
8. Housekeeping: live checks for `AnthropicClient`, `AzureOpenAIEmbedder`, `VoyageEmbedder`; real token
   counting for `OpenAICompatibleClient`; make `Stop(no_improvement_rounds)` tolerant of flat landscapes
   (currently it just stops; should it widen instead?).
