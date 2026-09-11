# bpto — Bayesian Prompt Tree Optimization

Python 3.12, async, Pydantic v2. Library in `bpto/`; downstream example task in `tasks/compression/`
(deliberately *outside* the library — it will move to its own repo; bpto only ever sees a `Task`).

## Commands

```bash
pip install -e .                 # required once so examples/ and tasks/ can import bpto
python -m pytest -q              # all tests are offline (MockClient / httpx.MockTransport / fake boto3)
python -m tasks.compression.run --mock --rounds 4     # offline end-to-end run -> runs/compression/
python -m tasks.ifbench.run --mock --rounds 2         # same for IFBench (real checkers need `pip install -e vendor/IFBench`)
python examples/smoke.py bedrock # LIVE, hard-capped at 12 calls (Budget). See "Live runs".
```

## Core model (keep these invariants)

- **Node lifecycle** `PROPOSED -> EVALUATED -> EXPANDED`. A tree may be several generations deep
  with nothing evaluated; `evaluate()` is an op like any other.
- **`tree.apply(op, select, chunk, checkpoint)` is the only mutation entry point.** Ops see
  `(tree, nodes)`; selectors are `tree -> list[Node]` (may be async). Chunking (`chunk=`) bounds
  scheduling; `ModelClient.max_concurrency` bounds in-flight requests. Tree *growth* is bounded
  only by selectors (`random(4)` on `leaves` is 4^d) — never by ops.
- **Scores are vectors.** `Scorer` -> per-example `Metrics` dict; `Objective(metrics, ctx)` ->
  `(scalar, feasible)` and is separate, cheap and re-runnable (constraints may depend on `ctx.depth`).
  Pareto is computed from the raw vectors. Don't collapse to a scalar inside a scorer.
- **Ancestor attribution** is the BO target: `DescendantValue(generations=k, agg)` (fixed
  generation) or `SubtreeValue(agg)` (any depth). `BOSelector` is *only a selector*; the tree
  and ops never know BO exists.
- **`random` and `guided` are the same operator** (`LLMExpander`) with different directives.
  Diversity comes from "return n distinct variants in one structured call", not temperature —
  current Anthropic models reject sampling params. `Pipeline([...])` strings ops into one expansion.
- **Provider boundary is `ModelClient._complete`.** Cache, semaphore, budget and usage live in the
  base class. Clients: `AnthropicClient`, `OpenAICompatibleClient` (httpx, no SDK),
  `BedrockClient` (boto3 Converse, schema-by-instruction + client-side JSON extraction), `MockClient`.
- **Completions are cached on `(prompt, config, schema)`.** Re-scoring with a new objective is free;
  `Dataset.sample(k, seed)` is a seeded-shuffle *prefix* so successive-halving rungs are nested.

## Conventions

- Tests never hit the network. Add a fake transport / fake client for any new provider code.
- Keep `bpto/` generic. Anything task-specific (schemas, partial-credit scorers, root prompts,
  data generators, reports) belongs under `tasks/<name>/`.
- `origin.op` / `origin.params` on every node must say how it was made — analysis depends on it.
- Failed examples are recorded (`ExampleResult.error`) and count as zero; `BudgetExceeded` must
  propagate (it is deliberately re-raised inside `evaluate`).
- Use `claude-opus-5` as the Anthropic default unless told otherwise.

## Live runs — cost discipline

The user does not want unscoped API spend. Before any live run: state the planned call count,
cap it with `Budget(max_calls=...)` on the client, and use a `CompletionCache` file so retries
are free. Credentials: `.env` (gitignored, `KEY=value` lines) currently holds a **Bedrock API key**
as `AWS_BEARER_TOKEN_BEDROCK` (the name boto3 reads natively; no remapping needed).
Working model: `us.amazon.nova-micro-v1:0`, region `us-east-1` (pass it explicitly; `~/.aws/config`
defaults to us-east-2). There is no Anthropic API key on this machine.

## Where things go

- `BACKLOG.md` - ordered next steps; keep it current when scope changes.
- `experiments/` - committed findings: one dir per run (NOTES.md, report, tree.txt, evaluations.jsonl,
  plot) plus `experiments/README.md` as the index of findings and standing conclusions. `runs/` holds the
  heavy, gitignored artifacts (cache, tree.json, events). After any live run, add both.

## Known gaps / caveats

- Mock-run plots (`--mock`) are plumbing checks only; the mock "model" is a regex.
- `tasks/compression` uses `generate_dataset()` (synthetic). `tasks/ifbench` is real data (300 rows,
  `tasks/ifbench/data/ifbench_test.jsonl`); its checkers come from github.com/allenai/IFBench, which is
  not on PyPI - clone to `vendor/` (gitignored) and `pip install -e` it. Tests skip if absent.
- `OpenAICompatibleClient.count_tokens` is a chars/4 estimate (no standard endpoint).
- Resume is step-granular: an interrupted step re-runs; safe for `evaluate` (cached) and for
  expansion under `leaves`/`unexpanded`, not under a custom selector that ignores node state.
- `VoyageEmbedder` / `AzureOpenAIEmbedder` / `BedrockEmbedder` / `AnthropicClient` are untested live.
  No embedding model has been used live yet; BO runs so far used `HashEmbedder`.
