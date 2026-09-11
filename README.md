# bpto — Bayesian Prompt Tree Optimization

Optimize a prompt by growing a tree of rewrites, scoring nodes on a training set, and
choosing where to expand next — first by hand, later with Bayesian optimization over
prompt embeddings.

- A **node** is a prompt (plus an optional per-node model config).
- **Ops** act on sets of nodes: `random(n)` and `guided(directive, n)` propose children,
  `evaluate()` runs the training set. You compose them: `[random, guided, random, evaluate]`.
- **Selectors** pick the nodes an op applies to: `leaves`, `unevaluated`, `top_k`, `pareto`, …
  BO plugs in as one more selector.
- **Scores are vectors.** A `Scorer` returns per-example metrics; an `Objective` turns the
  aggregated vector into a scalar (linear, constrained-with-depth, …) and is cheap to swap.
- **Ancestor attribution.** `DescendantValue(generations=2, agg=max)` values a node by its
  best grandchild — the target BO learns from.

## Example: question answering with an LLM judge

```python
import asyncio
from pydantic import BaseModel
from bpto import (AnthropicClient, Dataset, LinearObjective, ModelConfig, Task, Tree,
                  combine, evaluate, guided, llm_judge, random, select, token_count)

class Answer(BaseModel):
    answer: str

task = Task(
    root="Answer the question using only the context.\n\nContext:\n{context}\n\nQuestion: {question}",
    description="answers a question from a supplied context passage",
    dataset=Dataset.from_jsonl("qa_train.jsonl"),   # {"inputs": {"context": ..., "question": ...}, "answer": ...}
    schema=Answer,
    scorer=combine(
        llm_judge("Is the model answer factually equivalent to the reference answer?",
                  client=AnthropicClient("claude-opus-5")),          # judge model
        token_count(),
    ),
    objective=LinearObjective(judge=1.0, prompt_tokens=-0.001),
    client=AnthropicClient("claude-opus-5", max_concurrency=16),    # model under optimisation
    config=ModelConfig(max_tokens=512, effort="low"),
)

async def main():
    tree = Tree(task)
    await tree.apply(random(n=4),                       select=select.leaves)
    await tree.apply(guided("be more precise", n=3),    select=select.leaves)
    await tree.apply(evaluate(),                        select=select.unevaluated, chunk=32)
    print(tree.best().prompt)
    print(tree.pareto({"judge": True, "prompt_tokens": False}))

asyncio.run(main())
```

Swap `AnthropicClient` for `OpenAICompatibleClient("llama-3.3-70b", base_url="http://localhost:8000/v1")`
to run against vLLM/Ollama/OpenRouter/OpenAI; or subclass `ModelClient` and implement `_complete`.

## Search loop with ancestor attribution (BO-ready)

```python
from bpto import DescendantValue
from bpto.bo import BOSelector, VoyageEmbedder, GPR, EI

bo = BOSelector(VoyageEmbedder(), GPR(), EI(), value=DescendantValue(generations=2, agg=max))
for _ in range(10):
    await tree.apply(random(n=4),              select=bo.top(k=2, among=select.unexpanded))
    await tree.apply(guided("more accurate", n=3), select=select.leaves)
    await tree.apply(evaluate(),               select=select.unevaluated)
```

An expansion can itself be a pipeline — several ops applied in sequence to the subtree they
create, treated as *one* expansion of the chosen node. `random` and `guided` are the same
operator with different directives; a pipeline just strings purposes together:

```python
from bpto import Pipeline, SubtreeValue
expand = Pipeline([random(3), guided("be more precise", 2), guided("be shorter", 2), evaluate()])
bo = BOSelector(VoyageEmbedder(), GPR(), EI(), value=SubtreeValue(max))   # best leaf of the subtree
for _ in range(10):
    await tree.apply(expand, select=bo.top(k=2, among=select.unexpanded))
```

`BOSelector` fits `embedding(node) → value(node)` on every node with a value, ranks the
candidates by acquisition, and returns the top k. The tree and ops never know BO exists.

## Running a schedule with stop conditions, checkpoint and resume

```python
from bpto import Stop, run, step, successive_halving

schedule = [
    step(random(n=4),                       select.leaves),
    step(guided("be more precise", n=3),    select.leaves),
    *successive_halving([8, 32, None], keep=0.5),   # cheap eval on all, full eval on the best
]
res = await run(tree, schedule, stop=Stop(rounds=10, no_improvement_rounds=3), checkpoint="tree.json")
print(res.stopped_because, res.history[-1])

# later / after a crash: continues from the step after the last completed one
tree = Tree.load("tree.json", task)
res = await run(tree, schedule, stop=Stop(rounds=10), checkpoint="tree.json")
```

`Stop` also takes `max_nodes`, `max_depth`, `max_seconds`, `until=fn(tree)`; a client `Budget`
(calls / tokens) stops the run too. `successive_halving` samples are nested, so each rung only
pays for the new examples. Pair the checkpoint with `CompletionCache("cache.jsonl")` so a resumed
run also re-uses completions.

## Watching a run

```python
from bpto import EventLog, Progress, tree_text, plot_tree, lineage
EventLog("events.jsonl", tree)      # one JSONL line per proposed/expanded/evaluated node, with usage so far
Progress(tree)                      # one stderr line per evaluation: score, best so far, calls
...
print(tree_text(tree))              # indented tree, best children first
print(lineage(tree, tree.best()))   # root -> best, with scores
plot_tree(tree, "tree.png", color_metric="accuracy")
```

## Layout

```
bpto/
  prompt.py   data.py   metrics.py   task.py
  llm/        ModelClient (global concurrency, cache, budget), AnthropicClient, OpenAICompatibleClient, MockClient
  scoring.py  Scorer / Objective protocols; exact_match, token_count, llm_judge, Linear/ConstrainedObjective, pareto_front
  tree.py     Node lifecycle PROPOSED → EVALUATED → EXPANDED; ancestors / descendants(generations=k); save/load
  ops.py      random, guided, evaluate (chunked), Pipeline
  search.py   run(tree, schedule, Stop, checkpoint), successive_halving
  observe.py  EventLog (JSONL), Progress (stderr), tree_text / tree_dot / plot_tree / lineage
  select.py   leaves, unevaluated, unexpanded, depth, top_k, pareto, union
  value.py    DescendantValue (fixed generation), SubtreeValue (any depth, optional pipeline-only attribution)
  bo/         BOSelector; GPR (numpy); EI / UCB / Thompson; Voyage / OpenAI-compatible / Hash embedders; config_features
tasks/compression/   example downstream task (will move to its own repo) — see its README
tasks/ifbench/        IFBench instruction-following task (real data, code checkers) — see its README
```

Tests: `python -m pytest -q`. Real-API smoke test: `examples/names.py`.

## Notes

- Diversity in `random`/`guided` comes from asking for *n distinct variants per call*, not
  temperature (current Anthropic models reject sampling params; `temperature` is still honoured
  by clients whose models accept it). `calls=k` repeats for more spread.
- `max_concurrency` bounds in-flight requests globally; `apply(chunk=)` bounds how many nodes
  are scheduled at once. Tree growth itself is bounded only by the selectors you use.
- Completions are cached on `(prompt, config, schema)`; re-scoring with a new objective is free.
