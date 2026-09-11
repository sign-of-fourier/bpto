# Prompt compression

Goal: find the shortest prompt *template* that still extracts people's names accurately.
Two objectives — extraction quality (set-F1 with partial credit) and template length
(`template_tokens`: tokens of the template with placeholders stripped, counted once per
template through the provider's tokenizer) — combined either as a weighted sum or as a token
budget that tightens with tree depth. `prompt_tokens` (full rendered input) is also recorded,
for cost.

```bash
python -m tasks.compression.run --mock --rounds 4                      # offline smoke run
ANTHROPIC_API_KEY=... python -m tasks.compression.run --rounds 4       # Claude
python -m tasks.compression.run --provider openai --base-url http://localhost:8000/v1 --model llama-3.3-70b
python -m tasks.compression.run --rounds 8 --resume                    # continue a checkpointed run
python -m tasks.compression.run --constrained --start-tokens 80 --shrink 10
```

Outputs under `runs/compression/`: `tree.json` (checkpoint), `cache.jsonl` (completions),
`report.txt` (Pareto table + best prompt), `pareto.png`.

The schedule per round: `random(n)` on the best-k unexpanded nodes → `guided("as short as
possible…")` on the new children → successive halving (`cheap_n` examples for everyone, full set
for the top half). Round 0 expands the root.

As a library:

```python
from bpto import AnthropicClient, Tree, evaluate, guided, random, select
from tasks.compression import generate_dataset, load, make_task, shrinking_budget_objective

task = make_task(AnthropicClient("claude-opus-5"), load("data/train.jsonl"),   # or generate_dataset(60)
                 objective=shrinking_budget_objective(start_tokens=80, shrink_per_depth=10))
tree = Tree(task)
await tree.apply(random(n=4))
await tree.apply(guided("make it as short as possible without losing precision", n=3))
await tree.apply(evaluate(), select=select.unevaluated)
best = tree.best()                                      # feasible-only by default
front = tree.pareto({"f1": True, "template_tokens": False})
```

Data format (`train.jsonl`): `{"inputs": {"text": "..."}, "answer": ["Charles Okafor"]}`.
`generate_dataset(n, seed)` makes a seeded synthetic set with 0–3 people per passage plus
capitalised distractors (places, organisations, months, titles).

What lives here vs. in bpto: `Entities` (response schema), `set_f1` (scorer), the objectives, the
root prompt, the data, the report. bpto only sees a `Task`.
