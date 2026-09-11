"""Run the compression search end to end.

    python -m tasks.compression.run --model claude-opus-5 --rounds 3
    python -m tasks.compression.run --provider openai --base-url http://localhost:8000/v1 --model llama-3.3-70b
    python -m tasks.compression.run --mock            # offline smoke run

Writes: <out>/tree.json (checkpoint), <out>/cache.jsonl (completions), <out>/pareto.png, <out>/report.txt
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path

from bpto import (CompletionCache, EventLog, ModelConfig, Progress, Stop, Tree, evaluate, guided, lineage, plot_tree,
                  random, run, select, step, successive_halving, tree_text)

from . import Entities, generate_dataset, load, make_task, shrinking_budget_objective, weighted_objective
from .report import pareto_table, plot_pareto


def _mock_client():
    from bpto import MockClient
    from bpto.ops import Variants
    from .data import FIRST

    def handler(prompt, cfg, schema):
        if schema is Variants:
            n = int(re.search(r"Return (\d+) distinct", prompt).group(1))
            base = re.search(r"<prompt>\n(.*?)\n</prompt>", prompt, re.S).group(1)
            words = base.split()
            return Variants(prompts=[" ".join(w for j, w in enumerate(words) if (j + i) % 3) for i in range(n)])
        text = prompt.rsplit("\n", 1)[-1]
        # a "model" that gets better with more instruction words and always keeps first names only
        skill = min(1.0, len(prompt.split()) / 40)
        found = [m.group(0) for m in re.finditer(r"\b(" + "|".join(FIRST) + r")\b(?: [A-Z][a-z]+)?", text)]
        return Entities(names=found if skill > 0.5 else found[:1])
    return MockClient(handler, max_concurrency=8)


def _client(args, cache):
    if args.mock:
        return _mock_client()
    if args.provider == "anthropic":
        from bpto import AnthropicClient
        return AnthropicClient(args.model, max_concurrency=args.concurrency, cache=cache)
    from bpto import OpenAICompatibleClient
    return OpenAICompatibleClient(args.model, base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY"),
                                  max_concurrency=args.concurrency, cache=cache)


async def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = None if args.mock else CompletionCache(out / "cache.jsonl")
    client = _client(args, cache)
    dataset = load(args.data) if args.data else generate_dataset(args.n_examples, seed=args.seed)
    objective = (shrinking_budget_objective(args.start_tokens, args.shrink) if args.constrained
                 else weighted_objective(args.token_weight))
    task = make_task(client, dataset, objective=objective,
                     config=None if args.mock else ModelConfig(max_tokens=512, effort="low"),
                     expander_config=None if args.mock else ModelConfig(max_tokens=4096, effort="high"))

    ck = out / "tree.json"
    tree = Tree.load(ck, task) if (args.resume and ck.exists()) else Tree(task)
    EventLog(out / "events.jsonl", tree)
    if not args.quiet:
        Progress(tree)
    def schedule(tree, rnd):
        # round 0: expand the root; later rounds: only the best k unexpanded nodes (keeps growth linear)
        parents = select.leaves if not tree.evaluated_nodes() else select.top_k(args.expand_k, among=select.unexpanded)
        return [
            step(random(n=args.n_random), parents),
            step(guided("Make it as short as possible without losing extraction precision or recall.", n=args.n_guided),
                 select.where(lambda n: n.origin.op == "random" and not n.evaluated, select.leaves)),
            *successive_halving([args.cheap_n, None], keep=0.5, seed=args.seed),
        ]
    res = await run(tree, schedule, stop=Stop(rounds=args.rounds, no_improvement_rounds=3), checkpoint=ck)

    report = [repr(tree), f"stopped: {res.stopped_because}", f"usage: {client.usage}", "",
              "Pareto front (F1 vs template tokens):", pareto_table(tree), "",
              f"best (scalar objective): {tree.best().evaluation.metrics}", str(tree.best().prompt), "",
              "lineage of best:", lineage(tree, tree.best())]
    (out / "report.txt").write_text("\n".join(report))
    (out / "tree.txt").write_text(tree_text(tree))
    print("\n".join(report))
    try:
        print("plots:", plot_pareto(tree, out / "pareto.png"), plot_tree(tree, out / "tree.png", color_metric="f1"))
    except ImportError:
        print("matplotlib not installed; skipped plots")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--data", help="JSONL file; default: synthetic")
    ap.add_argument("--n-examples", type=int, default=60)
    ap.add_argument("--cheap-n", type=int, default=12, help="examples for the first evaluation rung")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--n-guided", type=int, default=2)
    ap.add_argument("--expand-k", type=int, default=3, help="leaves expanded per round after round 0")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--start-tokens", type=float, default=80)
    ap.add_argument("--shrink", type=float, default=10)
    ap.add_argument("--token-weight", type=float, default=0.002)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/compression")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="no per-node progress on stderr")
    asyncio.run(main(ap.parse_args()))
