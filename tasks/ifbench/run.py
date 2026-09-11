"""Run the IFBench prompt search end to end.

    python -m tasks.ifbench.run --mock --rounds 3                         # offline plumbing check
    python -m tasks.ifbench.run --provider bedrock --n-train 40 --cheap-n 10 --rounds 2 --max-calls 400
    python -m tasks.ifbench.run --provider anthropic --model claude-opus-5

Data: 300 IFBench prompts, split with --n-train / --seed into train (searched over) and held-out
(only the final best and root are scored on it, once). Writes tree.json, cache.jsonl, events.jsonl,
report.txt, tree.txt, tree.png under --out.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path

from bpto import (Budget, CompletionCache, EventLog, ModelConfig, Progress, Stop, Tree, evaluate, guided, lineage,
                  plot_tree, random, run, select, step, successive_halving, tree_text)

from . import available, compact_objective, load, load_or_fetch, loose_objective, make_task, strict_objective


def _mock():
    """Offline stand-in: variants drop words; the 'model' passes a constraint iff the template mentions its family."""
    from bpto import MockClient
    from bpto.ops import Variants

    def handler(prompt, cfg, schema):
        if schema is Variants:
            n = int(re.search(r"Return (\d+) distinct", prompt).group(1))
            base = re.search(r"<prompt>\n(.*?)\n</prompt>", prompt, re.S).group(1)
            words = base.split()
            return Variants(prompts=[" ".join(w for j, w in enumerate(words) if (j + i) % 4) for i in range(n)])
        return "MOCK RESPONSE\n" + prompt[:200]

    def checker(response, ids, kwargs, instruction=""):
        # template text is embedded in the response by the mock handler
        ok = [cid.split(":")[0] in response.lower() for cid in ids]
        return ok, ok
    return MockClient(handler, max_concurrency=8), checker


def _client(args, cache, budget):
    kw = dict(max_concurrency=args.concurrency, cache=cache, budget=budget)
    if args.provider == "bedrock":
        from bpto import BedrockClient
        return BedrockClient(args.model or os.environ.get("BEDROCK_MODEL", "us.amazon.nova-micro-v1:0"),
                             region=os.environ.get("AWS_REGION", "us-east-1"), **kw)
    if args.provider == "anthropic":
        from bpto import AnthropicClient
        return AnthropicClient(args.model or "claude-opus-5", **kw)
    from bpto import OpenAICompatibleClient
    return OpenAICompatibleClient(args.model or "gpt-4o-mini", base_url=args.base_url,
                                  api_key=os.environ.get("OPENAI_API_KEY"), **kw)


def _load_env():
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


async def main(args):
    _load_env()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.mock:
        client, checker = _mock()
    else:
        if not available():
            raise SystemExit("ifbench checkers not installed: pip install -e <clone of github.com/allenai/IFBench>")
        from .checkers import check as checker
        client = _client(args, CompletionCache(out / "cache.jsonl"), Budget(max_calls=args.max_calls))

    full = load(args.data) if args.data else load_or_fetch()
    train, held = full.split(args.n_train / len(full), seed=args.seed)
    objective = {"strict": strict_objective, "loose": loose_objective,
                 "compact": lambda: compact_objective(args.token_weight)}[args.objective]()
    task = make_task(client, train, objective=objective, checker=checker,
                     config=None if args.mock else ModelConfig(max_tokens=args.max_tokens),
                     expander_config=None if args.mock else ModelConfig(max_tokens=4096))

    ck = out / "tree.json"
    tree = Tree.load(ck, task) if (args.resume and ck.exists()) else Tree(task)
    EventLog(out / "events.jsonl", tree)
    if not args.quiet:
        Progress(tree)

    def schedule(tree, rnd):
        parents = select.leaves if not tree.evaluated_nodes() else select.top_k(args.expand_k, among=select.unexpanded)
        return [
            step(random(n=args.n_random), parents),
            step(guided("Make the instructions more precise about obeying unusual constraints literally, "
                        "without making the prompt much longer.", n=args.n_guided),
                 select.where(lambda n: n.origin.op == "random" and not n.evaluated, select.leaves)),
            *successive_halving([args.cheap_n, None], keep=0.5, seed=args.seed),
        ]
    res = await run(tree, schedule, stop=Stop(rounds=args.rounds, no_improvement_rounds=3), checkpoint=ck)

    best = tree.best()
    report = [repr(tree), f"stopped: {res.stopped_because}", f"usage: {client.usage}", "",
              f"train  root: {tree.root.evaluation.metrics if tree.root.evaluation else None}",
              f"train  best: {best.evaluation.metrics}", "", str(best.prompt), "",
              "lineage of best:", lineage(tree, best)]
    if args.holdout:
        # one pass each on the held-out split: the only numbers that mean anything
        held_scores = {}
        for label, node in (("root", tree.root), ("best", best)):
            ev = await evaluate(dataset=held.sample(args.holdout, seed=args.seed)).score(tree, node)
            held_scores[label] = {k: round(v, 3) for k, v in ev.metrics.items()}
        report += ["", f"held-out ({args.holdout} examples): {held_scores}"]
    (out / "report.txt").write_text("\n".join(report))
    (out / "tree.txt").write_text(tree_text(tree))
    print("\n".join(report))
    try:
        print("plot:", plot_tree(tree, out / "tree.png", color_metric="strict"))
    except ImportError:
        print("matplotlib not installed; skipped plot")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["bedrock", "anthropic", "openai"], default="bedrock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--data", help="JSONL file; default: fetch allenai/IFBench_test")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--cheap-n", type=int, default=10, help="examples for the first evaluation rung")
    ap.add_argument("--holdout", type=int, default=0, help="score root and best on this many held-out examples")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--n-guided", type=int, default=2)
    ap.add_argument("--expand-k", type=int, default=2)
    ap.add_argument("--objective", choices=["strict", "loose", "compact"], default="strict")
    ap.add_argument("--token-weight", type=float, default=0.001)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-calls", type=int, default=400, help="hard budget on model calls")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ifbench")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    asyncio.run(main(ap.parse_args()))
