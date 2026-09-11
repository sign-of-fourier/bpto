"""Scoped real-API smoke test: hard-capped at 12 model calls (~10 expected).

    python examples/smoke.py                 # Anthropic (ANTHROPIC_API_KEY or .env)
    python examples/smoke.py bedrock         # Amazon Nova via BedrockClient (AWS creds)

Plan: root + random(n=2) [1 call] -> evaluate 3 nodes x 3 examples [9 calls].
"""
import asyncio
import os
import sys
from pathlib import Path

from pydantic import BaseModel

from bpto import (Budget, Dataset, EventLog, LinearObjective, ModelConfig, Progress, Task, Tree, combine, evaluate,
                  exact_match, lineage, random, select, template_tokens, tree_text)

if Path(".env").exists():  # KEY=value lines, no export needed
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


class Names(BaseModel):
    name: str


RECORDS = [
    {"inputs": {"text": "There once was a man from Nantucket. His name was Charles."}, "answer": "Charles"},
    {"inputs": {"text": "Alice went to the market in Oslo and bought a fish."}, "answer": "Alice"},
    {"inputs": {"text": "The report was filed by Dr. Okafor on behalf of Acme Corp."}, "answer": "Okafor"},
]


def make_client(kind: str):
    if kind == "bedrock":
        from bpto import BedrockClient
        return BedrockClient(os.environ.get("BEDROCK_MODEL", "us.amazon.nova-micro-v1:0"),
                             region=os.environ.get("AWS_REGION", "us-east-1"), budget=Budget(max_calls=12))
    from bpto import AnthropicClient
    return AnthropicClient("claude-opus-5", max_concurrency=4, budget=Budget(max_calls=12))


async def main(kind: str):
    client = make_client(kind)
    task = Task(
        root="Extract the name of the person from the following text:\n{text}",
        description="extracts the name of the single person mentioned in a passage (surname only if that is all that is given)",
        dataset=Dataset.from_records(RECORDS),
        schema=Names,
        scorer=combine(exact_match(field="name"), template_tokens()),
        objective=LinearObjective(accuracy=1.0, template_tokens=-0.01),
        client=client,
        config=ModelConfig(max_tokens=200, effort="low") if kind != "bedrock" else ModelConfig(max_tokens=200),
        expander_config=ModelConfig(max_tokens=2000) if kind != "bedrock" else None,
    )
    tree = Tree(task)
    EventLog("smoke_events.jsonl", tree)
    Progress(tree)
    await tree.apply(random(n=2), select=select.root)
    await tree.apply(evaluate(), select=select.unevaluated)
    print()
    print(tree_text(tree, width=90))
    print()
    print("best lineage:\n" + lineage(tree, tree.best()))
    print("\nusage:", client.usage)
    for n in tree:
        for r in n.evaluation.per_example:
            if r.error or r.metrics.get("accuracy") == 0:
                print(f"  node {n.id} ex {r.example_id}: output={r.output!r} error={r.error}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "anthropic"))
