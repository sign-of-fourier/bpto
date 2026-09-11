"""The sample flow from the design discussion, against the real API.

    ANTHROPIC_API_KEY=... python examples/names.py
"""
import asyncio

from pydantic import BaseModel

from bpto import (AnthropicClient, CompletionCache, Dataset, LinearObjective, ModelConfig, Task, Tree,
                  combine, evaluate, exact_match, guided, random, select, token_count)


class Names(BaseModel):
    name: str


RECORDS = [
    {"inputs": {"text": "There once was a man from Nantucket. His name was Charles."}, "answer": "Charles"},
    {"inputs": {"text": "Alice went to the market and bought a fish."}, "answer": "Alice"},
    {"inputs": {"text": "The report was filed by Dr. Okafor on Tuesday."}, "answer": "Okafor"},
]


async def main():
    task = Task(
        root="Extract the name of the person from the following text:\n{text}",
        description="extracts the name of the person mentioned in a passage",
        dataset=Dataset.from_records(RECORDS),
        schema=Names,
        scorer=combine(exact_match(field="name"), token_count()),
        objective=LinearObjective(accuracy=1.0, prompt_tokens=-0.01),
        client=AnthropicClient("claude-opus-5", max_concurrency=8, cache=CompletionCache("cache.jsonl")),
        config=ModelConfig(max_tokens=256, effort="low"),           # cheap evaluation calls
        expander_config=ModelConfig(max_tokens=4096, effort="high"),  # thoughtful rewrites
    )
    tree = Tree(task)
    await tree.apply(random(n=3), select=select.leaves)
    await tree.apply(guided("make it more succinct", n=2), select=select.leaves)
    await tree.apply(evaluate(), select=select.unevaluated, chunk=16)

    print(tree, task.client.usage)
    for n in sorted(tree, key=lambda n: n.score, reverse=True)[:5]:
        print(f"{n.score:+.3f} depth={n.depth} {n.origin.op:8s} {n.evaluation.metrics} | {n.prompt.template!r}")
    print("pareto:", [n.id for n in tree.pareto({"accuracy": True, "prompt_tokens": False})])
    tree.save("tree.json")


if __name__ == "__main__":
    asyncio.run(main())
