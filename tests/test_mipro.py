import asyncio
import re

from bpto import Dataset, MockClient, Task, Tree, exact_match, select, step, run, Stop
from bpto.mipro import CategoricalTPE, GroundedProposer, dataset_summary
from bpto.ops import Variants


def test_tpe_prefers_the_candidate_with_high_scores():
    tpe = CategoricalTPE(4, seed=1)
    first = {tpe.suggest() for _ in range(50)}
    assert first == {0, 1, 2, 3}  # startup: every candidate once (suggest without observe keeps offering untried)
    for c in range(4):
        tpe.observe(c, 0.2)
    for _ in range(10):
        tpe.observe(2, 0.9); tpe.observe(1, 0.1)
    assert tpe.suggest() == 2
    assert max(tpe.scores(), key=tpe.scores().get) == 2


def test_grounded_proposer_rotates_tips_and_keeps_placeholders():
    seen = []

    def handler(prompt, cfg, schema):
        seen.append(prompt)
        assert schema is Variants
        n = int(re.search(r"Propose (\d+)", prompt).group(1))
        return Variants(prompts=[f"v{i}-{len(seen)} {{x}}" for i in range(n)] + ["missing placeholder"])
    client = MockClient(handler)
    data = Dataset.from_records([{"inputs": {"x": "a" * 500}, "answer": 1}, {"inputs": {"x": "b"}, "answer": 2}])
    task = Task(root="do {x}", dataset=data, scorer=exact_match(), objective=lambda m, c: (m.get("accuracy", 0), True),
                client=client, description="t")
    tree = Tree(task)
    prop = GroundedProposer(dataset_summary(data, k=2, max_chars=10), n=2, calls=3, seed=0)
    kids = asyncio.run(tree.apply(prop, select.root))
    assert len(kids) == 6 and all(k.prompt.placeholders == ("x",) for k in kids)
    assert all(k.origin.op == "mipro_propose" for k in kids)
    assert "expected answer: 1" in seen[0] and "..." in seen[0]  # summary rendered and truncated
    tips = {re.search(r"Tip: (.*)", p).group(1) if "Tip:" in p else "" for p in seen}
    assert len(tips) == 3
