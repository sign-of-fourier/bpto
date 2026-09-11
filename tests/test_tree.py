import pytest

from bpto import (ConstrainedObjective, DescendantValue, LinearObjective, NodeState, Prompt, Tree, evaluate,
                  guided, random, select)
from bpto.bo import BOSelector
from bpto.scoring import ObjectiveContext, pareto_front


def test_prompt_placeholders_and_render():
    p = Prompt(template="Hi {name}, {name} again {x}")
    assert p.placeholders == ("name", "x")
    assert p.render(name="a", x=1) == "Hi a, a again 1"


async def test_sample_flow(task):
    tree = Tree(task)
    await tree.apply(random(n=4), select=select.leaves)
    assert len(tree) == 5 and tree.root.state == NodeState.EXPANDED
    await tree.apply(guided("make it more succinct", n=3), select=select.leaves)
    assert len(tree) == 5 + 12
    assert all(n.origin.op == "guided" for n in tree.descendants(tree.root, generations=2))
    await tree.apply(evaluate(), select=select.unevaluated, chunk=4)
    assert all(n.evaluated for n in tree)
    # only the 'succinct' (ONLY) prompts get 'Bob' right -> 4/4; others 3/4
    best = tree.best()
    assert best.depth == 2 and best.evaluation.metrics["accuracy"] == 1.0
    assert tree.root.evaluation.metrics["accuracy"] == 0.75
    # cache: root prompt re-evaluated for free
    calls_before = task.client.usage.calls
    await tree.apply(evaluate(), select=select.root)
    assert task.client.usage.calls == calls_before


async def test_navigation(task):
    tree = Tree(task)
    await tree.apply(random(n=2))
    await tree.apply(random(n=2))
    gc = tree.descendants(tree.root, generations=2)
    assert len(gc) == 4
    assert tree.ancestor(gc[0], 2) is tree.root
    assert [a.depth for a in tree.ancestors(gc[0])] == [1, 0]
    assert len(tree.descendants(tree.root)) == 6
    assert tree.descendants(tree.root, generations=5) == []


async def test_descendant_value_and_top_k(task):
    tree = Tree(task)
    await tree.apply(random(n=2))
    await tree.apply(guided("succinct", n=2))
    await tree.apply(evaluate(), select=select.unevaluated)
    v = DescendantValue(generations=2, agg=max)
    assert v(tree.root, tree) == max(n.score for n in tree.descendants(tree.root, 2))
    assert DescendantValue(1, max)(tree.leaves[0], tree) is None
    top = select.top_k(1, among=select.depth(1), value=DescendantValue(1, max))(tree)
    assert len(top) == 1 and top[0].depth == 1


def test_constrained_objective_tightens_with_depth():
    obj = ConstrainedObjective(LinearObjective(accuracy=1.0), metric="prompt_tokens", bound=lambda c: 100 - 20 * c.depth)
    assert obj({"accuracy": 1, "prompt_tokens": 70}, ObjectiveContext(depth=0)) == (1.0, True)
    assert obj({"accuracy": 1, "prompt_tokens": 70}, ObjectiveContext(depth=2))[1] is False


def test_pareto():
    pts = {"a": {"acc": 1.0, "tok": 10}, "b": {"acc": 0.5, "tok": 5}, "c": {"acc": 0.5, "tok": 20}}
    assert sorted(pareto_front(pts, {"acc": True, "tok": False})) == ["a", "b"]


async def test_save_load(task, tmp_path):
    tree = Tree(task)
    await tree.apply(random(n=2))
    await tree.apply(evaluate(), select=select.unevaluated)
    tree.save(tmp_path / "t.json")
    t2 = Tree.load(tmp_path / "t.json", task)
    assert len(t2) == len(tree) and t2.best().id == tree.best().id
    assert t2.root.state == NodeState.EXPANDED


async def test_bo_selector_protocol(task):
    """BO with trivial embedder/surrogate/acquisition: verifies the wiring, not the maths."""
    class Emb:
        async def embed(self, texts):
            return [[float(len(t))] for t in texts]

    class Sur:
        def fit(self, X, y): self.m = sum(y) / len(y)
        def predict(self, X): return [self.m + x[0] * 1e-3 for x in X], [1.0] * len(X)

    tree = Tree(task)
    await tree.apply(random(n=3))
    await tree.apply(guided("succinct", n=2))
    await tree.apply(evaluate(), select=select.unevaluated)
    bo = BOSelector(Emb(), Sur(), lambda m, v, b: m, value=DescendantValue(1, max))
    chosen = await tree.apply(random(n=2), select=bo.top(k=2, among=select.leaves))
    assert len(chosen) == 4
    assert all(n.embedding is not None for n in tree if n.depth <= 2)
