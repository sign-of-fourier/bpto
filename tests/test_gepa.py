"""Offline tests for bpto.gepa: Pareto pool, sampling modes, reflective expander, minibatch gate, loop."""
import re

import pytest
from pydantic import BaseModel

from bpto import (Dataset, LinearObjective, MockClient, Prompt, Stop, Task, Tree, combine, evaluate, exact_match,
                  run, select)
from bpto.gepa import (ReflectiveExpander, beats_parent, candidates, example_scores, gepa, pareto_pool,
                       pareto_sample)
from bpto.ops import Variants
from bpto.tree import Evaluation, ExampleResult, Origin


class Out(BaseModel):
    answer: str


RECORDS = [{"id": f"e{i}", "inputs": {"q": f"q{i}"}, "answer": f"a{i}"} for i in range(4)]


def make_task(handler):
    return Task(root="Answer {q}", description="answers", dataset=Dataset.from_records(RECORDS), schema=Out,
                scorer=exact_match(field="answer"), objective=LinearObjective(accuracy=1.0), client=MockClient(handler))


def fake_eval(tree, node, correct: set[str]):
    """Attach an evaluation on the full dataset where `correct` example ids score 1."""
    per = [ExampleResult(example_id=ex.id, output="", metrics={"accuracy": float(ex.id in correct)}) for ex in tree.task.dataset]
    acc = len(correct) / len(RECORDS)
    node.evaluation = Evaluation(per_example=per, metrics={"accuracy": acc}, metrics_std={}, n=4, score=acc,
                                 feasible=True, dataset_ids=[ex.id for ex in tree.task.dataset])
    node.state = "evaluated"
    return node


def test_pareto_pool_winners_and_dominance():
    tree = Tree(make_task(lambda p, c, s: Out(answer="")))
    fake_eval(tree, tree.root, {"e0"})                                  # dominated by a
    a = fake_eval(tree, tree.add_child(tree.root, "A {q}", Origin(op="x")), {"e0", "e1"})
    b = fake_eval(tree, tree.add_child(tree.root, "B {q}", Origin(op="x")), {"e2"})
    c = fake_eval(tree, tree.add_child(tree.root, "C {q}", Origin(op="x")), set())   # wins nothing
    pool, wins = pareto_pool(tree)
    assert {n.id for n in pool} == {a.id, b.id}
    assert wins == {a.id: 2, b.id: 1}          # e3 is solved by nobody: not a win for anyone
    assert len(candidates(tree)) == 4
    assert example_scores(tree, a) == {"e0": 1.0, "e1": 1.0, "e2": 0.0, "e3": 0.0}


def test_pareto_pool_ties_share_wins():
    tree = Tree(make_task(lambda p, c, s: Out(answer="")))
    fake_eval(tree, tree.root, set())          # nothing solved anywhere -> everyone stays eligible, equal weight
    a = fake_eval(tree, tree.add_child(tree.root, "A {q}", Origin(op="x")), set())
    pool, wins = pareto_pool(tree)
    assert {n.id for n in pool} == {tree.root.id, a.id} and set(wins.values()) == {1}


def test_sampling_modes():
    tree = Tree(make_task(lambda p, c, s: Out(answer="")))
    fake_eval(tree, tree.root, {"e0"})
    a = fake_eval(tree, tree.add_child(tree.root, "A {q}", Origin(op="x")), {"e0", "e1", "e2"})
    b = fake_eval(tree, tree.add_child(tree.root, "B {q}", Origin(op="x")), {"e3"})
    assert [n.id for n in pareto_sample(1, mode="best")(tree)] == [a.id]
    assert {n.id for n in pareto_sample(5, mode="uniform", seed=1)(tree)} == {a.id, b.id}
    assert len(pareto_sample(5, mode="all", seed=1)(tree)) == 3
    picks = [pareto_sample(1, mode="weighted", seed=s)(tree)[0].id for s in range(200)]
    assert 0.6 < picks.count(a.id) / 200 < 0.9   # ∝ 3:1
    with pytest.raises(ValueError):
        pareto_sample(mode="nope")
    assert pareto_sample()(Tree(make_task(lambda p, c, s: Out(answer="")))) == []


def test_minibatch_only_nodes_are_not_candidates():
    tree = Tree(make_task(lambda p, c, s: Out(answer="")))
    fake_eval(tree, tree.root, {"e0"})
    child = fake_eval(tree, tree.add_child(tree.root, "A {q}", Origin(op="x")), {"e0", "e1"})
    child.evaluation.dataset_ids = ["e0", "e1"]   # minibatch only
    assert [n.id for n in candidates(tree)] == [tree.root.id]


def test_beats_parent_uses_parent_scores_on_same_ids():
    tree = Tree(make_task(lambda p, c, s: Out(answer="")))
    fake_eval(tree, tree.root, {"e0"})            # parent: e0=1, e1=0
    child = tree.add_child(tree.root, "A {q}", Origin(op="reflect"))
    child.evaluation = Evaluation(per_example=[], metrics={}, metrics_std={}, n=2, score=0.5, feasible=True,
                                  dataset_ids=["e0", "e1"])
    assert not beats_parent(tree, child)          # 0.5 == parent's 0.5 on {e0,e1}
    child.evaluation.score = 1.0
    assert beats_parent(tree, child)
    child.evaluation.dataset_ids = ["zz"]
    assert not beats_parent(tree, child)


async def test_reflective_expander_prompt_and_origin():
    seen = {}

    def handler(prompt, cfg, schema):
        if schema is Variants:
            seen["meta"] = prompt
            return Variants(prompts=["Better: answer {q} exactly"])
        return Out(answer="a0" if "q0" in prompt else "wrong")
    tree = Tree(make_task(handler))
    await tree.apply(evaluate())
    fb = lambda ex, r: f"expected {ex.answer} got {r.parsed['answer']}"
    kids = await tree.apply(ReflectiveExpander(fb, minibatch=2), select.root)
    assert len(kids) == 1 and "{q}" in kids[0].prompt.template
    assert "expected a1 got wrong" in seen["meta"] or "expected a2 got wrong" in seen["meta"]
    assert "### Example 2" in seen["meta"] and "### Example 3" not in seen["meta"]
    p = kids[0].origin.params
    assert p["source"] == tree.root.id and len(p["minibatch_ids"]) == 2 and "directive" not in p
    # failures come first in the minibatch
    assert "e0" not in p["minibatch_ids"]


async def test_reflect_on_unevaluated_child_uses_ancestor():
    tree = Tree(make_task(lambda p, c, s: Variants(prompts=["X {q}"]) if s is Variants else Out(answer="")))
    await tree.apply(evaluate())
    child = tree.add_child(tree.root, "mid {q}", Origin(op="x"))
    kids = await tree.apply(ReflectiveExpander(minibatch=1), lambda t: [child])
    assert kids[0].origin.params["source"] == tree.root.id


async def test_gepa_loop_end_to_end():
    """Each reflection adds one more solvable example; the loop should climb and keep the pool consistent."""
    def handler(prompt, cfg, schema):
        if schema is Variants:
            base = re.search(r"<prompt>\n(.*?)\n</prompt>", prompt, re.S).group(1)
            return Variants(prompts=[base + " +"])
        level = prompt.count("+")
        i = int(re.search(r"q(\d)", prompt).group(1))
        return Out(answer=f"a{i}" if i <= level else "wrong")
    tree = Tree(make_task(handler))
    res = await run(tree, gepa(minibatch=2, seed=3), stop=Stop(rounds=6))
    best = tree.best()
    assert best.score == 1.0 and best.depth >= 3
    # every pool member is evaluated on the full set; rejected children (if any) are minibatch-only
    for n in candidates(tree):
        assert set(n.evaluation.dataset_ids) == {"e0", "e1", "e2", "e3"}
    assert all(n.origin.op == "reflect" for n in tree.nodes.values() if n.depth > 0)
    assert any(s["step"].endswith("/full") for s in res.history)


async def test_gepa_rejects_children_that_do_not_improve():
    def handler(prompt, cfg, schema):
        if schema is Variants:
            base = re.search(r"<prompt>\n(.*?)\n</prompt>", prompt, re.S).group(1)
            return Variants(prompts=[base + " ~"])
        return Out(answer="a0" if "q0" in prompt else "wrong")   # nothing ever improves
    tree = Tree(make_task(handler))
    await run(tree, gepa(minibatch=2, seed=0), stop=Stop(rounds=4))     # round 0 = root evaluation
    assert [n.id for n in candidates(tree)] == [tree.root.id]
    assert sum(1 for n in tree.nodes.values() if n.depth > 0) == 3       # one child per round, all rejected
    assert all(n.parent_id == tree.root.id for n in tree.nodes.values() if n.depth > 0)
