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


def test_pool_on_raw_metric_ignores_constrained_scalar():
    """A constrained 'minimise tokens' objective gives every example the same scalar; `metric=` restores per-example info."""
    from bpto import ConstrainedObjective
    tree = Tree(make_task(lambda p, c, s: Out(answer="")))
    tree.task.objective = ConstrainedObjective(LinearObjective(tokens=-1.0), metric="accuracy", bound=0.5, sense=">=")
    fake_eval(tree, tree.root, {"e0"})
    a = fake_eval(tree, tree.add_child(tree.root, "A {q}", Origin(op="x")), {"e1", "e2"})
    for n in (tree.root, a):
        for r in n.evaluation.per_example:
            r.metrics["tokens"] = 10.0
    assert set(example_scores(tree, a).values()) == {-10.0}                     # scalar carries nothing per example
    assert example_scores(tree, a, "accuracy") == {"e0": 0.0, "e1": 1.0, "e2": 1.0, "e3": 0.0}
    pool, wins = pareto_pool(tree, metric="accuracy")
    assert wins == {tree.root.id: 1, a.id: 2}
    assert len(pareto_sample(1, mode="weighted", seed=0, metric="accuracy")(tree)) == 1


async def test_reflective_expander_passed_hook_orders_failures_first():
    seen = {}

    def handler(prompt, cfg, schema):
        if schema is Variants:
            seen["prompt"] = prompt
            return Variants(prompts=["Reply to {q}"])
        return Out(answer="")
    tree = Tree(make_task(handler))
    fake_eval(tree, tree.root, {"e0", "e1", "e2"})     # objective says e3 failed...
    exp = ReflectiveExpander(minibatch=1, passed=lambda ex, r: ex.id != "e1")   # ...but the task says e1 is the failure
    await exp.apply(tree, [tree.root])
    assert "q: q1" in seen["prompt"] and "q: q3" not in seen["prompt"]


# ---- gate rungs, reflector rows, context hook ----------------------------------------

from bpto.gepa import gate_steps, minibatch_for, paired_change, unclear

WIDE = [{"id": f"w{i}", "inputs": {"q": f"q{i}"}, "answer": f"a{i}"} for i in range(12)]


def wide_task(handler):
    return Task(root="Answer {q}", description="answers", dataset=Dataset.from_records(WIDE), schema=Out,
                scorer=exact_match(field="answer"), objective=LinearObjective(accuracy=1.0), client=MockClient(handler))


def test_paired_change_and_ties():
    tree = Tree(make_task(lambda p, c, s: Out(answer="")))
    fake_eval(tree, tree.root, {"e0", "e1"})
    swap = fake_eval(tree, tree.add_child(tree.root, "A {q}", Origin(op="reflect")), {"e0", "e2"})
    same = fake_eval(tree, tree.add_child(tree.root, "B {q}", Origin(op="reflect")), {"e0", "e1"})
    better = fake_eval(tree, tree.add_child(tree.root, "C {q}", Origin(op="reflect")), {"e0", "e1", "e2"})
    assert paired_change(tree, swap) == (1, 1) and unclear(tree, swap, margin=0)
    assert paired_change(tree, same) == (0, 0) and not unclear(tree, same)          # inert: more rows can't help
    assert paired_change(tree, better) == (1, 0) and not unclear(tree, better, margin=0)
    assert unclear(tree, better, margin=1)                                          # a one-row win is not evidence
    assert paired_change(tree, tree.root) is None
    half = tree.add_child(tree.root, "D {q}", Origin(op="reflect"))   # continuous scores: +0.5 and -0.5 cancel
    half.evaluation = Evaluation(per_example=[ExampleResult(example_id="e0", output="", metrics={"accuracy": 0.5}),
                                              ExampleResult(example_id="e2", output="", metrics={"accuracy": 0.5})],
                                 metrics={}, metrics_std={}, n=2, score=0.5, feasible=True, dataset_ids=["e0", "e2"])
    assert paired_change(tree, half) == (1, 1) and unclear(tree, half, margin=0)
    half.evaluation.per_example[1].metrics["accuracy"] = 0.25
    half.evaluation.score = 0.375
    assert not unclear(tree, half, margin=0) and unclear(tree, half, margin=0.25)   # fixed == broke, net -0.25


def test_gate_steps_validates_rungs():
    ds = Dataset.from_records(WIDE)
    with pytest.raises(ValueError):
        gate_steps(1, ds, 4, 0, extend=(4,))
    with pytest.raises(ValueError):
        gate_steps(1, ds, 4, 0, extend=(8, 12))                     # 12 = the full set: would skip the gate
    assert [s.name for s in gate_steps(1, ds, 4, 0)] == ["gepa/r1/minibatch", "gepa/r1/full"]
    assert len(gate_steps(1, Dataset.from_records(RECORDS), 8, 0)) == 2   # minibatch >= |full| without rungs: as before


def _swap_handler(root_ok: set[int], child_ok: set[int]):
    def handler(prompt, cfg, schema):
        if schema is Variants:
            return Variants(prompts=["Answer {q} +"])
        i = int(re.search(r"q(\d+)", prompt).group(1))
        ok = child_ok if "+" in prompt else root_ok
        return Out(answer=f"a{i}" if i in ok else "wrong")
    return handler


async def test_extend_resolves_a_tie_on_a_nested_rung():
    ds, seed = Dataset.from_records(WIDE), 5
    mb2 = [int(ex.id[1:]) for ex in minibatch_for(1, ds, 2, seed)]
    mb6 = [int(ex.id[1:]) for ex in minibatch_for(1, ds, 6, seed)]
    assert mb6[:2] == mb2                                           # rungs are nested
    # on the 2-row batch the child fixes one row and breaks the other; the next four rows only the child gets right
    tree = Tree(wide_task(_swap_handler({mb2[0]}, {mb2[1], *mb6[2:]})))
    res = await run(tree, gepa(minibatch=2, extend=(6,), seed=seed), stop=Stop(rounds=2))
    child = next(n for n in tree.nodes.values() if n.origin.op == "reflect")
    assert set(child.evaluation.dataset_ids) == {ex.id for ex in ds}   # accepted after the rung
    assert "gepa/r1/extend6" in [s["step"] for s in res.history]
    assert tree.meta["gate"][child.id] == {"round": 1, "outcome": "passed", "rows": 6, "fixed": 5, "broke": 1}
    # without the rung the same tie is rejected
    tree = Tree(wide_task(_swap_handler({mb2[0]}, {mb2[1], *mb6[2:]})))
    await run(tree, gepa(minibatch=2, seed=seed), stop=Stop(rounds=2))
    child = next(n for n in tree.nodes.values() if n.origin.op == "reflect")
    assert len(child.evaluation.dataset_ids) == 2


async def test_extend_skips_inert_children():
    tree = Tree(wide_task(_swap_handler({0, 1, 2}, {0, 1, 2})))      # the rewrite changes no answer
    await run(tree, gepa(minibatch=2, extend=(6,), seed=5), stop=Stop(rounds=2))
    child = next(n for n in tree.nodes.values() if n.origin.op == "reflect")
    assert len(child.evaluation.dataset_ids) == 2
    assert tree.meta["gate"][child.id]["outcome"] == "inert"


async def test_extend_catches_a_narrow_minibatch_win():
    """Run 6's false pass: +1 row on the minibatch, worse on more rows. With a margin it is checked, not accepted."""
    ds, seed = Dataset.from_records(WIDE), 5
    mb6 = [int(ex.id[1:]) for ex in minibatch_for(1, ds, 6, seed)]
    root_ok, child_ok = set(mb6[2:]), {mb6[0]}                      # child +1 on the 2 rows, -4 on the next 4
    tree = Tree(wide_task(_swap_handler(root_ok, child_ok)))
    await run(tree, gepa(minibatch=2, extend=(6,), seed=seed), stop=Stop(rounds=2))
    child = next(n for n in tree.nodes.values() if n.origin.op == "reflect")
    assert len(child.evaluation.dataset_ids) == 6
    assert tree.meta["gate"][child.id] == {"round": 1, "outcome": "rejected", "rows": 6, "fixed": 1, "broke": 4}
    # without rungs (GEPA's gate) the same child buys a full evaluation; with margin=0 only exact ties are extended
    for kw in ({}, {"extend": (6,), "margin": 0}):
        tree = Tree(wide_task(_swap_handler(root_ok, child_ok)))
        await run(tree, gepa(minibatch=2, seed=seed, **kw), stop=Stop(rounds=2))
        child = next(n for n in tree.nodes.values() if n.origin.op == "reflect")
        assert len(child.evaluation.dataset_ids) == len(ds)
    assert "gate" in tree.meta and "gate" not in Tree(wide_task(_swap_handler(root_ok, child_ok))).meta


async def test_reflect_rows_and_context_hook():
    seen = []

    def handler(prompt, cfg, schema):
        if schema is Variants:
            seen.append(prompt)
            return Variants(prompts=["Answer {q} +"])
        return Out(answer="wrong")

    def confusion(tree, src):
        return f"SUMMARY of {src.id}: all wrong"
    tree = Tree(wide_task(handler))
    await run(tree, gepa(minibatch=4, reflect_rows=1, context=confusion, seed=0), stop=Stop(rounds=2))
    child = next(n for n in tree.nodes.values() if n.origin.op == "reflect")
    assert len(child.origin.params["minibatch_ids"]) == 1 and child.origin.params["context"] == "confusion"
    assert f"SUMMARY of {tree.root.id}: all wrong" in seen[0] and "### Example 2" not in seen[0]
    assert len(child.evaluation.dataset_ids) == 4                    # the gate batch is still `minibatch`


async def test_defaults_render_the_same_meta_prompt():
    """No context, no reflect_rows: the meta-prompt (a cache key) and origin params are what they were."""
    seen = []
    tree = Tree(wide_task(lambda p, c, s: (seen.append(p), Variants(prompts=["Answer {q} +"]))[1] if s is Variants
                          else Out(answer="wrong")))
    await run(tree, gepa(minibatch=3, seed=0), stop=Stop(rounds=2))
    child = next(n for n in tree.nodes.values() if n.origin.op == "reflect")
    assert "context" not in child.origin.params and len(child.origin.params["minibatch_ids"]) == 3
    assert "feedback on each:\n\n### Example 1\n" in seen[0]
