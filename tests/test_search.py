import pytest

from bpto import (Budget, BudgetExceeded, Dataset, MockClient, Stop, Task, Tree, evaluate, guided, random, run,
                  select, step, successive_halving)
from bpto.ops import Op
from tests.conftest import RECORDS, handler


def _sched():
    return [step(random(n=2)), step(guided("make it more succinct", n=2)), step(evaluate(), select.unevaluated)]


async def test_run_rounds_and_history(task):
    res = await run(Tree(task), _sched(), stop=Stop(rounds=2))
    assert res.rounds == 2 and res.stopped_because == "rounds=2"
    assert [h["step"] for h in res.history] == ["random", "guided", "evaluate"] * 2
    assert all(n.evaluated for n in res.tree)
    assert res.tree.meta["run"]["round"] == 2


async def test_stop_conditions(task):
    res = await run(Tree(task), _sched(), stop=Stop(rounds=10, max_nodes=20))
    assert res.stopped_because == "max_nodes=20" and len(res.tree) >= 20
    res = await run(Tree(task), _sched(), stop=Stop(rounds=10, max_depth=2))
    assert res.stopped_because == "max_depth=2"
    res = await run(Tree(task), _sched(), stop=Stop(rounds=10, until=lambda t: len(t.evaluated_nodes()) > 0))
    assert res.rounds == 1


async def test_budget_stops_run(client):
    client.budget = Budget(max_calls=5)
    task = Task(root="Extract the name of the person from the following text:\n{text}", description="x",
                dataset=Dataset.from_records(RECORDS), scorer=lambda *a: {"accuracy": 1.0},
                objective=lambda m, c: (m.get("accuracy", 0.0), True), client=client)
    res = await run(Tree(task), _sched(), stop=Stop(rounds=10))
    assert res.stopped_because.startswith("budget")


class Boom(Op):
    name = "boom"
    async def run_one(self, tree, node):
        raise RuntimeError("crash")


async def test_checkpoint_and_resume(task, tmp_path):
    ck = tmp_path / "tree.json"
    sched = [step(random(n=2)), step(evaluate(), select.unevaluated), step(Boom(), select.root)]
    with pytest.raises(RuntimeError):
        await run(Tree(task), sched, stop=Stop(rounds=3), checkpoint=ck)
    t2 = Tree.load(ck, task)
    assert t2.meta["run"] == {"round": 0, "step": 2, "best": None, "stale": 0}
    assert len(t2) == 3 and all(n.evaluated for n in t2)
    # resume with a fixed schedule: continues at step index 2 of round 0, then finishes round 1
    sched[2] = step(guided("shorter", n=1))
    res = await run(t2, sched, stop=Stop(rounds=2))
    assert res.rounds == 2 and res.history[0]["step"] == "guided" and res.history[0]["round"] == 0
    assert len(res.history) == 4  # 1 remaining step of round 0 + 3 of round 1


async def test_successive_halving_reuses_cache(task):
    tree = Tree(task)
    await tree.apply(random(n=6))
    steps = successive_halving([2, 3, None], keep=0.5, seed=1)
    assert [s.name for s in steps] == ["eval[2]", "eval[3]", "eval[None]"]
    calls = []
    for st in steps:
        before = task.client.usage.calls
        await tree.apply(st.op, st.select)
        calls.append(task.client.usage.calls - before)
    # rung 1: 7 nodes x 2 examples; rung 2: 4 nodes x 1 new example; rung 3: 2 nodes x 1 new example
    assert calls == [14, 4, 2]
    ns = sorted(n.evaluation.n for n in tree.evaluated_nodes())
    assert ns.count(4) == 2 and ns.count(3) == 2 and ns.count(2) == 3


def test_nested_sampling():
    ds = Dataset.from_records(RECORDS)
    a = [e.id for e in ds.sample(2, seed=3)]
    b = [e.id for e in ds.sample(3, seed=3)]
    assert b[:2] == a
