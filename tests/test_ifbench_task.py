"""Offline tests for tasks/ifbench: loader, scorer with an injected checker, real checkers if installed."""
import json

import pytest

from bpto import Budget, Dataset, MockClient, Tree, evaluate, select
from tasks.ifbench import ROOT_PROMPT, available, compact_objective, constraint_types, ifbench_scorer, make_task
from tasks.ifbench.checkers import _relaxations

RECORDS = [
    {"id": "a", "inputs": {"instruction": "Write a haiku."}, "answer": None,
     "meta": {"instruction_id_list": ["count:word_count_range"], "kwargs": [{"min_words": 5, "max_words": 20}]}},
    {"id": "b", "inputs": {"instruction": "Say hi."}, "answer": None,
     "meta": {"instruction_id_list": ["format:list", "words:vowel"], "kwargs": [{}, {"letter": "e"}]}},
]


def fake_checker(response, ids, kwargs, instruction=""):
    strict = ["list" in response for _ in ids]
    return strict, [True] * len(ids)


@pytest.fixture
def dataset():
    return Dataset.from_records([json.loads(json.dumps(r)) for r in RECORDS])


async def test_scorer_metrics(dataset):
    client = MockClient(lambda p, c, s: "- a list\n- of things")
    task = make_task(client, dataset, checker=fake_checker)
    tree = Tree(task)
    await tree.apply(evaluate(), select.leaves)
    m = tree.root.evaluation.metrics
    assert m["strict"] == 1.0 and m["loose"] == 1.0 and m["inst_strict"] == 1.0
    assert m["template_tokens"] > 0 and m["output_tokens"] > 0
    assert "{instruction}" in str(tree.root.prompt)


async def test_partial_credit(dataset):
    def checker(response, ids, kwargs, instruction=""):
        return [i == 0 for i in range(len(ids))], [True] * len(ids)
    task = make_task(MockClient(lambda p, c, s: "x"), dataset, checker=checker)
    tree = Tree(task)
    await tree.apply(evaluate())
    m = tree.root.evaluation.metrics
    assert m["strict"] == 0.5          # example b fails one of two constraints
    assert m["inst_strict"] == 0.75    # (1 + 0.5) / 2
    assert m["loose"] == 1.0


def test_compact_objective_prefers_shorter():
    from bpto import ObjectiveContext
    obj = compact_objective(0.01)
    ctx = ObjectiveContext(depth=0, n_evaluated=0)
    assert obj({"strict": 0.8, "template_tokens": 50}, ctx)[0] > obj({"strict": 0.8, "template_tokens": 80}, ctx)[0]


def test_constraint_types(dataset):
    assert constraint_types(dataset) == {"count:word_count_range": 1, "format:list": 1, "words:vowel": 1}


def test_relaxations_shape():
    r = _relaxations("**title**\nbody\nfooter")
    assert len(r) == 8 and r[0].startswith("**") and "*" not in r[4]


async def test_held_out_score_does_not_store(dataset):
    task = make_task(MockClient(lambda p, c, s: "list"), dataset, checker=fake_checker)
    tree = Tree(task)
    ev = await evaluate(dataset=dataset).score(tree, tree.root)
    assert ev.metrics["strict"] == 1.0 and tree.root.evaluation is None


@pytest.mark.skipif(not available(), reason="ifbench checkers not installed")
def test_real_checkers():
    from tasks.ifbench.checkers import check
    s, l = check("one two three four five six", ["count:word_count_range"], [{"min_words": 5, "max_words": 20}])
    assert s == [True] and l == [True]
    s, l = check("one two", ["count:word_count_range"], [{"min_words": 5, "max_words": 20}])
    assert s == [False]
    assert check("", ["format:list"], [{}]) == ([False], [False])
