import asyncio

from bpto import Dataset, Tree, evaluate, select
from tasks.hotpotqa import ROOT_PROMPT, em_f1, make_task, normalize_answer, render_context
from tasks.hotpotqa.feedback import feedback, passed
from tasks.hotpotqa.mock import _mock_client

ROWS = [
    {"id": "a", "inputs": {"question": "Who wrote X?", "context": render_context(["X", "Y"], [["X was written by Jane Roe."], ["Y is a town."]])},
     "answer": "Jane Roe", "meta": {"type": "bridge", "supporting_titles": ["X"], "gold": ["X: X was written by Jane Roe."]}},
    {"id": "b", "inputs": {"question": "Is Y a town?", "context": "Y: Y is a town."}, "answer": "yes",
     "meta": {"type": "comparison", "supporting_titles": ["Y"], "gold": []}},
]


def test_normalisation_and_f1():
    assert normalize_answer("The Beatles!") == "beatles"
    assert em_f1("Jane Roe", "jane roe.") == (1.0, 1.0)
    assert em_f1("Roe", "Jane Roe") == (0.0, 2 / 3)
    assert em_f1("yes", "no") == (0.0, 0.0) and em_f1("yes, it is", "yes") == (0.0, 0.0)


def test_task_evaluates_and_feedback_reads_traces():
    data = Dataset.from_records(ROWS)
    task = make_task(_mock_client(data, base=1.0), data)
    tree = Tree(task)
    asyncio.run(tree.apply(evaluate(), select.root))
    ev = tree.root.evaluation
    assert ev.metrics["f1"] == 1.0 and ev.metrics["em"] == 1.0 and ev.feasible
    r = ev.per_example[0]
    assert passed(data[0], r)
    assert "'Jane Roe'" in feedback(data[0], r) and "['X']" in feedback(data[0], r)
    task2 = make_task(_mock_client(data, base=0.0, per_cue=0.0), data)
    tree2 = Tree(task2)
    asyncio.run(tree2.apply(evaluate(), select.root))
    r2 = tree2.root.evaluation.per_example[1]
    assert not passed(data[1], r2) and "comparison question" in feedback(data[1], r2)
    assert "{context}" in ROOT_PROMPT and "{question}" in ROOT_PROMPT
