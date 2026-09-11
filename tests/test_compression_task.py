import re

from bpto import Dataset, MockClient, Prompt, Tree, evaluate, guided, select
from bpto.ops import Variants
from tasks.compression import Entities, generate_dataset, make_task, shrinking_budget_objective, set_f1

DATA = Dataset.from_records([
    {"inputs": {"text": "Charles met Alice."}, "answer": ["Charles", "Alice"]},
    {"inputs": {"text": "Nobody came."}, "answer": []},
])


def handler(prompt, cfg, schema):
    if schema is Variants:
        base = re.search(r"<prompt>\n(.*?)\n</prompt>", prompt, re.S).group(1)
        return Variants(prompts=["Names in:\n{text}", base + " Be thorough."])
    text = prompt.rsplit("\n", 1)[-1]
    return Entities(names=[w for w in re.findall(r"[A-Z][a-z]+", text) if w != "Nobody"][:1])  # misses Alice


def test_set_f1_partial_credit():
    from bpto import Completion, Example
    m = set_f1()(None, Example(id="0", inputs={}, answer=["Charles", "Alice"]),
                 Completion(text="", parsed=Entities(names=["charles"])), None)
    assert m["precision"] == 1.0 and m["recall"] == 0.5 and round(m["f1"], 3) == 0.667


async def test_compression_flow():
    task = make_task(MockClient(handler), DATA, objective=shrinking_budget_objective(50, 25, floor=3))
    tree = Tree(task)
    await tree.apply(guided("shorter", n=2))
    await tree.apply(evaluate(), select=select.unevaluated)
    short = next(n for n in tree if n.prompt.template.startswith("Names"))
    assert short.evaluation.feasible and short.evaluation.metrics["f1"] < 1.0
    assert tree.root.evaluation.metrics['template_tokens'] > 50 > short.evaluation.metrics['template_tokens']
    assert tree.root.evaluation.feasible is False  # 50-token budget at depth 0 vs the long root
    assert tree.best().id == short.id
    assert {n.id for n in tree.pareto({"f1": True, "prompt_tokens": False})} >= {short.id}


def test_generate_dataset_is_seeded_and_has_distractors():
    a, b = generate_dataset(30, seed=1), generate_dataset(30, seed=1)
    assert [e.answer for e in a] == [e.answer for e in b]
    assert any(e.answer == [] for e in a) and any(len(e.answer) >= 2 for e in a)
    assert any("Nantucket" in e.inputs["text"] or "Oslo" in e.inputs["text"] for e in a)


async def test_template_tokens_counts_template_only(task):
    from bpto import Completion, Example, template_tokens
    from bpto.scoring import ScoreContext
    p = Prompt(template="one two three {text}")
    m = await template_tokens()(p, Example(id="0", inputs={"text": "x " * 50}), Completion(text=""),
                                ScoreContext(task, task.client, ""))
    assert m == {"template_tokens": 3.0}
