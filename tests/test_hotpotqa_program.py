import asyncio

from bpto import Dataset, MockClient, Tree, evaluate, select
from bpto.ops import Variants
from tasks.hotpotqa import AnswerWithReasoning, render_context
from tasks.hotpotqa.program import Titles, make_program_task, paragraphs, select_titles

CTX = render_context(["Alpha Corp", "Beta (film)", "Gamma"], [["Alpha Corp was founded by Jane Roe."], ["Beta is a 1999 film."], ["Gamma is a letter."]])
ROWS = [{"id": "a", "inputs": {"question": "Who founded Alpha Corp?", "context": CTX}, "answer": "Jane Roe",
         "meta": {"type": "bridge", "supporting_titles": ["Alpha Corp", "Beta (film)"], "gold": []}}]


def test_paragraph_parsing_and_title_matching():
    p = paragraphs(CTX)
    assert set(p) == {"Alpha Corp", "Beta (film)", "Gamma"} and p["Gamma"].startswith("Gamma:")
    assert select_titles({"titles": ["alpha corp", "Beta", "Nope"]}, CTX) == ["Alpha Corp", "Beta (film)"]


def test_program_task_runs_both_stages_and_scores_each():
    calls = []

    def handler(prompt, cfg, schema):
        calls.append(schema)
        if schema is Titles:
            return Titles(titles=["Alpha Corp"])
        assert schema is AnswerWithReasoning and "Alpha Corp was founded" in prompt and "Gamma is" not in prompt
        return AnswerWithReasoning(answer="Jane Roe")
    data = Dataset.from_records(ROWS)
    tree = Tree(make_program_task(MockClient(handler), data))
    asyncio.run(tree.apply(evaluate(), select.root))
    m = tree.root.evaluation.metrics
    assert m["f1"] == 1.0 and m["sel_recall"] == 0.5 and m["sel_precision"] == 1.0 and m["n_selected"] == 1.0
    assert m["steps"] == 2.0 and m["tokens_per_module.selector"] > 0 and m["capped"] == 0.0
    assert calls == [Titles, AnswerWithReasoning]
    r = tree.root.evaluation.per_example[0]
    assert r.trace["selector"][0]["parsed"] == {"titles": ["Alpha Corp"]} and "Alpha Corp was founded" in r.trace["answerer"][0]["input"]
    from tasks.hotpotqa.feedback import answerer_feedback, program_feedback
    assert "MISSED ['Beta (film)']" in program_feedback(data[0], r) and "model answered: 'Jane Roe'" in answerer_feedback(data[0], r)


def test_program_feedback_names_missed_paragraph():
    from bpto import Example
    from bpto.tree import ExampleResult
    from tasks.hotpotqa.feedback import program_feedback, program_passed
    ex = Example(id="x", inputs={"question": "q", "context": "Alpha: a text.\n\nBeta: b text.\n\nGamma: c text."},
                 answer="b", meta={"supporting_titles": ["Alpha", "Beta"], "type": "bridge"})
    r = ExampleResult(example_id="x", output="", parsed={"answer": "c"}, metrics={"sel_recall": 0.5, "f1": 0.0, "em": 0.0},
                      trace={"selector": [{"input": "", "output": "", "parsed": {"titles": ["Alpha", "Gamma"]}, "step_idx": 0}]})
    fb = program_feedback(ex, r)
    assert "MISSED ['Beta']" in fb and "unnecessary: ['Gamma']" in fb and not program_passed(ex, r)
    r2 = ExampleResult(example_id="x", output="", parsed={"titles": ["Alpha", "Beta"]}, metrics={"sel_recall": 1.0, "f1": 1.0, "em": 1.0})
    assert program_passed(ex, r2) and "exactly right" in program_feedback(ex, r2)
