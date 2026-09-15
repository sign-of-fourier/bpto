"""HotpotQA (distractor) task: optimise a single reading-comprehension prompt for multi-hop QA.

This directory is a *user* of bpto, not part of it. The prompt under search renders `{question}` and
`{context}` (10 paragraphs); the model answers with a JSON `{"answer": ...}` (prose before the object is
tolerated - the Bedrock client extracts the last balanced object - so prompts may ask for reasoning first).

Metrics per example: em, f1 (official HotpotQA normalisation: lowercase, strip punctuation/articles/whitespace,
token-level F1), output_tokens, prompt_tokens. Objective: maximise f1.
"""
from __future__ import annotations

import re
import string
from collections import Counter

from pydantic import BaseModel, Field

from bpto import Dataset, LinearObjective, ModelClient, ModelConfig, Task, combine, output_token_count, token_count

from .data import from_hf, load, load_or_fetch, render_context, save_jsonl

__all__ = ["Answer", "AnswerWithReasoning", "ROOT_PROMPT", "normalize_answer", "em_f1", "qa_scorer", "f1_objective", "make_task",
           "from_hf", "load", "load_or_fetch", "save_jsonl", "render_context"]


class Answer(BaseModel):
    answer: str = Field(description="The final answer: a short span copied from the context, or yes / no.")


class AnswerWithReasoning(BaseModel):
    """Same, with room to think first (DSPy's ChainOfThought signature). The prompt decides how to use it."""
    reasoning: str = Field(default="", description="Brief reasoning: which paragraphs are relevant and what they say.")
    answer: str = Field(description="The final answer: a short span copied from the context, or yes / no.")


ROOT_PROMPT = (
    "Answer the question using the paragraphs below. Some paragraphs are relevant and others are distractors; "
    "the answer usually requires combining facts from two paragraphs. Give the shortest exact answer: a name, "
    "date, number or phrase copied from the text, or yes / no for yes-no questions.\n\n"
    "Paragraphs:\n{context}\n\nQuestion: {question}"
)


def normalize_answer(s: str) -> str:
    """HotpotQA / SQuAD official normalisation."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred: str, gold: str) -> tuple[float, float]:
    p, g = normalize_answer(pred), normalize_answer(gold)
    em = float(p == g)
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:  # official: yes/no scored by EM only
        return em, em
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return em, 0.0
    prec, rec = common / len(pt), common / len(gt)
    return em, 2 * prec * rec / (prec + rec)


def qa_scorer():
    def _score(prompt, example, completion, ctx):
        got = completion.parsed
        pred = (got.get("answer") if isinstance(got, dict) else getattr(got, "answer", None)) if got is not None else None
        if pred is None:
            pred = completion.text.strip().splitlines()[-1] if completion.text.strip() else ""
        em, f1 = em_f1(pred, example.answer or "")
        return {"em": em, "f1": f1}
    return _score


def f1_objective():
    return LinearObjective(f1=1.0)


def make_task(client: ModelClient, dataset: Dataset | None = None, objective=None, root: str = ROOT_PROMPT,
              config: ModelConfig | None = None, reasoning: bool = False, **kw) -> Task:
    return Task(
        root=root,
        description="answers a multi-hop question from a set of Wikipedia paragraphs, most of which are distractors",
        dataset=dataset if dataset is not None else load_or_fetch(),
        schema=AnswerWithReasoning if reasoning else Answer,
        scorer=combine(qa_scorer(), token_count(), output_token_count()),
        objective=objective or f1_objective(),
        client=client,
        config=config or ModelConfig(max_tokens=512),
        **kw,
    )
