"""Two-stage HotpotQA program, after the GEPA paper's multi-hop setting: the prompt under search *selects* the
relevant paragraphs (a stand-in for writing retrieval queries), an answerer prompt then answers from only those.
Optimising the selector is where the paper's HotpotQA headroom lives; `tasks/hotpotqa` (single prompt over all 10
paragraphs) has none on Nova Micro.

The node is always a two-module graph `Program` run by bpto's executor: "selector" ({question}, {context} -> Titles)
-> edge (titles) -> "answerer" (question + selected paragraphs -> Answer). The `answerer` adapter turns the selected
titles into the paragraphs the answerer renders as {context}. Rollouts = 2 calls per example, charged to the same
client. Metrics: sel_recall / sel_precision / sel_f1 (titles vs supporting_facts), n_selected, em, f1 (answer), plus
the executor's steps / tokens_per_module.*; `prompt_tokens` (token_count) is the terminal module's input.

`make_program_task(modules=("selector",))` (default) keeps the answerer fixed (only the selector is rewritten:
`modules` is what the search mutates, the harness's round-robin list); `modules=("selector", "answerer")` puts both
under search. Either way the reflector for the answerer reads stage B's input/output from `trace["answerer"]`.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from bpto import (Dataset, Edge, Example, LinearObjective, ModelClient, ModelConfig, Program, Prompt, Task, combine,
                  output_token_count, token_count, trace_of)

from . import AnswerWithReasoning, em_f1, load_or_fetch


class Titles(BaseModel):
    titles: list[str] = Field(default_factory=list, description="Titles of the paragraphs needed to answer, copied exactly.")


SELECT_ROOT = (
    "Below are ten Wikipedia paragraphs, each starting with its title, followed by a question. Select the paragraphs "
    "that contain the facts needed to answer the question - usually two: one about the entity the question names "
    "and one about the entity it leads to. Return their titles exactly as written.\n\n"
    "Paragraphs:\n{context}\n\nQuestion: {question}"
)

ANSWER_PROMPT = (
    "Answer the question using the paragraphs below. Give the shortest exact answer: a name, date, number or phrase "
    "copied from the text, or yes / no for yes-no questions.\n\nParagraphs:\n{context}\n\nQuestion: {question}"
)


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).strip().lower())


def paragraphs(context: str) -> dict[str, str]:
    """title -> paragraph text, from the rendered context (blocks of 'Title: text')."""
    out = {}
    for block in context.split("\n\n"):
        if ":" in block:
            t, body = block.split(":", 1)
            out[t.strip()] = block.strip()
    return out


def select_titles(parsed, context: str) -> list[str]:
    got = (parsed.get("titles") if isinstance(parsed, dict) else getattr(parsed, "titles", None)) or []
    by_norm = {_norm_title(t): t for t in paragraphs(context)}
    picked = []
    for g in got:
        t = by_norm.get(_norm_title(g))
        if t is None:  # tolerate truncated / partial titles
            t = next((orig for n, orig in by_norm.items() if _norm_title(g) and (_norm_title(g) in n or n in _norm_title(g))), None)
        if t is not None and t not in picked:
            picked.append(t)
    return picked


def selected_titles(example: Example, titles, max_selected: int = 4) -> list[str]:
    return select_titles({"titles": list(titles or [])}, example.inputs["context"])[:max_selected]


def answerer_adapter(max_selected: int = 4):
    """Render-time inputs for the answerer: {context} becomes the paragraphs whose titles the selector returned."""
    def _adapt(example: Example, state: dict) -> dict:
        paras = paragraphs(example.inputs["context"])
        picked = selected_titles(example, state.get("titles"), max_selected)
        return {"context": "\n\n".join(paras[t] for t in picked) if picked else "(no paragraphs selected)"}
    return _adapt


def program_scorer(max_selected: int = 4):
    """Scores both stages from the terminal completion (answerer) and the selector's trace."""
    def _score(prompt, example, completion, ctx):
        sel = trace_of(ctx.trace, "selector") or {}
        picked = selected_titles(example, (sel.get("parsed") or {}).get("titles"), max_selected)
        gold = {_norm_title(t) for t in example.meta.get("supporting_titles", [])}
        got = {_norm_title(t) for t in picked}
        tp = len(gold & got)
        rec = tp / len(gold) if gold else 0.0
        prec = tp / len(got) if got else 0.0
        sf1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        a = completion.parsed
        pred = (a.get("answer") if isinstance(a, dict) else getattr(a, "answer", None)) or ""
        em, f1 = em_f1(pred, example.answer or "")
        return {"sel_recall": rec, "sel_precision": prec, "sel_f1": sf1, "n_selected": float(len(picked)),
                "em": em, "f1": f1, "answer_tokens": float(completion.output_tokens)}
    return _score


def program(selector: str = SELECT_ROOT, answerer: str = ANSWER_PROMPT) -> Program:
    return Program(modules={"selector": Prompt(template=selector), "answerer": Prompt(template=answerer)}, entry="selector",
                   edges={"selector": [Edge(to="answerer", mapping={"titles": "titles"})]})


DESCRIPTIONS = {
    "selector": "selects, from ten Wikipedia paragraphs, the ones needed to answer a multi-hop question "
                "(a second prompt then answers from the selected paragraphs only)",
    "answerer": "answers a multi-hop question from the two or so paragraphs a first prompt selected, giving the "
                "shortest exact answer (scored by token F1 against a short reference answer)",
}


def make_program_task(client: ModelClient, dataset: Dataset | None = None, objective=None, root: str = SELECT_ROOT,
                      config: ModelConfig | None = None, modules: tuple[str, ...] = ("selector",), max_selected: int = 4,
                      answer_config: ModelConfig | None = None, **kw) -> Task:
    description = dict(DESCRIPTIONS)
    if "answerer" not in modules:
        description["selector"] = description["selector"].replace("a second prompt", "a fixed second prompt")
    cfg = config or ModelConfig(max_tokens=512, temperature=0.0)
    return Task(
        root=program(root),
        description=description,
        dataset=dataset if dataset is not None else load_or_fetch(),
        schema={"selector": Titles, "answerer": AnswerWithReasoning},
        scorer=combine(program_scorer(max_selected), token_count(), output_token_count()),
        objective=objective or LinearObjective(f1=1.0),
        client=client,
        config={"selector": cfg, "answerer": answer_config or cfg},
        adapters={"answerer": answerer_adapter(max_selected)},
        **kw,
    )
