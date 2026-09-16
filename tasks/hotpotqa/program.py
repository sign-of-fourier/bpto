"""Two-stage HotpotQA program, after the GEPA paper's multi-hop setting: the prompt under search *selects* the
relevant paragraphs (a stand-in for writing retrieval queries), a fixed answerer prompt then answers from only
those. Optimising the selector is where the paper's HotpotQA headroom lives; `tasks/hotpotqa` (single prompt
over all 10 paragraphs) has none on Nova Micro.

Stage A (entry module "selector"): {question}, {context} -> Titles{titles}. Stage B ("answerer", run inside the
scorer): question + selected paragraphs -> Answer. Rollouts = 2 calls per example, charged to the same client.
Metrics: sel_recall / sel_precision / sel_f1 (titles vs supporting_facts), n_selected, em, f1 (answer).

`make_program_task(modules=("selector",))` (default) keeps the answerer fixed at ANSWER_PROMPT and the node a plain
Prompt; `modules=("selector", "answerer")` makes the node a Program with both under search - the scorer then takes
the answerer from the node and records stage B's input/output in `ctx.trace["answerer"]` for the reflector.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from bpto import Dataset, LinearObjective, ModelClient, ModelConfig, Program, Prompt, Task, combine, output_token_count, token_count

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


def program_scorer(answer_prompt: str = ANSWER_PROMPT, answer_config: ModelConfig | None = None, max_selected: int = 4):
    """Stage B inside the scorer: answer from the selected paragraphs, then score both stages. The answerer prompt
    is the node's "answerer" module when the node is a Program, else `answer_prompt`."""
    fixed = Prompt(template=answer_prompt)

    async def _score(prompt, example, completion, ctx):
        ans_prompt = prompt.modules["answerer"] if isinstance(prompt, Program) and "answerer" in prompt.modules else fixed
        context = example.inputs["context"]
        picked = select_titles(completion.parsed, context)[:max_selected]
        gold = {_norm_title(t) for t in example.meta.get("supporting_titles", [])}
        got = {_norm_title(t) for t in picked}
        tp = len(gold & got)
        rec = tp / len(gold) if gold else 0.0
        prec = tp / len(got) if got else 0.0
        sf1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        paras = paragraphs(context)
        sub = "\n\n".join(paras[t] for t in picked) if picked else "(no paragraphs selected)"
        cfg = (answer_config or ctx.task.config or ctx.client.default_config)
        rendered = ans_prompt.render(context=sub, question=example.inputs["question"])
        comp = await ctx.client.complete(rendered, config=cfg, schema=AnswerWithReasoning)
        a = comp.parsed
        pred = (a.get("answer") if isinstance(a, dict) else getattr(a, "answer", None)) or ""
        em, f1 = em_f1(pred, example.answer or "")
        ctx.trace["answerer"] = {"input": f"question: {example.inputs['question']}\nselected paragraphs:\n{sub}",
                                 "output": comp.text, "answer": pred}
        return {"sel_recall": rec, "sel_precision": prec, "sel_f1": sf1, "n_selected": float(len(picked)),
                "em": em, "f1": f1, "answer_tokens": float(comp.output_tokens)}
    return _score


DESCRIPTIONS = {
    "selector": "selects, from ten Wikipedia paragraphs, the ones needed to answer a multi-hop question "
                "(a second prompt then answers from the selected paragraphs only)",
    "answerer": "answers a multi-hop question from the two or so paragraphs a first prompt selected, giving the "
                "shortest exact answer (scored by token F1 against a short reference answer)",
}


def make_program_task(client: ModelClient, dataset: Dataset | None = None, objective=None, root: str = SELECT_ROOT,
                      config: ModelConfig | None = None, modules: tuple[str, ...] = ("selector",), **kw) -> Task:
    if "answerer" in modules:
        root_prompt = {"selector": root, "answerer": ANSWER_PROMPT}
        description = {k: DESCRIPTIONS[k] for k in root_prompt}
    else:
        root_prompt, description = root, DESCRIPTIONS["selector"].replace("a second prompt", "a fixed second prompt")
    return Task(
        root=root_prompt,
        description=description,
        dataset=dataset if dataset is not None else load_or_fetch(),
        schema=Titles,
        scorer=combine(program_scorer(), token_count(), output_token_count()),
        objective=objective or LinearObjective(f1=1.0),
        client=client,
        config=config or ModelConfig(max_tokens=512, temperature=0.0),
        **kw,
    )
