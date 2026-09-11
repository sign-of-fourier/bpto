"""IFBench task: optimise a system prompt so a model follows unusual, programmatically-verifiable constraints.

This directory is a *user* of bpto, not part of it. The prompt under search wraps the user's
instruction (`{instruction}`); each example carries its constraint ids + kwargs in `meta`, and
scoring runs AllenAI's checkers on the raw completion. No reference answers, no LLM judge.

Metrics per example: strict / loose (prompt-level, all constraints pass), inst_strict / inst_loose
(fraction of constraints passed), output_tokens, template_tokens.
"""
from __future__ import annotations

from bpto import (ConstrainedObjective, Dataset, LinearObjective, ModelClient, ModelConfig, Task, combine,
                  output_token_count, template_tokens)

from .checkers import Checker, available, check
from .data import constraint_types, from_hf, load, load_or_fetch, save_jsonl

__all__ = ["ROOT_PROMPT", "ifbench_scorer", "strict_objective", "loose_objective", "compact_objective", "make_task",
           "from_hf", "load", "load_or_fetch", "save_jsonl", "constraint_types", "check", "available"]

ROOT_PROMPT = (
    "You are a careful assistant. Read the request below and respond to it. The request contains one or more "
    "explicit formatting or content constraints (for example about word counts, letters, keywords, structure "
    "or punctuation). Follow every constraint exactly and literally, even if it makes the answer unusual. "
    "Do not explain the constraints or mention that you are following them; just produce the answer.\n\n"
    "Request:\n{instruction}"
)


def ifbench_scorer(checker: Checker = check):
    """Run the constraint checkers on the completion text. `checker` is injectable for offline tests."""
    def _score(prompt, example, completion, ctx):
        ids, kwargs = example.meta["instruction_id_list"], example.meta["kwargs"]
        strict, loose = checker(completion.text, ids, kwargs, example.inputs["instruction"])
        n = max(1, len(ids))
        return {"strict": float(all(strict)), "loose": float(all(loose)),
                "inst_strict": sum(strict) / n, "inst_loose": sum(loose) / n}
    return _score


def strict_objective():
    return LinearObjective(strict=1.0)


def loose_objective():
    return LinearObjective(loose=1.0)


def compact_objective(token_weight: float = 0.001, metric: str = "strict"):
    """Accuracy minus a small charge per template token: prefer the shortest prompt that still works."""
    return LinearObjective(**{metric: 1.0, "template_tokens": -token_weight})


def make_task(client: ModelClient, dataset: Dataset | None = None, objective=None, root: str = ROOT_PROMPT,
              config: ModelConfig | None = None, checker: Checker = check, **kw) -> Task:
    return Task(
        root=root,
        description="answers a user request while obeying its explicit formatting/content constraints exactly",
        dataset=dataset if dataset is not None else load_or_fetch(),
        schema=None,  # free text; the checkers read the raw completion
        scorer=combine(ifbench_scorer(checker), template_tokens(), output_token_count()),
        objective=objective or strict_objective(),
        client=client,
        config=config,
        **kw,
    )
