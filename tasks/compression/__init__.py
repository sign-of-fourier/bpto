"""Prompt compression task: keep extraction accuracy while shrinking the prompt template.

This directory is a *user* of bpto, not part of it. It will move to its own repo.
Everything task-specific - the response schema, partial-credit scoring, the multi-objective
or constrained objective, the data - is defined here and handed to bpto through `Task`.
"""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from bpto import (ConstrainedObjective, Dataset, LinearObjective, ModelClient, ModelConfig, Task, combine,
                  template_tokens, token_count)

from .data import generate_dataset, load, save_jsonl

__all__ = ["Entities", "ROOT_PROMPT", "set_f1", "weighted_objective", "shrinking_budget_objective", "make_task",
           "generate_dataset", "load", "save_jsonl"]


class Entities(BaseModel):
    names: list[str] = Field(default_factory=list, description="Names of people mentioned, in order of appearance.")


ROOT_PROMPT = (
    "You are an information extraction system. Read the passage below carefully and extract the full names "
    "of every person who is mentioned. Include each person only once, preserve the order in which they first "
    "appear, and do not include organisations, places, or pronouns. Titles such as Dr. or Professor are not "
    "part of the name. If no people are mentioned, return an empty list.\n\nPassage:\n{text}"
)


def _norm(s: str) -> str:
    return " ".join(s.lower().replace(".", "").split())


def set_f1(field: str = "names", name: str = "f1", normalize: Callable[[str], str] = _norm):
    """Partial credit: F1 between predicted and reference sets (empty == empty -> 1.0)."""
    def _score(prompt, example, completion, ctx):
        got = completion.parsed_as(Entities)
        pred = {normalize(x) for x in (getattr(got, field, None) or [])}
        ref = {normalize(x) for x in (example.answer or [])}
        if not pred and not ref:
            return {name: 1.0, "precision": 1.0, "recall": 1.0}
        tp = len(pred & ref)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(ref) if ref else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {name: f, "precision": p, "recall": r}
    return _score


def weighted_objective(token_weight: float = 0.002):
    """Scalarized multi-objective: f1 - w * template_tokens. The Pareto front is available separately."""
    return LinearObjective(f1=1.0, template_tokens=-token_weight)


def shrinking_budget_objective(start_tokens: float, shrink_per_depth: float, floor: float = 10.0):
    """Constrained: maximise f1 subject to template_tokens <= budget(depth), tighter as the tree deepens."""
    bound = lambda ctx: max(floor, start_tokens - shrink_per_depth * ctx.depth)
    return ConstrainedObjective(LinearObjective(f1=1.0), metric="template_tokens", bound=bound, penalty=0.01)


def make_task(client: ModelClient, dataset: Dataset | None = None, objective=None, root: str = ROOT_PROMPT,
              config: ModelConfig | None = None, **kw) -> Task:
    return Task(
        root=root,
        description="extracts the names of all people mentioned in a passage, as a list",
        dataset=dataset if dataset is not None else generate_dataset(),
        schema=Entities,
        scorer=combine(set_f1(), template_tokens(), token_count()),  # token_count = full input, for cost
        objective=objective or weighted_objective(),
        client=client,
        config=config,
        **kw,
    )
