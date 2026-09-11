"""Scorer (per-example -> Metrics vector) and Objective (vector -> scalar, feasible).

Scoring is user-defined. What ships here are protocols plus a few generic building blocks.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, TYPE_CHECKING

from pydantic import BaseModel, Field

from .data import Example
from .llm import Completion, ModelClient
from .metrics import Metrics
from .prompt import Prompt

if TYPE_CHECKING:
    from .task import Task


@dataclass
class ScoreContext:
    """Handed to scorers; gives LLM-as-judge scorers a client without changing the signature."""
    task: "Task"
    client: ModelClient
    rendered_prompt: str


ScorerFn = Callable[[Prompt, Example, Completion, ScoreContext], Metrics | Awaitable[Metrics]]


class Scorer(Protocol):
    def __call__(self, prompt: Prompt, example: Example, completion: Completion, ctx: ScoreContext) -> Metrics | Awaitable[Metrics]: ...


async def run_scorer(scorer: Scorer, prompt: Prompt, example: Example, completion: Completion, ctx: ScoreContext) -> Metrics:
    out = scorer(prompt, example, completion, ctx)
    if inspect.isawaitable(out):
        out = await out
    return dict(out)


def combine(*scorers: Scorer) -> Scorer:
    """Merge several scorers' metric dicts into one."""
    async def _combined(prompt, example, completion, ctx):
        out: Metrics = {}
        for s in scorers:
            out.update(await run_scorer(s, prompt, example, completion, ctx))
        return out
    return _combined


# ---- generic scorers ---------------------------------------------------------------

def exact_match(field: str | None = None, name: str = "accuracy", normalize: Callable[[Any], Any] = lambda x: x) -> Scorer:
    """1.0 if the (optionally field-extracted) parsed output equals example.answer."""
    def _score(prompt, example, completion, ctx):
        got = completion.parsed if completion.parsed is not None else completion.text
        if field is not None and got is not None:
            got = getattr(got, field, None) if not isinstance(got, dict) else got.get(field)
        return {name: 1.0 if normalize(got) == normalize(example.answer) else 0.0}
    return _score


def token_count(name: str = "prompt_tokens") -> Scorer:
    """Input tokens of the rendered prompt as reported by the model."""
    def _score(prompt, example, completion, ctx):
        return {name: float(completion.input_tokens)}
    return _score


def template_tokens(name: str = "template_tokens") -> Scorer:
    """Tokens of the prompt *template* (placeholders removed) - the thing being compressed -
    independent of the example. Counted once per template via client.count_tokens."""
    cache: dict[str, int] = {}

    async def _score(prompt, example, completion, ctx):
        key = prompt.hash
        if key not in cache:
            stripped = prompt.template.format(**{p: "" for p in prompt.placeholders})
            cache[key] = await ctx.client.count_tokens(stripped, ctx.task.config)
        return {name: float(cache[key])}
    return _score


def output_token_count(name: str = "output_tokens") -> Scorer:
    def _score(prompt, example, completion, ctx):
        return {name: float(completion.output_tokens)}
    return _score


class JudgeVerdict(BaseModel):
    score: float = Field(description="Score in [0, 1].")
    reason: str = ""


def llm_judge(rubric: str, name: str = "judge", client: ModelClient | None = None,
              config: Any = None, template: str | None = None) -> Scorer:
    """LLM-as-judge: scores the model output against example.answer with a rubric.

    Uses the task client unless `client` is given (e.g. a stronger/cheaper judge model).
    """
    tmpl = template or (
        "You are grading an answer.\n"
        "Rubric: {rubric}\n\n"
        "Input given to the model:\n<input>\n{inputs}\n</input>\n\n"
        "Reference answer:\n<reference>\n{answer}\n</reference>\n\n"
        "Model answer:\n<answer>\n{output}\n</answer>\n\n"
        "Give a score from 0 to 1 and a one-sentence reason."
    )

    async def _score(prompt, example, completion, ctx):
        judge = client or ctx.client
        text = tmpl.format(rubric=rubric, inputs=json.dumps(example.inputs, default=str),
                           answer=example.answer, output=completion.text)
        comp = await judge.complete(text, config=config, schema=JudgeVerdict)
        v = comp.parsed_as(JudgeVerdict)
        return {name: max(0.0, min(1.0, v.score))}
    return _score


# ---- objectives --------------------------------------------------------------------

@dataclass
class ObjectiveContext:
    depth: int
    n_evaluated: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Objective(Protocol):
    def __call__(self, metrics: Metrics, ctx: ObjectiveContext) -> tuple[float, bool]: ...


class LinearObjective:
    def __init__(self, **weights: float):
        self.weights = weights

    def __call__(self, metrics: Metrics, ctx: ObjectiveContext) -> tuple[float, bool]:
        return sum(w * metrics.get(k, 0.0) for k, w in self.weights.items()), True


class ConstrainedObjective:
    """objective + a constraint on one metric whose bound may depend on the tree context.

    bound: number or callable(ObjectiveContext) -> number, so it can tighten with depth.
    """

    def __init__(self, objective: Objective, metric: str, bound: float | Callable[[ObjectiveContext], float],
                 sense: str = "<=", penalty: float | None = None):
        self.objective, self.metric, self.bound, self.sense, self.penalty = objective, metric, bound, sense, penalty

    def __call__(self, metrics: Metrics, ctx: ObjectiveContext) -> tuple[float, bool]:
        score, feasible = self.objective(metrics, ctx)
        b = self.bound(ctx) if callable(self.bound) else self.bound
        v = metrics.get(self.metric, 0.0)
        ok = v <= b if self.sense == "<=" else v >= b
        if not ok and self.penalty is not None:
            score -= self.penalty * abs(v - b)
        return score, feasible and ok


def pareto_front(points: dict[str, Metrics], maximize: dict[str, bool]) -> list[str]:
    """Ids of non-dominated points. `maximize` maps metric -> True (max) / False (min)."""
    def dominates(a: Metrics, b: Metrics) -> bool:
        better_or_eq = all(
            (a.get(k, 0) >= b.get(k, 0)) if mx else (a.get(k, 0) <= b.get(k, 0)) for k, mx in maximize.items()
        )
        strictly = any(
            (a.get(k, 0) > b.get(k, 0)) if mx else (a.get(k, 0) < b.get(k, 0)) for k, mx in maximize.items()
        )
        return better_or_eq and strictly

    ids = list(points)
    return [i for i in ids if not any(dominates(points[j], points[i]) for j in ids if j != i)]
