"""Runs a graph `Program` for one example: entry -> edges -> terminal, every module call through the task client.

The executor owns the per-example step cap and writes the trace; caching, concurrency, budget and usage are the
client's. Metrics it produces (`steps`, `capped`, `tokens_per_module.<m>`, `parse_fail.<m>`) join the scorer's
vector; nothing is scalarized here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from .data import Example
from .llm import Completion, ModelConfig
from .metrics import Metrics
from .prompt import Edge, Program

if TYPE_CHECKING:
    from .task import Task

TEXT_FIELD = "$text"  # mapping source meaning the raw completion text


@dataclass
class ProgramRun:
    completion: Completion          # the terminal module's completion (what the scorer scores)
    rendered: str                   # the terminal module's rendered prompt
    trace: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # module -> visits [{input, output, parsed, step_idx}]
    metrics: Metrics = field(default_factory=dict)
    path: list[str] = field(default_factory=list)


def as_fields(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, BaseModel):
        return parsed.model_dump()
    return dict(parsed) if isinstance(parsed, dict) else {}


def pick_edge(edges: list[Edge], fields: dict[str, Any]) -> tuple[Edge, bool]:
    """(edge, parse_failed). One edge is unconditional; several are chosen by the `next` field."""
    if len(edges) == 1:
        return edges[0], False
    nxt = fields.get("next")
    for e in edges:
        if e.name == nxt:
            return e, False
    return next((e for e in edges if e.default), edges[0]), True


async def run_program(task: "Task", program: Program, example: Example, node_config: ModelConfig | None = None) -> ProgramRun:
    """Execute `program` on `example`. Raises what the client raises (BudgetExceeded must propagate)."""
    errs = program.graph_errors()
    if errs:
        raise ValueError("invalid program graph: " + "; ".join(errs))
    state: dict[str, Any] = dict(example.inputs)
    trace: dict[str, list[dict[str, Any]]] = {}
    tokens: dict[str, int] = {m: 0 for m in program.modules}
    parse_fail: dict[str, int] = {m: 0 for m in program.modules if program.is_orchestrating(m)}
    visits: dict[str, int] = {m: 0 for m in program.modules}
    path: list[str] = []
    current, step, capped = program.entry, 0, False
    comp: Completion | None = None
    rendered = ""
    while True:
        if step >= program.max_steps or (program.max_visits is not None and visits[current] >= program.max_visits):
            capped = True
            break
        adapter = task.adapters.get(current)
        render_inputs = {**state, **adapter(example, dict(state))} if adapter is not None else state  # adapter output is render-only
        rendered = program.modules[current].render(**render_inputs)
        cfg = (task.config_for(current) or task.client.default_config).merged(node_config)
        comp = await task.client.complete(rendered, config=cfg, schema=task.schema_for(current))
        fields = as_fields(comp.parsed)
        trace.setdefault(current, []).append({"input": rendered, "output": comp.text, "parsed": fields or None, "step_idx": step})
        tokens[current] += comp.input_tokens + comp.output_tokens
        visits[current] += 1
        path.append(current)
        step += 1
        edges = program.outgoing(current)
        if not edges:
            break
        edge, failed = pick_edge(edges, fields)
        if failed:
            parse_fail[current] += 1
        for placeholder, src in edge.mapping.items():
            state[placeholder] = comp.text if src == TEXT_FIELD else fields.get(src)
        current = edge.to
    if comp is None:  # max_steps == 0: nothing ran
        comp = Completion(text="")
    metrics: Metrics = {"steps": float(step), "capped": 1.0 if capped else 0.0}
    metrics.update({f"tokens_per_module.{m}": float(v) for m, v in tokens.items()})
    metrics.update({f"parse_fail.{m}": float(v) for m, v in parse_fail.items()})
    return ProgramRun(completion=comp, rendered=rendered, trace=trace, metrics=metrics, path=path)
