"""Reflective prompt mutation, after GEPA: show the model a few examples with the current prompt's
outputs and task feedback, ask it to write a better prompt.

Traces come from the parent's stored evaluation (or its nearest evaluated ancestor), so the
minibatch costs no rollouts; GEPA re-runs them, we read them back. `feedback(example, result)`
is task-specific text ("constraint X failed: ..."), supplied by the task.
"""
from __future__ import annotations

import random as _random
from typing import Any, Callable

from ..data import Example
from ..llm.base import ModelConfig
from ..ops import LLMExpander
from ..tree import ExampleResult, Node, NodeState, Origin, Tree, trace_visits

Feedback = Callable[[Example, ExampleResult], str]
Passed = Callable[[Example, ExampleResult], bool]

REFLECT_PROMPT = (
    "I gave an assistant the following prompt template to perform a task. The template is supposed to: "
    "{description}\nIt must keep these placeholders exactly, in curly braces: {placeholders}\n\n"
    "Current prompt template:\n<prompt>\n{prompt}\n</prompt>\n\n"
    "Here are examples of task inputs, the assistant's response under this prompt, and feedback on each:\n\n"
    "{directive}\n"
    "Read the feedback carefully. Identify what the prompt is missing or getting wrong: any task-specific "
    "rules, edge cases, formatting requirements or strategies that would have fixed the failures, while "
    "keeping what already works. Make substantive changes - add concrete rules, a step-by-step strategy, "
    "or brief examples of the kinds of requirement that were missed; do not merely rephrase. Then write "
    "{n} improved prompt template(s). Each must be complete and usable on its own.{seed}"
)


def default_feedback(example: Example, result: ExampleResult) -> str:
    if result.error:
        return f"error: {result.error}"
    if example.answer is not None:
        return f"expected: {example.answer!r}; metrics: {result.metrics}"
    return f"metrics: {result.metrics}"


class ReflectiveExpander(LLMExpander):
    """n rewrites of the node's prompt informed by a minibatch of its own (or an ancestor's) traces.

    On a Program node, `module=` names the module to rewrite (GEPA's one-module-per-child mutation); `feedback` and
    `passed` may then be dicts keyed by module. Traces shown for a non-entry module come from
    `ExampleResult.trace[module]` (executor-written visits, or a scorer-written {"input", "output"}) instead of the
    entry output; under loops a module is visited several times per example and `visits` says how many of the last
    visits to show (1 = the last only).
    """
    name = "reflect"

    def __init__(self, feedback: Feedback | dict[str, Feedback] = default_feedback, minibatch: int = 3, n: int = 1,
                 calls: int = 1, seed: int | None = None, max_output_chars: int = 1500, meta_prompt: str | None = None,
                 config: ModelConfig | None = None, passed: Passed | dict[str, Passed] | None = None,
                 module: str | None = None, max_input_chars: int = 4000, visits: int = 1):
        super().__init__(n=n, calls=calls, meta_prompt=meta_prompt or REFLECT_PROMPT, config=config, module=module)
        pick = lambda f: f[module] if isinstance(f, dict) else f
        self.feedback, self.minibatch, self.seed = pick(feedback), minibatch, seed
        self.max_output_chars, self.max_input_chars, self.visits = max_output_chars, max_input_chars, visits
        # which traces count as successes (failures are shown first); default: objective >= 1. May return an int
        # ordering key instead of a bool (0 first) when some failures are more informative than others.
        self.passed = pick(passed)

    def source(self, tree: Tree, node: Node) -> Node | None:
        """The node whose traces are used: itself if evaluated, else the nearest evaluated ancestor."""
        for n in [node, *tree.ancestors(node)]:
            if n.evaluation is not None:
                return n
        return None

    def pick(self, tree: Tree, node: Node) -> tuple[Node | None, list[tuple[Example, ExampleResult]]]:
        src = self.source(tree, node)
        if src is None:
            return None, []
        by_id = {ex.id: ex for ex in tree.task.dataset}
        rows = [(by_id[r.example_id], r) for r in src.evaluation.per_example if r.example_id in by_id]
        rnd = _random.Random((self.seed or 0) ^ int(node.id, 16))  # deterministic per node: no shared state
        rnd.shuffle(rows)
        # failures first - they carry the information; successes fill the remaining slots
        rows.sort(key=lambda er: self.passed(er[0], er[1]) if self.passed else self._passed(tree, er[0], er[1]))
        return src, rows[: self.minibatch]

    @staticmethod
    def _passed(tree: Tree, ex: Example, r: ExampleResult) -> bool:
        from ..scoring import ObjectiveContext
        if r.error:
            return False
        return tree.task.objective(r.metrics, ObjectiveContext(depth=0, n_evaluated=0))[0] >= 1.0

    def render_examples(self, rows: list[tuple[Example, ExampleResult]]) -> str:
        parts = []
        for i, (ex, r) in enumerate(rows, 1):
            visits = trace_visits(r, self.module) if self.module else []
            if visits:  # a traced module: show what that stage saw and produced (its last visit(s) under loops)
                shown = visits[-max(1, self.visits):]
                if len(shown) == 1:
                    inputs, out = str(shown[0].get("input", ""))[: self.max_input_chars], str(shown[0].get("output", ""))
                else:
                    inputs = "\n\n".join(f"[visit {j + 1}/{len(visits)}]\n" + str(v.get("input", ""))[: self.max_input_chars // len(shown)]
                                         for j, v in enumerate(shown, len(visits) - len(shown)))
                    out = "\n\n".join(f"[visit {j + 1}/{len(visits)}]\n" + str(v.get("output", ""))[: self.max_output_chars // len(shown)]
                                      for j, v in enumerate(shown, len(visits) - len(shown)))
            else:
                inputs, out = "\n".join(f"{k}: {v}" for k, v in ex.inputs.items()), (r.output or "")
            out = out[: self.max_output_chars]
            parts.append(f"### Example {i}\nInputs:\n{inputs}\n\nResponse:\n{out}\n\nFeedback:\n{self.feedback(ex, r)}\n")
        return "\n".join(parts) if parts else "(no traces available)\n"

    def render_meta(self, tree: Tree, node: Node, seed: int) -> str:
        _, rows = self.pick(tree, node)
        cur = self.target(node)
        return self.meta_prompt.format(
            n=self.n, prompt=cur.template, description=self.description(tree, node),
            directive=self.render_examples(rows), placeholders=", ".join("{%s}" % p for p in cur.placeholders),
            seed=f" (variation batch {seed})" if self.calls > 1 else "",
        )

    def params(self) -> dict[str, Any]:
        p = super().params()
        p.pop("directive", None)
        p["minibatch"] = self.minibatch
        return p

    async def run_one(self, tree: Tree, node: Node) -> list[Node]:
        src, rows = self.pick(tree, node)
        params = {**self.params(), "source": src.id if src else None, "minibatch_ids": [ex.id for ex, _ in rows]}
        children = [tree.add_child(node, self.child_prompt(node, p), Origin(op=self.name, params=dict(params)))
                    for p in await self.propose(tree, node)]
        node.state = NodeState.EXPANDED
        tree._emit("expanded", node)
        return children
