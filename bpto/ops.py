"""Operations applied to sets of nodes: expansion (random / guided) and evaluation.

An op only ever sees `(tree, nodes)`; chunking and concurrency are handled by the base
class so a 10k-node evaluate never schedules 10k x |dataset| tasks at once.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .data import Dataset, Example
from .llm import BudgetExceeded, Completion, ModelConfig
from .metrics import mean_metrics, std_metrics
from .prompt import Prompt
from .scoring import ObjectiveContext, ScoreContext, run_scorer
from .tree import Evaluation, ExampleResult, Node, NodeState, Origin, Tree

log = logging.getLogger("bpto")


def _chunks(xs: list, size: int):
    for i in range(0, len(xs), size):
        yield xs[i:i + size]


class Op(ABC):
    name: str = "op"

    async def apply(self, tree: Tree, nodes: list[Node], chunk: int = 32, after_chunk=None) -> list[Node]:
        out: list[Node] = []
        for batch in _chunks(nodes, max(1, chunk)):
            results = await asyncio.gather(*(self.run_one(tree, n) for n in batch))
            for r in results:
                out.extend(r)
            if after_chunk:
                after_chunk()
        return out

    @abstractmethod
    async def run_one(self, tree: Tree, node: Node) -> list[Node]: ...


# ---- expansion -------------------------------------------------------------------

class Variants(BaseModel):
    prompts: list[str] = Field(description="Distinct rewritten prompt templates.")


class Expander(Op):
    """Proposes children for a node. Subclasses implement `propose`."""

    def params(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    async def propose(self, tree: Tree, node: Node) -> list[Prompt]: ...

    async def run_one(self, tree: Tree, node: Node) -> list[Node]:
        origin = Origin(op=self.name, params=self.params())
        children = [tree.add_child(node, p, origin) for p in await self.propose(tree, node)]
        node.state = NodeState.EXPANDED
        tree._emit("expanded", node)
        return children


class LLMExpander(Expander):
    """Asks the model for `n` distinct variants of the node's prompt in one structured call.

    `meta_prompt` is a template with {n}, {prompt}, {description}, {directive}, {placeholders}.
    Children that lose a placeholder are dropped. `calls` > 1 repeats the request (with a
    different seed hint) to get more diversity than one call yields.
    """

    name = "llm"
    meta_prompt = (
        "You are rewriting a prompt template that is used to instruct an LLM.\n"
        "The prompt is supposed to: {description}\n"
        "It must keep these placeholders exactly, in curly braces: {placeholders}\n\n"
        "Current prompt template:\n<prompt>\n{prompt}\n</prompt>\n\n"
        "{directive}\n"
        "Return {n} distinct prompt templates. Each must be complete and usable on its own.{seed}"
    )

    def __init__(self, n: int = 4, directive: str = "", calls: int = 1, meta_prompt: str | None = None,
                 config: ModelConfig | None = None):
        self.n, self.directive, self.calls, self.config = n, directive, calls, config
        if meta_prompt:
            self.meta_prompt = meta_prompt

    def params(self) -> dict[str, Any]:
        p = {"n": self.n, "directive": self.directive, "calls": self.calls}
        if self.config is not None:
            p["config"] = self.config.model_dump(exclude_unset=True)
        return p

    def render_meta(self, tree: Tree, node: Node, seed: int) -> str:
        return self.meta_prompt.format(
            n=self.n, prompt=node.prompt.template, description=tree.task.description or "(unspecified)",
            directive=self.directive, placeholders=", ".join("{%s}" % p for p in node.prompt.placeholders),
            seed=f" (variation batch {seed})" if self.calls > 1 else "",
        )

    async def propose(self, tree: Tree, node: Node) -> list[Prompt]:
        task = tree.task
        cfg = (task.expander_config or task.expander_client.default_config).merged(self.config)
        comps = await asyncio.gather(*(
            task.expander_client.complete(self.render_meta(tree, node, i), config=cfg, schema=Variants)
            for i in range(self.calls)
        ))
        required = set(node.prompt.placeholders)
        out, seen = [], {node.prompt.template}
        for c in comps:
            for t in c.parsed_as(Variants).prompts:
                p = Prompt(template=t.strip())
                if t in seen or not required.issubset(p.placeholders):
                    log.debug("dropping variant (dup or missing placeholders): %r", t[:80])
                    continue
                seen.add(t); out.append(p)
        return out[: self.n * self.calls]


class random(LLMExpander):
    """n substantively different rewrites, meaning preserved."""
    name = "random"

    def __init__(self, n: int = 4, calls: int = 1, **kw):
        super().__init__(n=n, calls=calls, directive=(
            "Produce rewrites that differ substantially from each other and from the original in wording, "
            "structure and framing, while preserving the intent."), **kw)


class guided(LLMExpander):
    """n rewrites steered by a directive, e.g. guided("make it more succinct")."""
    name = "guided"

    def __init__(self, directive: str, n: int = 3, calls: int = 1, **kw):
        super().__init__(n=n, calls=calls, directive=f"Instruction: {directive}", **kw)
        self.user_directive = directive

    def params(self) -> dict[str, Any]:
        return {**super().params(), "directive": self.user_directive}


class Pipeline(Op):
    """Several ops as one expansion: each op is applied to the frontier produced by the previous one.

        Pipeline([random(3), guided("more precise", 2), guided("shorter", 2), evaluate()])

    Each op sees only the previous op's output, so a trailing `evaluate()` scores the leaves of
    the new subtree, not the intermediate generations (use `SubtreeValue` for BO; add
    `evaluate()` after every stage if you want intermediates scored).
    From the tree's point of view this is one expansion of one node; every node it creates is
    tagged with `origin.params["pipeline"]` so values/selectors can attribute the whole subtree
    to the node that was expanded. Use with `SubtreeValue` for BO.
    """
    name = "pipeline"

    def __init__(self, ops: list[Op], name: str | None = None, chunk: int = 32):
        self.ops, self.chunk = ops, chunk
        self.name = name or "pipeline(" + ",".join(o.name for o in ops) + ")"

    async def run_one(self, tree: Tree, node: Node) -> list[Node]:
        frontier = [node]
        created: list[Node] = []
        for op in self.ops:
            out = await op.apply(tree, frontier, chunk=self.chunk)
            new = [n for n in out if n is not node and n.id not in {c.id for c in created}]
            for n in new:
                n.origin.params.setdefault("pipeline", self.name)
                n.origin.params.setdefault("pipeline_root", node.id)
            created.extend(new)
            frontier = out if out else frontier
        node.state = NodeState.EXPANDED
        tree._emit("expanded", node)
        return created


# ---- evaluation ------------------------------------------------------------------

class evaluate(Op):
    """Run the task dataset against each node and store per-example + aggregate results.

    dataset: override the task dataset (e.g. a cheap subset). Re-evaluating a node with a
    different dataset replaces its Evaluation; completions are cached so overlap is free.
    """
    name = "evaluate"

    def __init__(self, dataset: Dataset | None = None, sample: int | None = None, seed: int | None = None):
        self.dataset, self.sample, self.seed = dataset, sample, seed

    def _data(self, tree: Tree) -> Dataset:
        ds = self.dataset or tree.task.dataset
        return ds.sample(self.sample, self.seed) if self.sample else ds

    async def _one_example(self, tree: Tree, node: Node, ex: Example) -> ExampleResult:
        task = tree.task
        rendered = node.prompt.render(**ex.inputs)
        cfg = (task.config or task.client.default_config).merged(node.config)
        try:
            comp = await task.client.complete(rendered, config=cfg, schema=task.schema)
            metrics = await run_scorer(task.scorer, node.prompt, ex, comp, ScoreContext(task, task.client, rendered))
            parsed = comp.parsed.model_dump() if isinstance(comp.parsed, BaseModel) else comp.parsed
            return ExampleResult(example_id=ex.id, output=comp.text, parsed=parsed, metrics=metrics)
        except BudgetExceeded:
            raise
        except Exception as e:  # a failed example scores as its failure, not a crash of the run
            log.warning("example %s on node %s failed: %s", ex.id, node.id, e)
            return ExampleResult(example_id=ex.id, output="", metrics={}, error=f"{type(e).__name__}: {e}")

    async def run_one(self, tree: Tree, node: Node) -> list[Node]:
        data = self._data(tree)
        results = await asyncio.gather(*(self._one_example(tree, node, ex) for ex in data))
        rows = [r.metrics for r in results if r.error is None]
        agg = mean_metrics(rows)
        # errors count as zeros for every metric so a crashy prompt is penalised, not hidden
        if len(rows) < len(results) and agg:
            agg = {k: v * len(rows) / len(results) for k, v in agg.items()}
        ctx = ObjectiveContext(depth=node.depth, n_evaluated=len(tree.evaluated_nodes()))
        score, feasible = tree.task.objective(agg, ctx)
        node.evaluation = Evaluation(per_example=results, metrics=agg, metrics_std=std_metrics(rows),
                                     n=len(results), score=score, feasible=feasible,
                                     dataset_ids=[ex.id for ex in data])
        if node.state == NodeState.PROPOSED:
            node.state = NodeState.EVALUATED
        tree._emit("evaluated", node)
        return [node]
