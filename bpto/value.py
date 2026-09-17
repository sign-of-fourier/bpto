"""Node valuation for search: what number represents "how good is this node to have expanded"."""
from __future__ import annotations

from statistics import mean
from typing import Callable, Protocol

from .tree import Node, Tree


class Value(Protocol):
    """One target per node, or a list of them (`agg=list`): every descendant score becomes its own observation at
    the node's input, which is how `BOSelector` learns a node's expected yield *and* its spread (repeated
    observations at one x). A scalar aggregate such as `max` rewards a node for having been expanded and can lock
    the search onto it (`runs/hotpot_botree`, 2026-09-17)."""
    def __call__(self, node: Node, tree: Tree) -> float | list[float] | None: ...


class DescendantValue:
    """Aggregate of own/descendant scores `generations` below the node.

    generations=0 -> the node's own score; 1 -> children; 2 -> grandchildren.
    Returns None when nothing at that generation has been evaluated, so the node
    is simply absent from a BO training set.
    """

    def __init__(self, generations: int = 0, agg: Callable[[list[float]], float | list[float]] = max,
                 feasible_only: bool = False, op: str | None = None):
        self.generations, self.agg, self.feasible_only, self.op = generations, agg, feasible_only, op

    def __call__(self, node: Node, tree: Tree) -> float | None:
        nodes = [node] if self.generations == 0 else tree.descendants(node, self.generations)
        scores = [
            n.score for n in nodes
            if n.evaluated
            and (not self.feasible_only or n.evaluation.feasible)
            and (self.op is None or n.origin.op == self.op)
        ]
        return self.agg(scores) if scores else None


class SubtreeValue:
    """Aggregate over *all* evaluated descendants (any depth) - the BO target when an expansion
    is a pipeline of unknown/variable depth. `include_self` counts the node's own score too.
    `pipeline_only` restricts to nodes created by a Pipeline rooted at this node."""

    def __init__(self, agg: Callable[[list[float]], float | list[float]] = max, include_self: bool = False,
                 feasible_only: bool = False, pipeline_only: bool = False):
        self.agg, self.include_self, self.feasible_only, self.pipeline_only = agg, include_self, feasible_only, pipeline_only

    def __call__(self, node: Node, tree: Tree) -> float | None:
        nodes = tree.descendants(node) + ([node] if self.include_self else [])
        scores = [
            n.score for n in nodes
            if n.evaluated
            and (not self.feasible_only or n.evaluation.feasible)
            and (not self.pipeline_only or n is node or n.origin.params.get("pipeline_root") == node.id)
        ]
        return self.agg(scores) if scores else None


own_score = DescendantValue(0)
best_descendant = SubtreeValue(max)
best_child = DescendantValue(1, max)
best_grandchild = DescendantValue(2, max)
child_scores = DescendantValue(1, list)  # every child as an observation: the recommended expansion-selection target
mean_grandchild = DescendantValue(2, mean)
