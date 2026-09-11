"""Selectors: callables `tree -> list[Node]`, composable by wrapping."""
from __future__ import annotations

from typing import Callable

from .tree import Node, NodeState, Tree

Selector = Callable[[Tree], list[Node]]


def all_nodes(tree: Tree) -> list[Node]:
    return list(tree)


def leaves(tree: Tree) -> list[Node]:
    return tree.leaves


def unevaluated(tree: Tree) -> list[Node]:
    return [n for n in tree if not n.evaluated]


def evaluated(tree: Tree) -> list[Node]:
    return tree.evaluated_nodes()


def unexpanded(tree: Tree) -> list[Node]:
    return [n for n in tree if n.state != NodeState.EXPANDED]


def root(tree: Tree) -> list[Node]:
    return [tree.root]


def where(pred: Callable[[Node], bool], among: Selector = all_nodes) -> Selector:
    return lambda tree: [n for n in among(tree) if pred(n)]


def depth(d: int, among: Selector = all_nodes) -> Selector:
    return where(lambda n: n.depth == d, among)


def feasible(among: Selector = evaluated) -> Selector:
    return where(lambda n: n.evaluated and n.evaluation.feasible, among)


def top_k(k: int, among: Selector = leaves, value=None) -> Selector:
    """Best k of `among` under `value` (default: own score). Nodes without a value are dropped."""
    from .value import own_score
    v = value or own_score

    def _sel(tree: Tree) -> list[Node]:
        scored = [(v(n, tree), n) for n in among(tree)]
        scored = [(s, n) for s, n in scored if s is not None]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [n for _, n in scored[:k]]
    return _sel


def top_frac(frac: float, among: Selector = leaves, value=None, at_least: int = 1) -> Selector:
    """Best ceil(frac * |among|) nodes (at least `at_least`)."""
    import math

    def _sel(tree: Tree) -> list[Node]:
        pool = among(tree)
        return top_k(max(at_least, math.ceil(frac * len(pool))), among=lambda t: pool, value=value)(tree)
    return _sel


def pareto(maximize: dict[str, bool], among: Selector = evaluated) -> Selector:
    from .scoring import pareto_front

    def _sel(tree: Tree) -> list[Node]:
        nodes = [n for n in among(tree) if n.evaluated]
        keep = set(pareto_front({n.id: n.evaluation.metrics for n in nodes}, maximize))
        return [n for n in nodes if n.id in keep]
    return _sel


def union(*selectors: Selector) -> Selector:
    def _sel(tree: Tree) -> list[Node]:
        seen, out = set(), []
        for s in selectors:
            for n in s(tree):
                if n.id not in seen:
                    seen.add(n.id); out.append(n)
        return out
    return _sel


def limit(k: int, among: Selector) -> Selector:
    return lambda tree: among(tree)[:k]
