"""Pareto-pool candidate selection, after GEPA (Agrawal et al. 2025).

A candidate is a node evaluated on the *pareto set* (the full task dataset unless the loop was
given another). Per-example scalar scores come from the task objective applied to each example's
metrics, so they are re-derivable and consistent with `node.score`.

    pool     = candidates that are the best on at least one example, minus dominated ones
    weighted = sample from the pool with probability ∝ number of examples the candidate wins

The sampling `mode` is a switch so each ingredient can be ablated or swapped for a surrogate:
weighted (GEPA) | uniform (pool, no weights) | best (argmax mean, no stochasticity) |
all (every candidate, uniform - no Pareto at all).
"""
from __future__ import annotations

import random as _random
from typing import Callable

from ..scoring import ObjectiveContext
from ..tree import Node, Tree

Selector = Callable[[Tree], list[Node]]


def example_scores(tree: Tree, node: Node, metric: str | None = None) -> dict[str, float]:
    """example_id -> per-example scalar for one evaluated node (errors score 0).

    Default: the task objective applied to the example's metrics. `metric` names a raw metric instead -
    for a constrained objective (e.g. "minimise tokens s.t. f1 >= floor") the scalar is the same on
    every example and carries no per-example information, but the accuracy metric does."""
    ev = node.evaluation
    if ev is None:
        return {}
    ctx = ObjectiveContext(depth=node.depth, n_evaluated=len(tree.evaluated_nodes()))
    out = {}
    for r in ev.per_example:
        if r.error:
            out[r.example_id] = 0.0
        elif metric is not None:
            out[r.example_id] = float(r.metrics.get(metric, 0.0))
        else:
            out[r.example_id] = tree.task.objective(r.metrics, ctx)[0]
    return out


def candidates(tree: Tree, ids: set[str] | None = None) -> list[Node]:
    """Nodes evaluated on (at least) the pareto set. Minibatch-only evaluations do not qualify."""
    ids = ids if ids is not None else {ex.id for ex in tree.task.dataset}
    return [n for n in tree.evaluated_nodes() if ids.issubset(set(n.evaluation.dataset_ids))]


def pareto_pool(tree: Tree, ids: set[str] | None = None, metric: str | None = None) -> tuple[list[Node], dict[str, int]]:
    """(pool, wins): winners on >= 1 example with dominated candidates removed; wins = examples won.
    `metric` is passed to `example_scores`."""
    ids = ids if ids is not None else {ex.id for ex in tree.task.dataset}
    cands = candidates(tree, ids)
    if not cands:
        return [], {}
    scores = {n.id: example_scores(tree, n, metric) for n in cands}
    best = {i: max(scores[n.id].get(i, 0.0) for n in cands) for i in ids}
    # a "win" needs a positive best: tying at 0 on an example nobody solves is not evidence of anything
    wins = {n.id: sum(1 for i in ids if best[i] > 0 and scores[n.id].get(i, 0.0) >= best[i]) for n in cands}
    winners = [n for n in cands if wins[n.id] > 0] or cands  # nothing solved yet: fall back to everyone
    if not any(wins.values()):
        wins = {n.id: 1 for n in cands}

    def dominated(a: Node) -> bool:
        sa = scores[a.id]
        for b in winners:
            if b is a:
                continue
            sb = scores[b.id]
            if all(sb.get(i, 0.0) >= sa.get(i, 0.0) for i in ids) and any(sb.get(i, 0.0) > sa.get(i, 0.0) for i in ids):
                return True
        return False

    pool = [n for n in winners if not dominated(n)]
    return pool, {n.id: wins[n.id] for n in pool}


def pareto_sample(k: int = 1, mode: str = "weighted", seed: int | None = None, ids: set[str] | None = None,
                  metric: str | None = None) -> Selector:
    """Selector: k parents (without replacement) drawn from the Pareto pool.

    mode: "weighted" (∝ examples won, GEPA), "uniform", "best" (top-k by mean score, deterministic),
    "all" (uniform over every candidate, ignoring the Pareto pool).
    metric: per-example score for pool membership (see `example_scores`); "best" still ranks by `node.score`.
    """
    if mode not in {"weighted", "uniform", "best", "all"}:
        raise ValueError(f"unknown mode {mode!r}")

    def _sel(tree: Tree) -> list[Node]:
        rnd = _random.Random(seed if seed is not None else len(tree))
        if mode == "all":
            pool, weights = candidates(tree, ids), None
        else:
            pool, wins = pareto_pool(tree, ids, metric)
            weights = [wins[n.id] for n in pool] if mode == "weighted" else None
        if not pool:
            return []
        if mode == "best":
            return sorted(pool, key=lambda n: n.score, reverse=True)[:k]
        chosen: list[Node] = []
        pool, weights = list(pool), (list(weights) if weights else None)
        while pool and len(chosen) < k:
            i = rnd.choices(range(len(pool)), weights=weights)[0] if weights else rnd.randrange(len(pool))
            chosen.append(pool.pop(i))
            if weights:
                weights.pop(i)
        return chosen
    return _sel
