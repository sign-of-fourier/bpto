"""The GEPA loop as a `run()` schedule: select parents from the Pareto pool -> reflect -> evaluate the
children on a minibatch -> children that beat their parent on that minibatch get the full pareto set.

Simplifications vs Agrawal et al. 2025: no merge (module round-robin via `modules=`), pareto set =
the task dataset unless `pareto_dataset` is given, and `parents_per_round` parents per round
(1 = GEPA's sequential loop). The gate is GEPA's (strict improvement on one minibatch) unless `extend=` asks for rungs.
Rollouts = unique model calls (`client.usage.calls`); the reflection
minibatch is read from the parent's cached evaluation instead of being re-run.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .. import select as S
from ..data import Dataset
from ..llm.base import ModelConfig
from ..ops import evaluate
from ..search import Step, step
from ..tree import Node, Tree
from .reflect import Context, Feedback, Passed, ReflectiveExpander, default_feedback
from .select import example_scores, pareto_sample


def minibatch_for(round_: int, dataset: Dataset, size: int, seed: int) -> Dataset:
    return dataset.sample(size, seed=seed * 100_003 + round_)


def beats_parent(tree: Tree, child: Node) -> bool:
    """Child's mean score on its (minibatch) evaluation > parent's mean on the same example ids."""
    parent = tree.nodes.get(child.parent_id) if child.parent_id else None
    if child.evaluation is None or parent is None or parent.evaluation is None:
        return False
    ids = child.evaluation.dataset_ids
    ps = example_scores(tree, parent)
    if not all(i in ps for i in ids):
        return False
    return child.evaluation.score > sum(ps[i] for i in ids) / max(1, len(ids))


def _accepted(tree: Tree, pool: list[Node]) -> list[Node]:
    return [n for n in pool if beats_parent(tree, n)]


def paired_change(tree: Tree, child: Node) -> tuple[int, int] | None:
    """(fixed, broke): rows of the child's evaluation where its per-example score is above / below the parent's on the
    same row. None if the two cannot be paired. fixed == broke == 0 means the rewrite changed no score (inert)."""
    parent = tree.nodes.get(child.parent_id) if child.parent_id else None
    if child.evaluation is None or parent is None or parent.evaluation is None:
        return None
    ps, cs = example_scores(tree, parent), example_scores(tree, child)
    if not all(i in ps for i in cs):
        return None
    return sum(cs[i] > ps[i] for i in cs), sum(cs[i] < ps[i] for i in cs)


def _net(tree: Tree, child: Node) -> float:
    """Child minus parent, summed over the child's rows (in rows, for a 0/1 metric)."""
    ps, cs = example_scores(tree, tree.nodes[child.parent_id]), example_scores(tree, child)
    return sum(cs[i] - ps[i] for i in cs)


def unclear(tree: Tree, child: Node, margin: float = 1.0) -> bool:
    """Some rows changed, but the net over the child's rows is within `margin` rows of zero either way (0 = exact
    ties only): the case a larger (nested) batch can resolve. Inert children and clear wins / losses are decided."""
    pc = paired_change(tree, child)
    return pc is not None and pc != (0, 0) and abs(_net(tree, child)) <= margin + 1e-9


def gate_steps(r: int, full: Dataset, minibatch: int, seed: int, extend: Sequence[int] = (), margin: float = 1.0,
               tag: str | None = None, op: str = "reflect") -> list[Step]:
    """The survive seat for round r: evaluate this round's new children (origin.op == `op`) on a `minibatch`-row
    batch and give the full set to those that beat their parent on it. `extend=()` is GEPA's single strict gate.

    With `extend=(k, ...)`, a child whose result is `unclear` (a tie with changed rows, or a net within `margin` rows
    either way) is re-evaluated on each larger rung in turn - prefixes of one seeded shuffle, so each contains the
    last and its rows hit the completion cache; the parent's rows come from its full evaluation. It is accepted only
    if it beats its parent on the largest batch it reached. Each child's outcome (inert / passed / rejected, rows,
    fixed, broke) is recorded in `tree.meta["gate"][node_id]`."""
    sizes = [minibatch, *extend]
    if extend and (any(b <= a for a, b in zip(sizes, sizes[1:])) or sizes[-1] >= len(full)):
        raise ValueError(f"extend rungs must increase from minibatch={minibatch} and stay below the full set "
                         f"({len(full)} rows): {list(extend)}")
    tag = tag or f"gepa/r{r}"
    ids = {ex.id for ex in full}
    rungs = [frozenset(ex.id for ex in minibatch_for(r, full, k, seed)) for k in sizes]
    is_new = lambda n: n.origin.op == op and n.evaluation is None
    on_minibatch = lambda n: n.origin.op == op and n.evaluation is not None and not ids.issubset(set(n.evaluation.dataset_ids))
    steps = [step(evaluate(dataset=minibatch_for(r, full, minibatch, seed)), S.where(is_new), name=f"{tag}/minibatch")]
    for prev, k in zip(rungs, sizes[1:]):
        tied = lambda t, prev=prev: [n for n in S.where(on_minibatch)(t)
                                     if set(n.evaluation.dataset_ids) == prev and unclear(t, n, margin)]
        steps.append(step(evaluate(dataset=minibatch_for(r, full, k, seed)), tied, name=f"{tag}/extend{k}"))

    def decide(t: Tree) -> list[Node]:
        pool = S.where(on_minibatch)(t)
        accepted = _accepted(t, pool)
        if extend:  # record this round's children (the ones evaluated on one of its rungs)
            log = t.meta.setdefault("gate", {})
            for n in pool:
                if frozenset(n.evaluation.dataset_ids) in rungs:
                    fixed, broke = paired_change(t, n) or (0, 0)
                    outcome = "passed" if n in accepted else "inert" if (fixed, broke) == (0, 0) else "rejected"
                    log[n.id] = {"round": r, "outcome": outcome, "rows": len(n.evaluation.dataset_ids),
                                 "fixed": fixed, "broke": broke}
        return accepted
    steps.append(step(evaluate(dataset=full), decide, name=f"{tag}/full"))
    return steps


def gepa(feedback: Feedback | dict[str, Feedback] = default_feedback, *, minibatch: int = 3, parents_per_round: int = 1,
         n: int = 1, mode: str = "weighted", pareto_dataset: Dataset | None = None, seed: int = 0,
         expander_config: ModelConfig | None = None, modules: Sequence[str] | None = None,
         passed: Passed | dict[str, Passed] | None = None, reflect_rows: int | None = None,
         extend: Sequence[int] = (), margin: float = 1.0, context: Context | None = None) -> Callable[[Tree, int], list[Step]]:
    """Schedule factory: `run(tree, gepa(feedback), stop=Stop(...))`. Round 0 evaluates the root.

    `modules`: on Program nodes, GEPA's round-robin - round r rewrites `modules[r % len(modules)]` (one module per
    child); `feedback` / `passed` may then be dicts keyed by module. None rewrites the entry module every round.
    `minibatch` is the gate's batch; `reflect_rows` the number of traces shown to the reflector (default: the same,
    as in GEPA). `extend` / `margin`: larger nested batches for children whose minibatch result is unclear (a tie, or
    within `margin` rows either way; see `gate_steps`).
    `context`: text above the reflector's examples (see `ReflectiveExpander`)."""
    def schedule(tree: Tree, r: int) -> list[Step]:
        module = modules[r % len(modules)] if modules else None
        full = pareto_dataset or tree.task.dataset
        ids = {ex.id for ex in full}
        if not tree.evaluated_nodes():
            return [step(evaluate(dataset=full), S.root, name="gepa/root")]
        tag = f"gepa/r{r}"
        return [
            step(ReflectiveExpander(feedback, minibatch=reflect_rows or minibatch, n=n, seed=seed + r,
                                    config=expander_config, module=module, passed=passed, context=context),
                 pareto_sample(parents_per_round, mode=mode, seed=seed + r, ids=ids), name=f"{tag}/reflect"),
            *gate_steps(r, full, minibatch, seed, extend, margin, tag=tag),
        ]
    return schedule
