"""The GEPA loop as a `run()` schedule: select parents from the Pareto pool -> reflect -> evaluate the
children on a minibatch -> children that beat their parent on that minibatch get the full pareto set.

Simplifications vs Agrawal et al. 2025: no merge (module round-robin via `modules=`), pareto set =
the task dataset unless `pareto_dataset` is given, and `parents_per_round` parents per round
(1 = GEPA's sequential loop). Rollouts = unique model calls (`client.usage.calls`); the reflection
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
from .reflect import Feedback, Passed, ReflectiveExpander, default_feedback
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


def gepa(feedback: Feedback | dict[str, Feedback] = default_feedback, *, minibatch: int = 3, parents_per_round: int = 1,
         n: int = 1, mode: str = "weighted", pareto_dataset: Dataset | None = None, seed: int = 0,
         expander_config: ModelConfig | None = None, modules: Sequence[str] | None = None,
         passed: Passed | dict[str, Passed] | None = None) -> Callable[[Tree, int], list[Step]]:
    """Schedule factory: `run(tree, gepa(feedback), stop=Stop(...))`. Round 0 evaluates the root.

    `modules`: on Program nodes, GEPA's round-robin - round r rewrites `modules[r % len(modules)]` (one module per
    child); `feedback` / `passed` may then be dicts keyed by module. None rewrites the entry module every round."""
    def schedule(tree: Tree, r: int) -> list[Step]:
        module = modules[r % len(modules)] if modules else None
        full = pareto_dataset or tree.task.dataset
        ids = {ex.id for ex in full}
        if not tree.evaluated_nodes():
            return [step(evaluate(dataset=full), S.root, name="gepa/root")]
        mb = minibatch_for(r, full, minibatch, seed)
        tag = f"gepa/r{r}"
        is_new = lambda n: n.origin.op == "reflect" and n.evaluation is None
        on_minibatch = lambda n: n.origin.op == "reflect" and n.evaluation is not None and not ids.issubset(set(n.evaluation.dataset_ids))
        return [
            step(ReflectiveExpander(feedback, minibatch=minibatch, n=n, seed=seed + r, config=expander_config,
                                    module=module, passed=passed),
                 pareto_sample(parents_per_round, mode=mode, seed=seed + r, ids=ids), name=f"{tag}/reflect"),
            step(evaluate(dataset=mb), S.where(is_new), name=f"{tag}/minibatch"),
            step(evaluate(dataset=full), lambda t: _accepted(t, S.where(on_minibatch)(t)), name=f"{tag}/full"),
        ]
    return schedule
