"""Optional driver: run an op schedule round after round with stop conditions, checkpoint
and resume. Everything here is sugar over `tree.apply`; a script calling `apply` by hand is
equally valid.

    sched = [step(random(4)), step(guided("more accurate", 3)), step(evaluate(), select.unevaluated)]
    await run(tree, sched, stop=Stop(rounds=10), checkpoint="tree.json")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import select as S
from .data import Dataset
from .llm import BudgetExceeded
from .ops import Op, evaluate
from .tree import Tree

log = logging.getLogger("bpto.search")


@dataclass
class Step:
    op: Op
    select: Callable = S.leaves
    chunk: int = 32
    name: str = ""

    def __post_init__(self):
        self.name = self.name or f"{self.op.name}"


def step(op: Op, select: Callable = S.leaves, chunk: int = 32, name: str = "") -> Step:
    return Step(op, select, chunk, name)


Schedule = Sequence[Step] | Callable[[Tree, int], Sequence[Step]]


@dataclass
class Stop:
    """Any condition that is met stops the run (checked between steps)."""
    rounds: int | None = None
    max_nodes: int | None = None
    max_depth: int | None = None
    max_seconds: float | None = None
    no_improvement_rounds: int | None = None  # rounds since tree.best() last improved
    until: Callable[[Tree], bool] | None = None

    def reason(self, tree: Tree, round_idx: int, started: float, stale_rounds: int) -> str | None:
        if self.rounds is not None and round_idx >= self.rounds:
            return f"rounds={self.rounds}"
        if self.max_nodes is not None and len(tree) >= self.max_nodes:
            return f"max_nodes={self.max_nodes}"
        if self.max_depth is not None and tree.max_depth >= self.max_depth:
            return f"max_depth={self.max_depth}"
        if self.max_seconds is not None and time.monotonic() - started >= self.max_seconds:
            return f"max_seconds={self.max_seconds}"
        if self.no_improvement_rounds is not None and stale_rounds >= self.no_improvement_rounds:
            return f"no improvement for {stale_rounds} rounds"
        if self.until is not None and self.until(tree):
            return "until() satisfied"
        return None


@dataclass
class RunResult:
    tree: Tree
    rounds: int
    stopped_because: str
    seconds: float
    history: list[dict] = field(default_factory=list)  # per-step summaries


async def run(tree: Tree, schedule: Schedule, stop: Stop | None = None, checkpoint: str | Path | None = None,
              on_step: Callable[[Tree, int, Step], None] | None = None) -> RunResult:
    """Apply `schedule` repeatedly until `stop` (default: one round) or the client budget trips.

    Progress is recorded in `tree.meta["run"]` and, with `checkpoint`, saved after every chunk.
    Re-running on a loaded tree continues from the step after the last completed one. A step
    interrupted mid-way is re-run; that is safe for evaluate (cached) and for expansion under the
    usual selectors (expanded nodes are no longer leaves/unexpanded).
    """
    stop = stop or Stop(rounds=1)
    started = time.monotonic()
    state = tree.meta.setdefault("run", {"round": 0, "step": 0, "best": None, "stale": 0})
    history: list[dict] = []
    reason = None

    def steps_for(r: int) -> Sequence[Step]:
        return schedule(tree, r) if callable(schedule) else schedule

    while True:
        r = state["round"]
        reason = stop.reason(tree, r, started, state["stale"])
        if reason:
            break
        steps = steps_for(r)
        interrupted = False
        while state["step"] < len(steps):
            st = steps[state["step"]]
            t0, n_before, calls_before = time.monotonic(), len(tree), tree.task.client.usage.calls
            try:
                touched = await tree.apply(st.op, st.select, chunk=st.chunk, checkpoint=checkpoint)
            except BudgetExceeded as e:
                reason, interrupted = f"budget: {e}", True
                break
            rec = {"round": r, "step": st.name, "touched": len(touched), "new_nodes": len(tree) - n_before,
                   "calls": tree.task.client.usage.calls - calls_before, "seconds": round(time.monotonic() - t0, 2)}
            history.append(rec)
            log.info("round %d %s: %s", r, st.name, rec)
            if on_step:
                on_step(tree, r, st)
            state["step"] += 1
            if checkpoint:
                tree.save(checkpoint)
            reason = stop.reason(tree, r, started, state["stale"])
            if reason and state["step"] < len(steps):
                interrupted = True  # stop mid-round; resume picks up at this step
                break
        if interrupted:
            break
        best = tree.best()
        best_score = best.score if best else None
        state["stale"] = 0 if (best_score is not None and (state["best"] is None or best_score > state["best"])) else state["stale"] + 1
        state["best"] = best_score if best_score is not None else state["best"]
        state["round"], state["step"] = r + 1, 0
        if checkpoint:
            tree.save(checkpoint)

    if checkpoint:
        tree.save(checkpoint)
    return RunResult(tree, state["round"], reason or "done", time.monotonic() - started, history)


def successive_halving(sizes: Sequence[int | None], keep: float = 0.5, seed: int = 0,
                       among: Callable = S.unevaluated, dataset: Dataset | None = None) -> list[Step]:
    """Multi-fidelity evaluation as schedule steps.

    sizes=[8, 32, None]: evaluate every unevaluated node on 8 examples, the best `keep` fraction on
    32, the best `keep` of those on the full set. Samples are nested (same seed), so each rung only
    pays for the new examples. A node's Evaluation is replaced at each rung it survives.
    """
    steps = [step(evaluate(dataset=dataset, sample=sizes[0], seed=seed), among, name=f"eval[{sizes[0]}]")]
    prev_n = sizes[0]
    for n in sizes[1:]:
        steps.append(step(evaluate(dataset=dataset, sample=n, seed=seed), _rung_selector(prev_n, keep, dataset), name=f"eval[{n}]"))
        prev_n = n
    return steps


def _rung_selector(prev_size: int | None, keep: float, dataset: Dataset | None = None) -> Callable:
    def _sel(tree: Tree):
        full = len(dataset if dataset is not None else tree.task.dataset)
        target = full if prev_size is None else min(prev_size, full)
        pool = [n for n in tree.evaluated_nodes() if n.evaluation.n == target]
        return S.top_frac(keep, among=lambda t: pool)(tree)
    return _sel
