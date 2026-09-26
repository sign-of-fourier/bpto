"""Official GEPA (`pip install gepa==0.1.4`) driven through bpto's `ModelClient`s.

GEPA's engine is synchronous; bpto's clients are async. `Bridge` runs one event loop in a background thread and
every task / reflection call is submitted to it, so the engine's `batch_evaluate` and `reflect_many` fan out
concurrently under the client's own semaphore, cache and `Budget` - the same machinery the bpto arms use.

Accounting (`count=`): "requested" (default) is GEPA's own unit, one metric call per row evaluated, cached or not.
Every arm of the benchmark runs this engine, so the unit is identical by construction; billed (fresh) calls are
reported beside it. "billed" passes fresh calls as `num_metric_calls`, but the engine honours that only on
minibatches - validation passes are always charged per row (engine `_evaluate_on_valset`) - so it is a mixed unit,
kept for the pre-test only. The engine checks the budget only between iterations, so it can overshoot by one
iteration; the task client's `Budget(max_calls=)` is the hard stop and `BudgetExceeded` propagates.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gepa import EvaluationBatch

from bpto.llm.base import BudgetExceeded, ModelClient, ModelConfig
from bpto.llm.cache import CompletionCache

from .common import feedback, parse_label, render


class LogOnlyCache(CompletionCache):
    """Records every completion, never serves one. For the reflector: GEPA's samplers can build the same reflection
    prompt twice (with q > 1 an iteration's minibatch chunks overlap the next iteration's, so a repeated parent meets
    the same rows), and GEPA expects a fresh temperature-1 sample each time, not a replay."""

    def get(self, key: str):
        return None


class Bridge:
    """A private event loop in a daemon thread; `run(coro)` blocks the calling (engine) thread until it resolves."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._t.start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._t.join(timeout=5)


def _text(prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n\n".join(m["content"] for m in prompt if isinstance(m.get("content"), str))


class ClientLM:
    """GEPA's `LanguageModel` (prompt -> str) plus `batch_complete`, which its default reflector uses to issue one
    iteration's reflections concurrently. The reflection prompt and output parsing stay GEPA's own."""

    def __init__(self, client: ModelClient, bridge: Bridge, config: ModelConfig | None = None):
        self.client, self.bridge, self.config = client, bridge, config
        self.calls = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        return self.batch_complete([prompt])[0]

    def batch_complete(self, messages_list: list) -> list[str]:
        self.calls += len(messages_list)

        async def go():
            outs = await asyncio.gather(*(self.client.complete(_text(m), config=self.config) for m in messages_list))
            return [o.text for o in outs]
        return self.bridge.run(go())


@dataclass
class Row:
    id: str
    narrative: str
    label: str


@dataclass
class Trace:
    row: Row
    prompt: str
    output: str
    pred: str | None
    error: str | None
    slot_restored: bool


@dataclass
class Stats:
    requested: int = 0          # rows GEPA asked to evaluate
    billed: int = 0             # fresh task calls
    cached: int = 0
    errors: int = 0
    slot_restored: int = 0      # evaluations whose instruction had lost {narrative}
    evaluations: list[dict] = field(default_factory=list)  # one per evaluate() call: size, fresh, concurrent batch id


class ClassifyAdapter:
    """GEPA adapter for single-prompt classification. Candidate = {"instruction": text with a {narrative} slot}."""

    propose_new_texts = None  # GEPA's own reflector and reflection prompt, not ours

    def __init__(self, client: ModelClient, labels: list[str], bridge: Bridge, *, config: ModelConfig | None = None,
                 count: str = "requested"):
        assert count in ("billed", "requested")
        self.client, self.labels, self.bridge, self.config, self.count = client, labels, bridge, config, count
        self.stats = Stats()
        self._batch_id = 0

    async def _one(self, instruction: str, row: Row) -> tuple[Trace, bool]:
        prompt, restored = render(instruction, row.narrative)
        try:
            comp = await self.client.complete(prompt, config=self.config)
        except BudgetExceeded:
            raise
        except Exception as e:  # recorded as a failed example (score 0), as bpto's evaluate does
            return Trace(row, prompt, "", None, f"{type(e).__name__}: {e}", restored), False
        return Trace(row, prompt, comp.text, parse_label(comp.text, self.labels), None, restored), comp.cached

    async def _evaluate(self, batch: list[Row], candidate: dict[str, str]) -> EvaluationBatch:
        instruction = candidate["instruction"]
        results = await asyncio.gather(*(self._one(instruction, r) for r in batch))
        traces = [t for t, _ in results]
        cached = sum(c for _, c in results)
        errors = sum(t.error is not None for t in traces)
        fresh = len(batch) - cached  # a failed call was still sent (ModelClient counts it before awaiting)
        s = self.stats
        s.requested += len(batch)
        s.billed += fresh
        s.cached += cached
        s.errors += errors
        s.slot_restored += int(any(t.slot_restored for t in traces))
        s.evaluations.append({"batch": self._batch_id, "rows": len(batch), "fresh": fresh})
        return EvaluationBatch(
            outputs=[t.output for t in traces],
            scores=[float(t.pred == t.row.label) for t in traces],
            trajectories=traces,
            num_metric_calls=fresh if self.count == "billed" else len(batch),
        )

    # GEPAAdapter
    def evaluate(self, batch: list[Row], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        self._batch_id += 1
        return self.bridge.run(self._evaluate(batch, candidate))

    def batch_evaluate(self, items: list[tuple[dict[str, str], list[Row]]]) -> list[EvaluationBatch]:
        """Every (candidate, rows) pair of one engine step at once: q parents' minibatches, q children's."""
        self._batch_id += 1

        async def go():
            return await asyncio.gather(*(self._evaluate(rows, cand) for cand, rows in items))
        return self.bridge.run(go())

    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: EvaluationBatch,
                                components_to_update: list[str]) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        recs = [{"Inputs": t.row.narrative, "Generated Outputs": t.output or "(no output)",
                 "Feedback": feedback(t.pred, t.row.label, t.error)} for t in eval_batch.trajectories]
        return {name: recs for name in components_to_update}


def run_official(seed_instruction: str, train: list[Row], val: list[Row], labels: list[str], *,
                 task_client: ModelClient, reflect_client: ModelClient, max_metric_calls: int, minibatch: int,
                 seed: int, task_config: ModelConfig | None = None, reflect_config: ModelConfig | None = None,
                 run_dir: str | None = None, sampling_strategy=None, count: str = "requested", **gepa_kw):
    """One official-GEPA run. Returns (GEPAResult, adapter, reflection LM) so the caller can audit the counters."""
    import gepa

    bridge = Bridge()
    try:
        adapter = ClassifyAdapter(task_client, labels, bridge, config=task_config, count=count)
        lm = ClientLM(reflect_client, bridge, reflect_config)
        if hasattr(sampling_strategy, "bind"):
            sampling_strategy.bind(bridge)
        stall = StallStopper(adapter)
        adapter.stall, adapter.logger = stall, gepa_kw.pop("logger", _Quiet())
        result = gepa.optimize(
            seed_candidate={"instruction": seed_instruction}, trainset=train, valset=val, adapter=adapter,
            reflection_lm=lm, reflection_minibatch_size=minibatch, max_metric_calls=max_metric_calls, seed=seed,
            run_dir=run_dir, sampling_strategy=sampling_strategy, raise_on_exception=True, stop_callbacks=[stall],
            logger=adapter.logger, **gepa_kw)
        return result, adapter, lm
    finally:
        bridge.close()


class StallStopper:
    """Stop when `patience` consecutive iterations bill no task call. Under count="billed" a fully cached iteration
    costs nothing, so a reflector that keeps proposing texts already evaluated would never exhaust max_metric_calls."""

    def __init__(self, adapter: ClassifyAdapter, patience: int = 20):
        self.adapter, self.patience = adapter, patience
        self._last, self._still, self._i = -1, 0, -1
        self.fired = False

    def __call__(self, state) -> bool:
        if state.i != self._i:
            self._i = state.i
            billed = self.adapter.stats.billed
            self._still = self._still + 1 if billed == self._last else 0
            self._last = billed
        self.fired = self._still >= self.patience
        return self.fired


class _Quiet:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, message: str):
        self.lines.append(message)
