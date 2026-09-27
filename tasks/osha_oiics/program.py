"""The two-step program and its adapter for official GEPA. A candidate has two components:

  route  {narrative} -> one of the 18 major groups (named by title or two-digit number)
  code   {narrative}, {group}, {codes} -> one detailed code of that group. {codes} is filled by code with the chosen
         group's "code: title" lines (retrieval, not optimized); a group with one code skips the call.

Score = the detailed code is exact (GEPA sees only this). Group accuracy is recorded beside it, from the same calls.
Each row is up to two task-model calls; GEPA's metric unit stays one per row evaluated.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gepa import EvaluationBatch

from bpto.llm.base import BudgetExceeded, ModelClient, ModelConfig
from tasks.osha_sir.common import parse_label
from tasks.osha_sir.official_gepa import Bridge, ClientLM, StallStopper, _Quiet

COMPONENTS = ("route", "code")
SLOTS = {"route": ("narrative",), "code": ("narrative", "group", "codes")}
SLOT_LABELS = {"narrative": "Narrative", "group": "Major group", "codes": "Codes"}


def render(template: str, values: dict[str, str]) -> tuple[str, bool]:
    """Fill {slot}s by plain replacement (other braces are left alone); a slot a rewrite dropped is appended."""
    out, restored = template, False
    for k, v in values.items():
        if "{" + k + "}" in out:
            out = out.replace("{" + k + "}", v)
        else:
            out, restored = f"{out.rstrip()}\n\n{SLOT_LABELS[k]}: {v}", True
    return out, restored


def parse_group(text: str, groups: dict[str, str]) -> str | None:
    """A group title (osha_sir's lenient parser), else the last two-digit group number in the text."""
    by_title = {t: g for g, t in groups.items()}
    hit = parse_label(text, list(by_title))
    if hit is not None:
        return by_title[hit]
    nums = [n for n in re.findall(r"(?<!\d)(\d{2})(?!\d)", text) if n in groups]
    return nums[-1] if nums else None


def parse_code(text: str, codes: dict[str, str]) -> str | None:
    """The last four-digit number that is one of the group's codes, else a title (lenient parser)."""
    nums = [n for n in re.findall(r"(?<!\d)(\d{4})(?!\d)", text) if n in codes]
    if nums:
        return nums[-1]
    by_title = {t: c for c, t in codes.items()}
    hit = parse_label(text, list(by_title))
    return by_title[hit] if hit is not None else None


def code_lines(codes: dict[str, str]) -> str:
    return "\n".join(f"{c}: {t}" for c, t in codes.items())


@dataclass
class Row:
    id: str
    narrative: str
    group: str
    code: str


@dataclass
class Trace:
    row: Row
    route_out: str = ""
    group: str | None = None
    code_out: str = ""
    code: str | None = None
    code_called: bool = False
    error: str | None = None
    slot_restored: bool = False
    tokens: dict = field(default_factory=dict)  # {"route": [in, out], "code": [in, out]} for fresh or cached calls


@dataclass
class Stats:
    requested: int = 0          # rows GEPA asked to evaluate
    billed: int = 0             # fresh task calls (up to two per row)
    cached: int = 0
    errors: int = 0
    slot_restored: int = 0
    group_correct: int = 0
    code_correct: int = 0
    evaluations: list[dict] = field(default_factory=list)


class TwoStepAdapter:
    propose_new_texts = None  # GEPA's own reflector and reflection prompt

    def __init__(self, client: ModelClient, book: dict, bridge: Bridge | None, *, config: ModelConfig | None = None,
                 oracle_group: bool = False):
        self.client, self.bridge, self.config, self.oracle_group = client, bridge, config, oracle_group
        self.groups, self.codes = book["groups"], book["codes"]
        self.stats = Stats()
        self._batch_id = 0

    def _group_text(self, g: str) -> str:
        return f"{g} {self.groups[g]}"

    async def _call(self, t: Trace, step: str, prompt: str) -> tuple[str, bool]:
        comp = await self.client.complete(prompt, config=self.config)
        t.tokens[step] = [comp.input_tokens, comp.output_tokens]
        return comp.text, comp.cached

    async def run_row(self, cand: dict[str, str], row: Row) -> tuple[Trace, int, int]:
        """(trace, fresh calls, cached calls). `oracle_group` skips step 1 and hands step 2 the gold group."""
        t, fresh, cached = Trace(row), 0, 0
        try:
            if self.oracle_group:
                t.group = row.group
            else:
                prompt, r1 = render(cand["route"], {"narrative": row.narrative})
                t.slot_restored |= r1
                t.route_out, hit = await self._call(t, "route", prompt)
                fresh, cached = fresh + (not hit), cached + hit
                t.group = parse_group(t.route_out, self.groups)
            if t.group is None:
                return t, fresh, cached
            codes = self.codes[t.group]
            if len(codes) == 1:
                t.code = next(iter(codes))
                return t, fresh, cached
            prompt, r2 = render(cand["code"], {"narrative": row.narrative, "group": self._group_text(t.group),
                                               "codes": code_lines(codes)})
            t.slot_restored |= r2
            t.code_called = True
            t.code_out, hit = await self._call(t, "code", prompt)
            fresh, cached = fresh + (not hit), cached + hit
            t.code = parse_code(t.code_out, codes)
        except BudgetExceeded:
            raise
        except Exception as e:  # a failed call scores 0, as bpto's evaluate does
            t.error = f"{type(e).__name__}: {e}"
            fresh += 1
        return t, fresh, cached

    async def _evaluate(self, batch: list[Row], candidate: dict[str, str]) -> EvaluationBatch:
        res = await asyncio.gather(*(self.run_row(candidate, r) for r in batch))
        traces = [t for t, _, _ in res]
        s = self.stats
        s.requested += len(batch)
        s.billed += sum(f for _, f, _ in res)
        s.cached += sum(c for _, _, c in res)
        s.errors += sum(t.error is not None for t in traces)
        s.slot_restored += int(any(t.slot_restored for t in traces))
        s.group_correct += sum(t.group == t.row.group for t in traces)
        s.code_correct += sum(t.code == t.row.code for t in traces)
        s.evaluations.append({"batch": self._batch_id, "rows": len(batch), "fresh": sum(f for _, f, _ in res)})
        return EvaluationBatch(outputs=[f"{t.group}|{t.code}" for t in traces],
                               scores=[float(t.code == t.row.code) for t in traces],
                               trajectories=traces, num_metric_calls=len(batch))

    # GEPAAdapter
    def evaluate(self, batch: list[Row], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        self._batch_id += 1
        return self.bridge.run(self._evaluate(batch, candidate))

    def batch_evaluate(self, items: list[tuple[dict[str, str], list[Row]]]) -> list[EvaluationBatch]:
        self._batch_id += 1

        async def go():
            return await asyncio.gather(*(self._evaluate(rows, cand) for cand, rows in items))
        return self.bridge.run(go())

    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: EvaluationBatch,
                                components_to_update: list[str]) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        out = {}
        for name in components_to_update:
            out[name] = [self._route_record(t) if name == "route" else self._code_record(t)
                         for t in eval_batch.trajectories]
        return out

    def _gold(self, t: Trace) -> str:
        return f"{t.row.code} {self.codes[t.row.group][t.row.code]} (major group {self._group_text(t.row.group)})"

    def _route_record(self, t: Trace) -> dict:
        gold = self._group_text(t.row.group)
        if t.error and not t.route_out:
            fb = f"The call failed ({t.error}); gold group was {gold}."
        elif t.group is None:
            fb = f"No major group could be read from the answer; gold was {gold}. Answer with one major group."
        elif t.group == t.row.group:
            fb = f"Correct: {gold}."
        else:
            fb = f"Predicted {self._group_text(t.group)}; gold was {gold} (detailed code {self._gold(t)})."
        return {"Inputs": t.row.narrative, "Generated Outputs": t.route_out or "(no output)", "Feedback": fb}

    def _code_record(self, t: Trace) -> dict:
        given = self._group_text(t.group) if t.group else "(none: step 1 gave no group)"
        inputs = f"Narrative: {t.row.narrative}\nMajor group given by step 1: {given}"
        if t.group != t.row.group:
            fb = (f"Step 1 routed this to {given}, but the gold code is {self._gold(t)}, so this step could not be "
                  "right; nothing to learn here about choosing within a group.")
        elif not t.code_called:
            fb = f"The group has a single code; it was assigned without a call. Gold: {self._gold(t)}."
        elif t.error:
            fb = f"The call failed ({t.error}); gold was {self._gold(t)}."
        elif t.code is None:
            fb = f"No code of the group could be read from the answer; gold was {self._gold(t)}. Answer with one code."
        elif t.code == t.row.code:
            fb = f"Correct: {self._gold(t)}."
        else:
            fb = f"Predicted {t.code} {self.codes[t.group][t.code]}; gold was {self._gold(t)}."
        return {"Inputs": inputs, "Generated Outputs": t.code_out or "(no output)", "Feedback": fb}


def run_official(seed: dict[str, str], train: list[Row], val: list[Row], book: dict, *, task_client: ModelClient,
                 reflect_client: ModelClient, max_metric_calls: int, minibatch: int, seed_value: int,
                 task_config: ModelConfig | None = None, reflect_config: ModelConfig | None = None,
                 run_dir: str | None = None, sampling_strategy=None, **gepa_kw):
    """One official-GEPA run on the two-step program (osha_sir's bridge; GEPA's round-robin over components)."""
    import gepa

    bridge = Bridge()
    try:
        adapter = TwoStepAdapter(task_client, book, bridge, config=task_config)
        lm = ClientLM(reflect_client, bridge, reflect_config)
        if hasattr(sampling_strategy, "bind"):
            sampling_strategy.bind(bridge)
        stall = StallStopper(adapter)
        adapter.stall, adapter.logger = stall, gepa_kw.pop("logger", _Quiet())
        result = gepa.optimize(
            seed_candidate=dict(seed), trainset=train, valset=val, adapter=adapter, reflection_lm=lm,
            reflection_minibatch_size=minibatch, max_metric_calls=max_metric_calls, seed=seed_value, run_dir=run_dir,
            sampling_strategy=sampling_strategy, raise_on_exception=True, stop_callbacks=[stall],
            logger=adapter.logger, module_selector="round_robin", **gepa_kw)
        return result, adapter, lm
    finally:
        bridge.close()
