"""Graph programs: evaluate runs entry -> edges -> terminal through the client; trace, executor metrics, step cap."""
import asyncio
import json

import pytest
from pydantic import BaseModel

from bpto import (Dataset, Edge, LinearObjective, MockClient, ModelConfig, Program, Prompt, Task, Tree, evaluate,
                  select, trace_of, trace_visits)
from bpto.executor import pick_edge
from bpto.gepa import ReflectiveExpander, gepa
from bpto.ops import Variants
from bpto.search import Stop, run

# `prev` is what a loop-back edge writes; the dataset supplies its first-visit value (a loop-back placeholder must
# be renderable on the first visit, so it is a dataset column with a default)
DATA = Dataset.from_records([{"id": f"e{i}", "inputs": {"text": f"doc{i}", "prev": ""}, "answer": f"a{i}"} for i in range(3)])


class Extracted(BaseModel):
    facts: str


class Route(BaseModel):
    summary: str
    next: str


def graph(max_steps=8, max_visits=None):
    return Program(
        modules={"extract": Prompt(template="extract from {text}{prev}"),
                 "shorten": Prompt(template="shorten {facts} (was {text})"),
                 "orchestrate": Prompt(template="judge {summary}")},
        entry="extract",
        edges={"extract": [Edge(to="shorten", mapping={"facts": "facts"})],
               "shorten": [Edge(to="orchestrate", mapping={"summary": "$text"})],
               "orchestrate": [Edge(name="again", to="extract", mapping={"prev": "summary"}), Edge(name="done", to="final", default=True)]},
        max_steps=max_steps, max_visits=max_visits,
    ).model_copy(update={"modules": {**{
        "extract": Prompt(template="extract from {text}{prev}"), "shorten": Prompt(template="shorten {facts} (was {text})"),
        "orchestrate": Prompt(template="judge {summary}")}, "final": Prompt(template="final {summary}")}})


def make_task(handler, **kw):
    async def scorer(prompt, example, completion, ctx):
        return {"m": 1.0 if completion.text.startswith("final") else 0.0}
    return Task(root=graph(**kw), dataset=DATA, scorer=scorer, objective=LinearObjective(m=1.0),
                client=MockClient(handler), schema={"extract": Extracted, "orchestrate": Route},
                config={"shorten": ModelConfig(model="cheap")},
                description={m: f"does {m}" for m in ("extract", "shorten", "orchestrate", "final")})


def test_graph_errors_and_terminals():
    g = graph()
    assert g.is_graph and g.graph_errors() == [] and g.terminals == ["final"] and g.is_orchestrating("orchestrate")
    bad = g.model_copy(update={"edges": {**g.edges, "final": [Edge(to="extract")]}})
    assert any("no terminal" in e for e in bad.graph_errors())
    bad = g.model_copy(update={"edges": {**g.edges, "orchestrate": [Edge(to="extract"), Edge(to="final")]}})
    assert any("distinct non-empty names" in e for e in bad.graph_errors())
    bad = g.model_copy(update={"edges": {**g.edges, "shorten": [Edge(to="nowhere")]}})
    assert any("unknown module 'nowhere'" in e for e in bad.graph_errors())
    assert Program(modules={"a": Prompt(template="x")}, entry="a").is_graph is False
    # with_module keeps the graph
    assert g.with_module("extract", "new {text}").edges == g.edges


def test_pick_edge():
    a, b = Edge(name="again", to="x"), Edge(name="done", to="y", default=True)
    assert pick_edge([a], {}) == (a, False)
    assert pick_edge([a, b], {"next": "again"}) == (a, False)
    assert pick_edge([a, b], {"next": "???"}) == (b, True)
    assert pick_edge([a, b], {}) == (b, True)


def test_linear_then_loop_once():
    seen = []

    def handler(prompt, cfg, schema):
        seen.append((prompt, cfg.model, schema))
        if schema is Extracted:
            return Extracted(facts="F" if prompt.endswith("doc0") or prompt[-1].isdigit() else "F2")
        if schema is Route:
            # loop back once: the first pass carries no prev, the second does (identical prompts would be cache hits)
            return Route(summary="s-" + prompt[-6:], next="done" if "F2" in prompt else "again")
        return "short:" + prompt if prompt.startswith("shorten") else "final:" + prompt
    tree = Tree(make_task(handler))
    asyncio.run(tree.apply(evaluate(), select.root))
    ev = tree.root.evaluation
    assert ev.metrics["m"] == 1.0 and ev.metrics["steps"] == 7.0 and ev.metrics["capped"] == 0.0
    assert ev.metrics["parse_fail.orchestrate"] == 0.0 and "parse_fail.extract" not in ev.metrics
    assert ev.metrics["tokens_per_module.shorten"] > 0 and ev.metrics["tokens_per_module.final"] > 0
    r = ev.per_example[0]
    assert r.output.startswith("final:final short:shorten F2")
    assert [v["step_idx"] for v in trace_visits(r, "extract")] == [0, 3]
    assert trace_of(r, "orchestrate")["parsed"]["next"] == "done" and trace_of(r, "orchestrate", 0)["parsed"]["next"] == "again"
    assert trace_of(r, "shorten", 0)["input"] == "shorten F (was doc0)"  # mapped field + the example's own input
    assert trace_of(r, "extract", 1)["input"].startswith("extract from doc0s-")  # loop-back edge wrote prev
    assert trace_of(r, "final")["input"].startswith("final short:shorten F2")  # $text mapping
    # per-module config and schema reached the client
    assert all(m == "cheap" for p, m, _ in seen if p.startswith("shorten"))
    assert all(m == "claude-opus-5" for p, m, _ in seen if not p.startswith("shorten"))


def test_parse_fail_takes_default_edge_and_step_cap_stops_cycles():
    def handler(prompt, cfg, schema):
        if schema is Extracted:
            return Extracted(facts="F")
        if schema is Route:
            return Route(summary="s", next="nonsense")  # never a valid edge -> default (done)
        return prompt
    tree = Tree(make_task(handler))
    asyncio.run(tree.apply(evaluate(), select.root))
    m = tree.root.evaluation.metrics
    assert m["parse_fail.orchestrate"] == 1.0 and m["steps"] == 4.0 and m["m"] == 1.0

    def loop_forever(prompt, cfg, schema):
        if schema is Extracted:
            return Extracted(facts="F")
        if schema is Route:
            return Route(summary="s", next="again")
        return prompt
    tree = Tree(make_task(loop_forever, max_steps=7))
    asyncio.run(tree.apply(evaluate(), select.root))
    m = tree.root.evaluation.metrics
    assert m["capped"] == 1.0 and m["steps"] == 7.0 and m["m"] == 0.0  # terminal output is the last call's
    r = tree.root.evaluation.per_example[0]
    assert r.error is None and len(trace_visits(r, "extract")) == 3 and "final" not in (r.trace or {})
    tree = Tree(make_task(loop_forever, max_visits=1))
    asyncio.run(tree.apply(evaluate(), select.root))
    assert tree.root.evaluation.metrics["steps"] == 3.0 and tree.root.evaluation.metrics["capped"] == 1.0


def test_adapter_runs_before_render_and_client_cache_dedupes():
    def handler(prompt, cfg, schema):
        if schema is Extracted:
            return Extracted(facts="F")
        if schema is Route:
            return Route(summary="s", next="done")
        return prompt
    task = make_task(handler)
    task.adapters["shorten"] = lambda ex, state: {"facts": state["facts"].lower() + "!"}
    tree = Tree(task)
    asyncio.run(tree.apply(evaluate(), select.root))
    assert trace_of(tree.root.evaluation.per_example[0], "shorten")["input"] == "shorten f! (was doc0)"
    calls = len(task.client.calls)
    asyncio.run(tree.apply(evaluate(), select.root))  # identical rollouts: every module call is a cache hit
    assert len(task.client.calls) == calls and task.client.usage.cache_hits == calls


def test_invalid_graph_is_a_per_example_error_not_a_crash():
    task = make_task(lambda p, c, s: p)
    task.root = task.root.model_copy(update={"edges": {**task.root.edges, "final": [Edge(to="extract")]}})
    tree = Tree(task)
    asyncio.run(tree.apply(evaluate(), select.root))
    assert all("invalid program graph" in r.error for r in tree.root.evaluation.per_example)


def test_round_trip_and_reflection_on_a_looped_module():
    def handler(prompt, cfg, schema):
        if schema is Variants:
            base = prompt.split("<prompt>\n", 1)[1].split("\n</prompt>", 1)[0]
            handler.meta = prompt
            return Variants(prompts=[f"better {base}"])
        if schema is Extracted:
            return Extracted(facts="F")
        if schema is Route:
            handler.n = getattr(handler, "n", 0) + 1
            return Route(summary=f"s{handler.n}", next="again" if handler.n % 2 else "done")
        return prompt
    tree = Tree(make_task(handler))
    asyncio.run(tree.apply(evaluate(), select.root))
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        tree.save(pathlib.Path(td) / "t.json")
        loaded = Tree.load(pathlib.Path(td) / "t.json", tree.task)
    assert loaded.root.prompt == tree.root.prompt and loaded.root.prompt.edges == tree.root.prompt.edges
    assert isinstance(loaded.root.evaluation.per_example[0].trace["orchestrate"], list)
    kids = asyncio.run(tree.apply(ReflectiveExpander(lambda ex, r: "fb", minibatch=2, n=1, module="orchestrate", visits=2), select.root))
    assert kids[0].prompt.modules["orchestrate"].template == "better judge {summary}" and kids[0].prompt.edges == tree.root.prompt.edges
    assert "[visit 1/2]" in handler.meta and "[visit 2/2]" in handler.meta and "judge s" in handler.meta


def test_gepa_round_robin_over_modules():
    def handler(prompt, cfg, schema):
        if schema is Variants:
            base = prompt.split("<prompt>\n", 1)[1].split("\n</prompt>", 1)[0]
            return Variants(prompts=[f"v {base}"])
        if schema is Extracted:
            return Extracted(facts="F")
        if schema is Route:
            return Route(summary="s", next="done")
        return prompt
    tree = Tree(make_task(handler))
    res = asyncio.run(run(tree, gepa(minibatch=2, modules=["extract", "shorten"]), stop=Stop(rounds=5)))
    mods = [n.origin.params.get("module") for n in tree if n.origin.op == "reflect"]
    assert mods == ["shorten", "extract", "shorten", "extract"]  # round 0 evaluates the root; round r rewrites modules[r % 2]
