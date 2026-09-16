"""Program-valued nodes: one node = several module prompts; module-wise mutation; additive surrogate."""
import asyncio
import json

import numpy as np
import pytest

from bpto import Dataset, LinearObjective, MockClient, Program, Prompt, Task, Tree, evaluate, select
from bpto.bo import EI, AdditiveGPR, BOSelector, GPR, HashEmbedder
from bpto.gepa import ReflectiveExpander
from bpto.ops import Variants
from bpto.tree import Node
from bpto.value import own_score

DATA = Dataset.from_records([{"id": f"e{i}", "inputs": {"q": f"q{i}"}, "answer": f"a{i}"} for i in range(4)])


def two_stage_task(handler=None):
    """Entry module "a" is what evaluate sends; the scorer runs module "b" and records its trace."""
    calls = []

    def _handler(prompt, cfg, schema):
        calls.append(prompt)
        return "out:" + prompt

    async def scorer(prompt, example, completion, ctx):
        assert isinstance(prompt, Program)
        rendered = prompt.modules["b"].render(q=example.inputs["q"], prev=completion.text)
        comp = await ctx.client.complete(rendered)
        ctx.trace["b"] = {"input": rendered, "output": comp.text}
        return {"m": 1.0 if "good" in prompt.modules["b"].template else 0.5}
    task = Task(root={"a": "first {q}", "b": "second {q} after {prev}"}, dataset=DATA, scorer=scorer,
                objective=LinearObjective(m=1.0), client=MockClient(handler or _handler),
                description={"a": "does the first thing", "b": "does the second thing"})
    return task, calls


def test_program_model_delegates_to_entry_and_round_trips():
    p = Program(modules={"a": Prompt(template="x {q}"), "b": Prompt(template="y {z}")}, entry="a")
    assert p.template == "x {q}" and p.placeholders == ("q",) and p.render(q=1) == "x 1"
    p2 = p.with_module("b", "yy {z}")
    assert p2.modules["a"] == p.modules["a"] and p2.modules["b"].template == "yy {z}" and p2.hash != p.hash
    with pytest.raises(KeyError):
        p.with_module("c", "nope")
    n = Node(prompt=p, embedding={"a": [1.0], "b": [2.0]})
    back = Node.model_validate(json.loads(n.model_dump_json()))
    assert isinstance(back.prompt, Program) and back.prompt == p and back.embedding == {"a": [1.0], "b": [2.0]}
    plain = Node.model_validate(json.loads(Node(prompt=Prompt(template="t")).model_dump_json()))
    assert isinstance(plain.prompt, Prompt)


def test_evaluate_runs_program_and_records_trace(tmp_path):
    task, calls = two_stage_task()
    tree = Tree(task)
    assert isinstance(tree.root.prompt, Program)
    asyncio.run(tree.apply(evaluate(), select.root))
    assert tree.root.evaluation.metrics["m"] == 0.5
    assert len(calls) == 2 * len(DATA) and calls[0] == "first q0" and calls[1] == "second q0 after out:first q0"
    r = tree.root.evaluation.per_example[0]
    assert r.trace["b"]["output"] == "out:second q0 after out:first q0"
    tree.save(tmp_path / "t.json")
    loaded = Tree.load(tmp_path / "t.json", task)
    assert loaded.root.prompt == tree.root.prompt and loaded.root.evaluation.per_example[0].trace == r.trace


def test_reflective_expander_rewrites_one_module_only():
    seen = {}

    def handler(prompt, cfg, schema):
        if schema is Variants:
            seen["meta"] = prompt
            base = prompt.split("<prompt>\n", 1)[1].split("\n</prompt>", 1)[0]
            # one valid rewrite, one that drops a placeholder (must be discarded), one that touches the other module's name
            return Variants(prompts=[f"good {base}", "no placeholders at all", f"{base} again"])
        return "out:" + prompt
    task, _ = two_stage_task(handler)
    tree = Tree(task)
    asyncio.run(tree.apply(evaluate(), select.root))
    fb = {"a": lambda ex, r: "fb-a", "b": lambda ex, r: "fb-b:" + r.trace["b"]["output"]}
    kids = asyncio.run(tree.apply(ReflectiveExpander(fb, minibatch=2, n=3, module="b", passed={"a": None, "b": None}), select.root))
    assert len(kids) == 2 and all(isinstance(k.prompt, Program) for k in kids)
    assert all(k.prompt.modules["a"] == tree.root.prompt.modules["a"] for k in kids)
    assert kids[0].prompt.modules["b"].template == "good second {q} after {prev}"
    assert kids[0].origin.params["module"] == "b"
    meta = seen["meta"]
    assert "does the second thing" in meta and "{q}, {prev}" in meta and "second {q} after {prev}" in meta
    assert "fb-b:out:second" in meta and "Inputs:\nsecond q" in meta  # the module's own trace, not the entry output
    # a plain expander with no module still rewrites the entry module
    kids_a = asyncio.run(tree.apply(ReflectiveExpander(fb["a"], minibatch=2, n=1), select.root))
    assert kids_a[0].prompt.modules["a"].template.startswith("good first") and kids_a[0].prompt.modules["b"] == tree.root.prompt.modules["b"]
    assert "module" not in kids_a[0].origin.params


def test_additive_gpr_equals_gpr_with_one_block_and_recovers_components():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (50, 2))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 1]) + rng.normal(0, 0.05, 50)
    g, a1 = GPR().fit(X.tolist(), y.tolist()), AdditiveGPR().fit(X.tolist(), y.tolist())
    for p, q in zip(g.predict(X[:5].tolist()), a1.predict(X[:5].tolist())):
        assert np.allclose(p, q)
    a2 = AdditiveGPR(blocks=[(0, 1), (1, 2)]).fit(X.tolist(), y.tolist())
    grid = np.linspace(-3, 3, 40)
    mu0, _ = a2.predict(np.column_stack([grid, np.zeros(40)]).tolist(), block=0)
    mu1, _ = a2.predict(np.column_stack([np.zeros(40), grid]).tolist(), block=1)
    assert np.corrcoef(mu0, np.sin(grid))[0, 1] > 0.98 and np.corrcoef(mu1, np.cos(2 * grid))[0, 1] > 0.95
    # additivity: an unseen combination is predicted from the marginals (train on the "L", predict the corner)
    Xl = np.array([[x, -2.5] for x in np.linspace(-3, 3, 15)] + [[-2.5, x] for x in np.linspace(-3, 3, 15)])
    yl = np.sin(Xl[:, 0]) + 0.5 * np.cos(2 * Xl[:, 1])
    corner = [[1.5, 1.5]]
    add = AdditiveGPR(blocks=[(0, 1), (1, 2)]).fit(Xl.tolist(), yl.tolist()).predict(corner)[0][0]
    full = GPR().fit(Xl.tolist(), yl.tolist()).predict(corner)[0][0]
    truth = np.sin(1.5) + 0.5 * np.cos(3.0)
    assert abs(add - truth) < abs(full - truth)


@pytest.mark.asyncio
async def test_bo_selector_embeds_modules_and_ranks_per_module():
    task, _ = two_stage_task()
    tree = Tree(task)
    await tree.apply(evaluate(), select.root)
    from bpto.tree import Evaluation, Origin
    # children differing in module b with a planted value: "good" in b -> 1.0
    def child(a, b, m):
        n = tree.add_child(tree.root, Program(modules={"a": Prompt(template=a), "b": Prompt(template=b)}, entry="a"), Origin(op="test"))
        n.evaluation = Evaluation(per_example=[], metrics={"m": m}, metrics_std={"m": 0.1}, n=4, score=m, feasible=True, dataset_ids=[])
        return n
    for i in range(6):
        child(f"first {{q}} v{i}", f"second {{q}} after {{prev}} good{i}", 0.9 + 0.01 * i)
        child(f"first {{q}} v{i}", f"second {{q}} after {{prev}} bad{i}", 0.4 + 0.01 * i)
    sel = BOSelector(HashEmbedder(dim=64), AdditiveGPR(), EI(), value=own_score,
                     noise=lambda n, t: n.evaluation.metrics_std["m"] / n.evaluation.n ** 0.5)
    cands = [child("first {q} new", "second {q} after {prev} good-new", 0.0), child("first {q} new", "second {q} after {prev} bad-new", 0.0)]
    for c in cands:
        c.evaluation = None
    ranked = await sel.rank(tree, cands, module="b")
    assert isinstance(tree.root.embedding, dict) and set(tree.root.embedding) == {"a", "b"}
    assert sel.surrogate.blocks == [(0, 64), (64, 128)] and sel.last_fit["module"] == "b"
    assert ranked[0][1] is cands[0]
    best = await sel.argmax_modules(tree)
    assert "good" in best["b"].prompt.modules["b"].template


@pytest.mark.asyncio
async def test_incumbent_is_posterior_mean_not_a_lucky_observation():
    """One noisy 1.0 among many ~0.5 observations must not zero the EI of everything else."""
    from bpto.tree import Evaluation, Origin
    task, _ = two_stage_task()
    tree = Tree(task)
    await tree.apply(evaluate(), select.root)
    def child(i, m, n):
        c = tree.add_child(tree.root, Program(modules={"a": Prompt(template=f"first {{q}} v{i}"), "b": Prompt(template="second {q} after {prev}")}, entry="a"), Origin(op="t"))
        c.evaluation = Evaluation(per_example=[], metrics={"m": m}, metrics_std={"m": 0.4}, n=n, score=m, feasible=True, dataset_ids=[])
        return c
    for i in range(12):
        child(i, 0.5 + 0.01 * (i % 3), 40)
    child(99, 1.0, 5)  # a 5-row minibatch fluke
    sel = BOSelector(HashEmbedder(dim=32), AdditiveGPR(), EI(), value=own_score, noise=lambda n, t: 0.4 / n.evaluation.n ** 0.5)
    cands = [child(200 + i, 0.0, 40) for i in range(3)]
    for c in cands:
        c.evaluation = None
    ranked = await sel.rank(tree, cands)
    assert sel.last_fit["best_obs"] == 1.0 and sel.last_fit["best_y"] < 0.9
    assert max(a for a, _ in ranked) > 1e-4


def test_pit_targets_are_finite_standard_normal_and_tie_safe():
    from bpto.bo import PIT
    y = [0.7, 0.72, 0.72, 1.0, 0.5, 0.69, 1.0, 1.0]
    t = PIT()
    z = t.fit_transform(y)
    assert all(np.isfinite(z)) and z[1] == z[2] and z[3] == z[6] == z[7] == max(z) and z[4] == min(z)
    assert abs(np.mean(z)) < 0.2 and 0.6 < np.std(z) < 1.2
    assert all(s > 0 for s in t.slope(y))


@pytest.mark.asyncio
async def test_bo_selector_pit_ranks_like_raw_on_a_planted_peak():
    task, _ = two_stage_task()
    tree = Tree(task)
    await tree.apply(evaluate(), select.root)
    from bpto.tree import Evaluation, Origin
    def child(i, m):
        c = tree.add_child(tree.root, Program(modules={"a": Prompt(template=f"first {{q}} {'good' if m > 0.8 else 'bad'} v{i}"), "b": Prompt(template="second {q} after {prev}")}, entry="a"), Origin(op="t"))
        c.evaluation = Evaluation(per_example=[], metrics={"m": m}, metrics_std={"m": 0.1}, n=40, score=m, feasible=True, dataset_ids=[])
        return c
    for i in range(8):
        child(i, 0.9 + 0.01 * i); child(i + 10, 0.5 + 0.01 * i)
    cands = [child(100, 0.0), child(101, 0.0)]
    cands[0].prompt = cands[0].prompt.with_module("a", "first {q} good new"); cands[1].prompt = cands[1].prompt.with_module("a", "first {q} bad new")
    for c in cands:
        c.evaluation = None
    sel = BOSelector(HashEmbedder(dim=64), AdditiveGPR(), EI(), value=own_score, transform="pit", noise=lambda n, t: 0.1 / n.evaluation.n ** 0.5)
    ranked = await sel.rank(tree, cands)
    assert ranked[0][1] is cands[0] and sel.last_fit["transform"] == "pit" and abs(sel.last_fit["best_obs"]) < 3
