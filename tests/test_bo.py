import json
import math
import random

import httpx
import numpy as np

from bpto import Dataset, LinearObjective, MockClient, Prompt, Task, Tree
from bpto.bo import EI, GPR, UCB, BOSelector, HashEmbedder, HTTPEmbedder, Thompson, config_features
from bpto.tree import Evaluation, NodeState, Origin
from bpto.value import own_score


def test_gpr_recovers_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (40, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.05, 40)
    gp = GPR().fit(X.tolist(), y.tolist())
    Xt = np.linspace(-3, 3, 50)[:, None]
    mu, var = gp.predict(Xt.tolist())
    assert np.max(np.abs(np.array(mu) - np.sin(Xt[:, 0]))) < 0.15
    far_mu, far_var = gp.predict([[30.0]])
    assert far_var[0] > max(var)  # uncertainty grows away from data


def test_gpr_per_point_noise_is_respected():
    X = [[0.0], [1.0], [2.0]]
    y = [0.0, 5.0, 0.0]
    tight = GPR(lengthscale=1.0, noise=1e-4).fit(X, y).predict([[1.0]])[0][0]
    loose = GPR(lengthscale=1.0, noise=1e-4).fit(X, y, noise=[0.0, 3.0, 0.0]).predict([[1.0]])[0][0]
    assert abs(tight - 5.0) < 0.1 and loose < tight


def test_acquisitions():
    mean, var = [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]
    ei = EI(xi=0.0)(mean, var, 1.0)
    assert ei[0] < ei[1] and ei[2] < 1e-5
    ucb = UCB(kappa=2)(mean, var, 1.0)
    assert ucb == [2.0, 3.0, 1.0]
    t = Thompson(seed=1)(mean, var, 1.0)
    assert len(t) == 3 and t[2] == 1.0


def test_hash_embedder_is_deterministic_and_similar_for_similar_text():
    e = HashEmbedder()
    import asyncio
    a, b, c = asyncio.run(e.embed(["extract the names", "extract the names", "compute the sum of numbers"]))
    assert a == b
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert dot(a, b) > dot(a, c)


async def test_http_embedder_batches_and_caches():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body["input"])
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [float(len(t))]} for i, t in enumerate(body["input"])]})

    e = HTTPEmbedder("m", "http://x/v1", "k", batch_size=2, transport=httpx.MockTransport(handler))
    out = await e.embed(["a", "bb", "ccc", "a"])
    assert out == [[1.0], [2.0], [3.0], [1.0]] and calls == [["a", "bb"], ["ccc"]]
    await e.embed(["bb"])
    assert len(calls) == 2


# ---- end-to-end: BO beats random at picking which leaf to evaluate ------------------------

class NumEmbedder:
    """Templates are numbers; embedding is the number. Makes the objective a known f(x)."""
    async def embed(self, texts):
        return [[float(t.split()[0])] for t in texts]


def f(x):
    return math.exp(-((x - 0.7) ** 2) / 0.02) + 0.3 * math.sin(6 * x)  # peak near 0.7


def _fake_eval(node):
    x = float(node.prompt.template.split()[0])
    node.evaluation = Evaluation(per_example=[], metrics={"acc": f(x)}, metrics_std={}, n=1,
                                 score=f(x), feasible=True, dataset_ids=[])
    node.state = NodeState.EVALUATED


def _task():
    return Task(root="0.0 {x}", dataset=Dataset.from_records([{"inputs": {"x": 1}, "answer": 1}]),
                scorer=lambda *a: {"acc": 0.0}, objective=LinearObjective(acc=1.0), client=MockClient())


async def test_bo_selector_finds_the_peak():
    rng = random.Random(0)
    tree = Tree(_task())
    xs = [rng.random() for _ in range(60)]
    nodes = [tree.add_child(tree.root, f"{x:.4f} {{x}}", Origin(op="random")) for x in xs]
    # evaluate a sparse random subset; BO must pick among the rest
    train, cands = nodes[:12], nodes[12:]
    for n in train:
        _fake_eval(n)
    bo = BOSelector(NumEmbedder(), GPR(), EI(xi=0.0), value=own_score)
    picks = await bo.top(k=5, among=lambda t: cands)(tree)
    true_best = max(cands, key=lambda n: f(float(n.prompt.template.split()[0])))
    random_hit_rate = 5 / len(cands)
    # over a few sequential BO rounds the peak candidate gets found
    found = true_best in picks
    for _ in range(4):
        if found:
            break
        for p in picks:
            _fake_eval(p)
        cands = [c for c in cands if not c.evaluated]
        picks = await bo.top(k=5, among=lambda t: cands)(tree)
        found = true_best in picks
    assert found, f"BO failed to find peak; chance per round was {random_hit_rate:.2f}"
    assert bo.last_fit["n_train"] >= 12


async def test_config_features_are_appended():
    from bpto import ModelConfig
    tree = Tree(_task())
    a = tree.add_child(tree.root, "0.2 {x}", Origin(op="random"), config=ModelConfig(temperature=0.5, max_tokens=1000))
    assert config_features(a) == [0.5, 0.1]
    bo = BOSelector(NumEmbedder(), value=own_score, features=config_features)
    await bo._ensure_embeddings([a])
    assert bo._x(a) == [0.2, 0.5, 0.1]
