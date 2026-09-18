import json
import math
import random

import httpx
import numpy as np

from bpto import Dataset, LinearObjective, MockClient, Prompt, Task, Tree
from bpto.bo import EI, GPR, UCB, BOSelector, HashEmbedder, HTTPEmbedder, Thompson, config_features
from bpto.tree import Evaluation, NodeState, Origin
from bpto.value import child_scores, own_score


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


class PaddedEmbedder:
    """The 1-d signal buried in 8 noisy dimensions: PCA must recover contrast for the single lengthscale."""

    async def embed(self, texts):
        out = []
        for t in texts:
            x = float(t.split()[0])
            r = random.Random(hash(t) % 10_000)
            out.append([x] + [0.01 * r.random() for _ in range(8)])
        return out


async def test_bo_selector_list_targets_and_pca():
    tree = Tree(_task())
    parents = [tree.add_child(tree.root, f"{x:.4f} {{x}}", Origin(op="random")) for x in (0.1, 0.3, 0.5, 0.7, 0.9)]
    for p in parents:
        _fake_eval(p)
        for d in (-0.02, 0.0, 0.02):  # three children per parent: repeated observations at one input
            c = tree.add_child(p, f"{float(p.prompt.template.split()[0]) + d:.4f} {{x}}", Origin(op="random"))
            _fake_eval(c)

    kids = child_scores  # DescendantValue(1, list): every child score is an observation at the parent's input
    bo = BOSelector(PaddedEmbedder(), GPR(), EI(xi=0.0), value=kids, pca=2)
    ranked = await bo.rank(tree, parents)
    assert bo.last_fit["n_train"] == 20 and bo.last_fit["n_inputs"] == 6  # root has 5 evaluated children too
    assert bo.last_fit["pca"]["dims"] == 2 and bo.last_fit["pca"]["explained"] > 0.9
    # the parent whose children peak (f peaks near x=0.5) ranks first, or the flat-acquisition warning would have fired
    assert ranked[0][1] is max(parents, key=lambda p: sum(kids(p, tree)))


# ---- batch acquisitions (q > 1) -------------------------------------------------------------------


def _clustered_posterior():
    """Three candidates: 0 and 1 are siblings (corr 0.99, same mean/var), 2 is independent with a slightly lower
    mean. Any joint criterion pairs a sibling with 2; independent top-2 by EI takes both siblings."""
    mean = [1.0, 1.0, 0.8]
    cov = np.array([[1.0, 0.99, 0.0], [0.99, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return mean, cov


def test_predict_cov_matches_predict_and_is_psd():
    from bpto.bo import AdditiveGPR
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (25, 2))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1]
    Xt = rng.uniform(-3, 3, (10, 2)).tolist()
    for gp in (GPR().fit(X.tolist(), y.tolist()), AdditiveGPR([(0, 1), (1, 2)]).fit(X.tolist(), y.tolist())):
        mu, var = gp.predict(Xt)
        mu2, cov = gp.predict_cov(Xt)
        assert cov.shape == (10, 10)
        assert np.allclose(mu, mu2) and np.allclose(np.diag(cov), var, rtol=1e-6, atol=1e-9)
        assert np.linalg.eigvalsh(cov).min() > -1e-9
        # near-neighbours are positively correlated, far points are not
        near, far = Xt[0], (np.asarray(Xt[0]) + [0.05, 0.0]).tolist()
        _, c = gp.predict_cov([near, far, [3.0, -3.0]])
        assert c[0, 1] / math.sqrt(c[0, 0] * c[1, 1]) > 0.9
    gp = AdditiveGPR([(0, 1), (1, 2)]).fit(X.tolist(), y.tolist())
    _, cov_b = gp.predict_cov(Xt, block=1)
    assert np.allclose(np.diag(cov_b), gp.predict(Xt, block=1)[1], rtol=1e-6, atol=1e-9)


def test_batch_acquisitions_avoid_siblings():
    from bpto.bo import QEI, KrigingBeliever
    mean, cov = _clustered_posterior()
    top2 = sorted(range(3), key=lambda i: -EI(xi=0.0)(mean, list(np.diag(cov)), 0.5)[i])[:2]
    assert sorted(top2) == [0, 1]                                   # the failure mode
    for batch in (KrigingBeliever(), QEI(n_samples=2000, seed=0)):
        picks = batch(mean, cov, 0.5, 2)
        assert len(picks) == 2 and len(set(picks)) == 2
        assert 2 in picks, (type(batch).__name__, picks)
    q = QEI(n_samples=2000, seed=0)
    assert sorted(q(mean, cov, 0.5, 5)) == [0, 1, 2] and q.last["qei"] > 0   # q > n: every candidate, once


def test_quantecarlo_batch_uses_client_select():
    from bpto.bo import QuantecarloQEI

    class FakeClient:
        def __init__(self):
            self.calls = []

        def select(self, mu, cov, best_y, q, **kw):
            self.calls.append((np.asarray(mu), np.asarray(cov), best_y, q, kw))
            return {"indices": [2, 0], "qei": 0.4, "regime": "exact", "n_cands": 3, "n_sampled": 3, "n_batches": 3}

    mean, cov = _clustered_posterior()
    fc = FakeClient()
    b = QuantecarloQEI(fc, pi_floor=0.1)
    assert b(mean, cov, 0.5, 2) == [2, 0]
    (mu, c, by, q, kw), = fc.calls
    assert np.allclose(mu, mean) and np.allclose(c, cov) and by == 0.5 and q == 2 and kw == {"pi_floor": 0.1}
    assert b.last["regime"] == "exact" and b.last["qei"] == 0.4


async def test_bo_selector_top_k_uses_batch_and_records_it():
    from bpto.bo import QEI
    tree = Tree(_task())
    xs = [i / 40 for i in range(40)]
    nodes = [tree.add_child(tree.root, f"{x:.4f} {{x}}", Origin(op="random")) for x in xs]
    train, cands = nodes[::4], [n for i, n in enumerate(nodes) if i % 4]
    for n in train:
        _fake_eval(n)
    plain = BOSelector(NumEmbedder(), GPR(), EI(xi=0.0), value=own_score)
    joint = BOSelector(NumEmbedder(), GPR(), EI(xi=0.0), value=own_score, batch=QEI(n_samples=1000, seed=0))
    p1 = await plain.top(k=1, among=lambda t: cands)(tree)
    j1 = await joint.top(k=1, among=lambda t: cands)(tree)
    assert p1 == j1 and "batch" not in joint.last_fit                         # k == 1: byte-identical path
    p4 = await plain.top(k=4, among=lambda t: cands)(tree)
    j4 = await joint.top(k=4, among=lambda t: cands)(tree)
    assert len(j4) == 4 and len({n.id for n in j4}) == 4 and all(n in cands for n in j4)
    rec = joint.last_fit["batch"]
    assert rec["method"] == "QEI" and rec["ids"] == [n.id for n in j4] and rec["top_k_ids"] == [n.id for n in p4]
    assert rec["qei"] >= 0.0
    # the joint batch spreads over x where independent top-4 stacks neighbours around the EI peak
    spread = lambda ns: max(float(n.prompt.template.split()[0]) for n in ns) - min(float(n.prompt.template.split()[0]) for n in ns)
    assert spread(j4) >= spread(p4)
