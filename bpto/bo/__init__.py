"""Bayesian optimisation over prompt embeddings.

`BOSelector` is *only a selector*: it fits a surrogate on `features(node) -> value(node)` over
nodes that have a value, scores candidates by acquisition, and returns the top k. The tree and
ops never know BO exists. Because embeddings can't be inverted back to prompts, BO here is
generate-then-filter: over-propose cheaply, evaluate only what the acquisition picks.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

from ..select import Selector, unexpanded
from ..prompt import Program
from ..tree import Node, Tree
from ..value import Value, best_grandchild
from .acquisition import EI, UCB, Mean, Thompson
from .embedders import AzureOpenAIEmbedder, BedrockEmbedder, HashEmbedder, HTTPEmbedder, OpenAIEmbedder, VoyageEmbedder
import logging

from .gpr import GPR, PIT, AdditiveGPR

log = logging.getLogger("bpto")

__all__ = ["BOSelector", "Embedder", "Surrogate", "Acquisition", "EI", "UCB", "Thompson", "Mean", "GPR", "AdditiveGPR", "PIT",
           "AzureOpenAIEmbedder", "BedrockEmbedder", "HashEmbedder", "HTTPEmbedder", "OpenAIEmbedder", "VoyageEmbedder", "config_features"]


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Surrogate(Protocol):
    def fit(self, X: list[list[float]], y: list[float], noise: list[float] | None = None) -> None: ...
    def predict(self, X: list[list[float]]) -> tuple[list[float], list[float]]:
        """(mean, variance) per row."""
        ...


class Acquisition(Protocol):
    def __call__(self, mean: list[float], var: list[float], best_y: float) -> list[float]: ...


def config_features(node: Node) -> list[float]:
    """Numeric features of a node's ModelConfig, for when the config is part of the search space."""
    c = node.config
    if c is None:
        return [0.0, 0.0]
    return [c.temperature or 0.0, float(c.max_tokens) / 1e4]


class BOSelector:
    def __init__(self, embedder: Embedder, surrogate: Surrogate | None = None, acquisition: Acquisition | None = None,
                 value: Value = best_grandchild, features: Callable[[Node], list[float]] | None = None,
                 noise: Callable[[Node, Tree], float | None] | None = None, transform: str | None = None,
                 pca: int | None = None):
        self.embedder = embedder
        self.surrogate = surrogate or GPR()
        self.acquisition = acquisition or EI()
        self.value, self.features, self.noise = value, features, noise
        self.transform = transform  # "pit": rank-based probability-integral transform of the targets (see gpr.PIT)
        # pca=k: project inputs to their top-k principal directions, fit each call on the pooled rows (training inputs
        # and candidates). In 256-d unit-vector space every prompt is about equally far from every other (nearest
        # neighbour ~40% of the median distance); the projection restores contrast for a single lengthscale.
        self.pca = pca
        self.last_fit: dict = {}
        self._same_top = (None, 0)

    async def _ensure_embeddings(self, nodes: list[Node]) -> None:
        """Plain nodes get one vector; Program nodes one per module (each text embedded once, keyed by module)."""
        todo = [n for n in nodes if n.embedding is None]
        if not todo:
            return
        texts, slots = [], []
        for n in todo:
            if isinstance(n.prompt, Program):
                for name, p in n.prompt.modules.items():
                    texts.append(p.template); slots.append((n, name))
            else:
                texts.append(n.prompt.template); slots.append((n, None))
        vecs = await self.embedder.embed(texts)
        for (n, name), v in zip(slots, vecs):
            if name is None:
                n.embedding = list(map(float, v))
            else:
                if not isinstance(n.embedding, dict):
                    n.embedding = {}
                n.embedding[name] = list(map(float, v))

    def _x(self, node: Node) -> list[float]:
        e = node.embedding
        vec = [x for name in self._modules(node) for x in e[name]] if isinstance(e, dict) else list(e)
        return vec + (self.features(node) if self.features else [])

    @staticmethod
    def _modules(node: Node) -> list[str]:
        return list(node.prompt.modules) if isinstance(node.prompt, Program) else []

    def _blocks(self, node: Node) -> list[tuple[int, int]]:
        """Feature-vector spans of each module (plus one for `features`), for an additive surrogate."""
        out, i = [], 0
        for name in self._modules(node):
            out.append((i, i + len(node.embedding[name]))); i += len(node.embedding[name])
        if not out:
            out.append((0, len(node.embedding)))
            i = len(node.embedding)
        if self.features:
            k = len(self.features(node))
            out.append((i, i + k))
        return out

    async def rank(self, tree: Tree, candidates: list[Node], module: str | None = None) -> list[tuple[float, Node]]:
        """Fit on every node with a value, score `candidates` by acquisition, best first.

        `module=m` (Program nodes, additive surrogate): acquisition on the posterior of module m's component
        alone - "how promising is this candidate's module m as the thing to rewrite next"; the incumbent is the
        best component mean among the training nodes (components are only identified up to a constant).
        """
        # `value` may return one target or a list of them (e.g. every child's score, observed at the parent's input):
        # repeated observations at one x are the noisy-BO way to learn a parent's expected yield and its spread
        train, owners = [], []
        for n in tree:
            y = self.value(n, tree)
            ys = y if isinstance(y, list) else [y]
            ys = [v for v in ys if v is not None]
            if not ys:
                continue
            se = self.noise(n, tree) if self.noise is not None else None
            ses = se if isinstance(se, list) else [se] * len(ys)
            train += [(n, v, s) for v, s in zip(ys, ses)]
            owners.append(n)
        await self._ensure_embeddings(owners + candidates)
        if not train or not candidates:
            return [(0.0, n) for n in candidates]
        if getattr(self.surrogate, "blocks", 0) is None:
            self.surrogate.blocks = self._blocks(train[0][0])
        X, y = [self._x(n) for n, _, _ in train], [y for _, y, _ in train]
        Xc = [self._x(n) for n in candidates]
        if self.pca and not getattr(self.surrogate, "blocks", None):
            X, Xc = self._project(X, Xc)
        y_raw = list(y)
        if self.transform == "pit":
            # targets -> N(0, 1) by rank; per-node noise is in the raw metric's units, so it is rescaled by the local
            # slope of the transform (dz/dy at each point, from the fitted ECDF), floored to avoid a zero slope on ties
            pit = PIT()
            y = pit.fit_transform(y_raw)
            dz = pit.slope(y_raw)
        if self.noise is not None:
            se = [s or 0.0 for _, _, s in train]
            if self.transform == "pit":
                se = [s * d for s, d in zip(se, dz)]
            self.surrogate.fit(X, y, se)
        else:
            self.surrogate.fit(X, y)
        # The EI incumbent is the best POSTERIOR EXPECTATION at the training inputs, max_i mu(x_i) - the model's
        # denoised estimate of each observed point - rather than the best observation max_i y_i. It is a model output,
        # not an average of anything. With a noise term the GP does not interpolate its data, so mu(x_i) != y_i, and a
        # single lucky observation cannot become an incumbent nothing can improve on. This is the usual plug-in for
        # EI under noisy observations (Huang et al. 2006; BoTorch's `best_f` from the posterior).
        if module is None:
            mean, var = self.surrogate.predict(Xc)
            best_y = max(self.surrogate.predict(X)[0])
        else:
            b = self._modules(train[0][0]).index(module)
            mean, var = self.surrogate.predict(Xc, block=b)
            best_y = max(self.surrogate.predict(X, block=b)[0])
        acq = self.acquisition(mean, var, best_y)
        ranked = sorted(zip(acq, candidates), key=lambda t: t[0], reverse=True)
        self.last_fit = {"n_train": len(train), "n_inputs": len(owners), "best_y": float(best_y), "pca": getattr(self, "last_pca", None), "best_obs": float(max(y)), "module": module,
                         "transform": self.transform, "flat": bool(max(acq) - min(acq) <= 0.0 or max(acq) < 1e-12),
                         **({"ell": getattr(self.surrogate, "ell_", None), "noise": getattr(self.surrogate, "noise_", None)}),
                         "pred": {n.id: (m, v, a) for n, m, v, a in zip(candidates, mean, var, acq)}}
        # a stuck search is visible immediately, not after hours: flat acquisition, or the same winner again and again
        if len(candidates) > 1 and self.last_fit["flat"]:
            log.warning("BOSelector: acquisition is flat over %d candidates (best_y=%.3f) - selection is arbitrary", len(candidates), best_y)
        top = ranked[0][1].id
        self._same_top = (top, self._same_top[1] + 1 if self._same_top[0] == top else 1)
        if self._same_top[1] in (10, 25, 50, 100):
            log.warning("BOSelector: the same candidate %s has ranked first %d times in a row", top, self._same_top[1])
        return ranked

    def _project(self, X: list[list[float]], Xc: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
        import numpy as np
        pool = np.asarray(X + Xc, float)
        mu = pool.mean(0)
        _, S, Vt = np.linalg.svd(pool - mu, full_matrices=False)
        k = min(self.pca, len(Vt))
        P = Vt[:k].T
        self.last_pca = {"dims": k, "explained": float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-12))}
        return ((np.asarray(X) - mu) @ P).tolist(), ((np.asarray(Xc) - mu) @ P).tolist()

    async def argmax_modules(self, tree: Tree, among: list[Node] | None = None) -> dict[str, Node]:
        """"Merge for free": per module, the node whose module-m component has the highest posterior mean.
        Recombining those modules is the additive model's predicted best program; fit first via `rank`."""
        nodes = [n for n in (among or list(tree)) if isinstance(n.prompt, Program)]
        await self._ensure_embeddings(nodes)
        out = {}
        for b, name in enumerate(self._modules(nodes[0])):
            mu, _ = self.surrogate.predict([self._x(n) for n in nodes], block=b)
            out[name] = max(zip(mu, nodes), key=lambda t: t[0])[1]
        return out

    def top(self, k: int, among: Selector = unexpanded):
        """An async selector: `tree.apply` awaits it. Fits, ranks `among(tree)`, returns the top k."""
        async def _sel(tree: Tree) -> list[Node]:
            ranked = await self.rank(tree, among(tree))
            return [n for _, n in ranked[:k]]
        return _sel
