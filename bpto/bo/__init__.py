"""Bayesian optimisation over prompt embeddings.

`BOSelector` is *only a selector*: it fits a surrogate on `features(node) -> value(node)` over
nodes that have a value, scores candidates by acquisition, and returns the top k. The tree and
ops never know BO exists. Because embeddings can't be inverted back to prompts, BO here is
generate-then-filter: over-propose cheaply, evaluate only what the acquisition picks.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

from ..select import Selector, unexpanded
from ..tree import Node, Tree
from ..value import Value, best_grandchild
from .acquisition import EI, UCB, Mean, Thompson
from .embedders import AzureOpenAIEmbedder, HashEmbedder, HTTPEmbedder, OpenAIEmbedder, VoyageEmbedder
from .gpr import GPR

__all__ = ["BOSelector", "Embedder", "Surrogate", "Acquisition", "EI", "UCB", "Thompson", "Mean", "GPR",
           "AzureOpenAIEmbedder", "HashEmbedder", "HTTPEmbedder", "OpenAIEmbedder", "VoyageEmbedder", "config_features"]


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
                 noise: Callable[[Node, Tree], float | None] | None = None):
        self.embedder = embedder
        self.surrogate = surrogate or GPR()
        self.acquisition = acquisition or EI()
        self.value, self.features, self.noise = value, features, noise
        self.last_fit: dict = {}

    async def _ensure_embeddings(self, nodes: list[Node]) -> None:
        todo = [n for n in nodes if n.embedding is None]
        if todo:
            vecs = await self.embedder.embed([n.prompt.template for n in todo])
            for n, v in zip(todo, vecs):
                n.embedding = list(map(float, v))

    def _x(self, node: Node) -> list[float]:
        return node.embedding + (self.features(node) if self.features else [])

    async def rank(self, tree: Tree, candidates: list[Node]) -> list[tuple[float, Node]]:
        train = [(n, self.value(n, tree)) for n in tree]
        train = [(n, y) for n, y in train if y is not None]
        await self._ensure_embeddings([n for n, _ in train] + candidates)
        if not train or not candidates:
            return [(0.0, n) for n in candidates]
        X, y = [self._x(n) for n, _ in train], [y for _, y in train]
        if self.noise is not None:
            self.surrogate.fit(X, y, [self.noise(n, tree) or 0.0 for n, _ in train])
        else:
            self.surrogate.fit(X, y)
        mean, var = self.surrogate.predict([self._x(n) for n in candidates])
        best_y = max(y for _, y in train)
        acq = self.acquisition(mean, var, best_y)
        self.last_fit = {"n_train": len(train), "best_y": best_y,
                         "pred": {n.id: (m, v, a) for n, m, v, a in zip(candidates, mean, var, acq)}}
        return sorted(zip(acq, candidates), key=lambda t: t[0], reverse=True)

    def top(self, k: int, among: Selector = unexpanded):
        """An async selector: `tree.apply` awaits it. Fits, ranks `among(tree)`, returns the top k."""
        async def _sel(tree: Tree) -> list[Node]:
            ranked = await self.rank(tree, among(tree))
            return [n for _, n in ranked[:k]]
        return _sel
