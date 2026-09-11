"""Gaussian process regression in numpy. RBF kernel, hyperparameters by log-marginal-likelihood
grid search (no scipy). Targets are standardised internally; per-point noise is supported so
evaluation uncertainty (metrics_std / sqrt(n)) can be passed straight in."""
from __future__ import annotations

import math

import numpy as np


class GPR:
    def __init__(self, lengthscale: float | None = None, noise: float | None = None,
                 grid: int = 12, jitter: float = 1e-8):
        self.lengthscale, self.noise, self.grid, self.jitter = lengthscale, noise, grid, jitter
        self._X = self._alpha = self._L = None
        self._y_mu = 0.0
        self._y_sd = 1.0

    # ---- kernel ----------------------------------------------------------------------
    def _k(self, A: np.ndarray, B: np.ndarray, ell: float) -> np.ndarray:
        d2 = np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2 * A @ B.T
        return np.exp(-0.5 * np.maximum(d2, 0) / (ell * ell))

    def _lml(self, X, y, ell, noise, pn) -> float:
        K = self._k(X, X, ell) + np.diag(noise + pn + self.jitter)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -math.inf
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return float(-0.5 * y @ a - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * math.log(2 * math.pi))

    # ---- fit / predict ---------------------------------------------------------------
    def fit(self, X, y, noise=None) -> "GPR":
        X, y = np.asarray(X, float), np.asarray(y, float)
        self._y_mu, self._y_sd = float(y.mean()), float(y.std() or 1.0)
        ys = (y - self._y_mu) / self._y_sd
        pn = np.zeros(len(y)) if noise is None else (np.asarray(noise, float) / self._y_sd) ** 2

        if len(X) > 1:
            d = np.sqrt(np.maximum(np.sum((X[:, None, :] - X[None, :, :]) ** 2, -1), 0))
            med = float(np.median(d[np.triu_indices(len(X), 1)])) or 1.0
        else:
            med = 1.0
        ells = [self.lengthscale] if self.lengthscale else list(med * np.logspace(-1, 1, self.grid))
        noises = [self.noise] if self.noise is not None else list(np.logspace(-4, 0, self.grid))
        best = (-math.inf, ells[0], noises[0])
        for ell in ells:
            for nz in noises:
                v = self._lml(X, ys, ell, nz, pn)
                if v > best[0]:
                    best = (v, ell, nz)
        _, self.ell_, self.noise_ = best
        K = self._k(X, X, self.ell_) + np.diag(self.noise_ + pn + self.jitter)
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, ys))
        self._X = X
        return self

    def predict(self, X) -> tuple[list[float], list[float]]:
        X = np.asarray(X, float)
        Ks = self._k(X, self._X, self.ell_)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.maximum(1.0 - np.sum(v * v, 0), 1e-12)
        return list(mu * self._y_sd + self._y_mu), list(var * self._y_sd ** 2)
