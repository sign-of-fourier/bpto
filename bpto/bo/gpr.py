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

    def _lml(self, X, y, ell, noise, pn, K0=None) -> float:
        K = (self._k(X, X, ell) if K0 is None else K0) + np.diag(noise + pn + self.jitter)
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
            med = float(np.median(d[np.triu_indices(len(X), 1)]))
            med = med if med > 1e-6 else 1.0  # (near-)identical inputs: don't collapse the lengthscale
        else:
            med = 1.0
        ells = [self.lengthscale] if self.lengthscale else list(med * np.logspace(-1, 1, self.grid))
        noises = [self.noise] if self.noise is not None else list(np.logspace(-4, 0, self.grid))
        best = (-math.inf, ells[0], noises[0])
        for ell in ells:
            K0 = self._k(X, X, ell)
            for nz in noises:
                v = self._lml(X, ys, ell, nz, pn, K0)
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


class AdditiveGPR:
    """GP with an additive kernel over blocks of the feature vector: k(x, x') = Σ_m RBF_m(x[b_m], x'[b_m]).

    For a program node the blocks are the per-module embeddings (`BOSelector` sets `blocks` from the node's
    module dims when left None). A sum of independent GPs is a GP with the summed kernel, so unseen
    combinations of modules are predicted from the modules seen separately, and `predict(X, block=m)` gives the
    posterior of the m-th component alone (identified up to a constant; compare it between points, not to y).
    Each block's lengthscale is its own median heuristic times one shared multiplier, so the LML grid stays
    (multiplier × noise) as in `GPR`. With a single block this is exactly `GPR`.
    """

    def __init__(self, blocks: list[tuple[int, int]] | None = None, lengthscale: float | None = None,
                 noise: float | None = None, grid: int = 12, jitter: float = 1e-8):
        self.blocks, self.lengthscale, self.noise, self.grid, self.jitter = blocks, lengthscale, noise, grid, jitter
        self._X = self._alpha = self._L = None
        self._y_mu, self._y_sd = 0.0, 1.0

    def _d2(self, A: np.ndarray, B: np.ndarray, m: int) -> np.ndarray:
        a, b = self.blocks[m]
        A, B = A[:, a:b], B[:, a:b]
        return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2 * A @ B.T, 0)

    def _kb(self, A: np.ndarray, B: np.ndarray, m: int, mult: float, d2=None) -> np.ndarray:
        ell = self._med[m] * mult
        return np.exp(-0.5 * (self._d2(A, B, m) if d2 is None else d2) / (ell * ell))

    def _k(self, A: np.ndarray, B: np.ndarray, mult: float, d2s=None) -> np.ndarray:
        return sum(self._kb(A, B, m, mult, None if d2s is None else d2s[m]) for m in range(len(self.blocks)))

    def _lml(self, y, K0, noise, pn) -> float:
        K = K0 + np.diag(noise + pn + self.jitter)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -math.inf
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return float(-0.5 * y @ a - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * math.log(2 * math.pi))

    def fit(self, X, y, noise=None) -> "AdditiveGPR":
        X, y = np.asarray(X, float), np.asarray(y, float)
        if self.blocks is None:
            self.blocks = [(0, X.shape[1])]
        self._y_mu, self._y_sd = float(y.mean()), float(y.std() or 1.0)
        ys = (y - self._y_mu) / self._y_sd
        pn = np.zeros(len(y)) if noise is None else (np.asarray(noise, float) / self._y_sd) ** 2
        self._med, d2s = [], []
        for m, (a, b) in enumerate(self.blocks):
            d2 = self._d2(X, X, m)
            d2s.append(d2)
            if len(X) > 1:
                d = np.sqrt(d2)
                med = float(np.median(d[np.triu_indices(len(X), 1)]))
            else:
                med = 1.0
            # (near-)identical texts in this block would make the kernel a delta function; fall back to unit scale
            self._med.append(med if med > 1e-6 else 1.0)
        mults = [self.lengthscale] if self.lengthscale else list(np.logspace(-1, 1, self.grid))
        noises = [self.noise] if self.noise is not None else list(np.logspace(-4, 0, self.grid))
        best = (-math.inf, mults[0], noises[0])
        for mult in mults:
            K0 = self._k(X, X, mult, d2s)
            for nz in noises:
                v = self._lml(ys, K0, nz, pn)
                if v > best[0]:
                    best = (v, mult, nz)
        _, self.mult_, self.noise_ = best
        self.mult_, self.noise_ = float(self.mult_), float(self.noise_)
        self.ell_ = [float(m * self.mult_) for m in self._med]
        K = self._k(X, X, self.mult_) + np.diag(self.noise_ + pn + self.jitter)
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, ys))
        self._X = X
        return self

    def predict(self, X, block: int | None = None) -> tuple[list[float], list[float]]:
        """(mean, variance) of f (block=None) or of the block-m component alone (zero-mean, in y units)."""
        X = np.asarray(X, float)
        # prior variance of the full f is len(blocks) (unit RBFs summed); of one component, 1
        Ks = self._k(X, self._X, self.mult_) if block is None else self._kb(X, self._X, block, self.mult_)
        prior = float(len(self.blocks)) if block is None else 1.0
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.maximum(prior - np.sum(v * v, 0), 1e-12)
        shift = self._y_mu if block is None else 0.0
        return list(mu * self._y_sd + shift), list(var * self._y_sd ** 2)


class PIT:
    """Probability-integral transform of the targets: y -> rank-based ECDF -> standard-normal quantile.

    A GP assumes Gaussian residuals on an unbounded scale; a bounded, skewed metric (F1 in [0, 1] piling up near
    its ceiling) violates both. Mapping each y to Φ⁻¹((rank − 0.5) / n) makes the training targets exactly
    standard-normal by construction and is scale-free (only the order of the y's matters). The ½ offset keeps the
    ECDF strictly inside (0, 1), so no target maps to ±∞. Ties share their average rank. Acquisitions are monotone in
    the target, so ranking candidates in the transformed space is the same decision; predictions are reported in
    that space (there is no exact inverse for new points, only interpolation of the ECDF).
    """

    def __init__(self, eps: float = 0.5):
        self.eps = eps  # rank offset; 0.5 = mid-rank ECDF, (r − eps) / n

    def fit_transform(self, y) -> list[float]:
        y = np.asarray(y, float)
        n = len(y)
        order = np.argsort(y, kind="mergesort")
        ranks = np.empty(n, float)
        ranks[order] = np.arange(1, n + 1)
        # average ranks for ties, so identical targets get identical z
        for v in np.unique(y):
            m = y == v
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
        u = (ranks - self.eps) / n
        z = _ndtri(u)
        self.y_, self.z_ = y[order], z[order]
        return [float(v) for v in z]

    def transform(self, y) -> list[float]:
        """Interpolate the fitted ECDF for new values (reporting / local slopes)."""
        return [float(v) for v in np.interp(np.asarray(y, float), self.y_, self.z_)]

    def slope(self, y) -> list[float]:
        """dz/dy at each y by central difference on the fitted map, floored at the global slope: used to carry a
        standard error given in y units into z units."""
        y = np.asarray(y, float)
        span = float(self.y_[-1] - self.y_[0]) or 1.0
        h = 0.05 * span
        lo, hi = np.array(self.transform(y - h)), np.array(self.transform(y + h))
        local = (hi - lo) / (2 * h)
        global_ = float(self.z_[-1] - self.z_[0]) / span
        return [float(max(s, global_)) for s in local]


def _ndtri(p: np.ndarray) -> np.ndarray:
    """Standard-normal quantile without scipy (Acklam's rational approximation, |rel err| < 1.2e-9)."""
    p = np.asarray(p, float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    m = p < lo
    q = np.sqrt(-2 * np.log(p[m]))
    out[m] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    m = p > hi
    q = np.sqrt(-2 * np.log(1 - p[m]))
    out[m] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    m = (p >= lo) & (p <= hi)
    q = p[m] - 0.5
    r = q * q
    out[m] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return out
