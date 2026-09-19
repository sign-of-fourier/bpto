"""Acquisition functions over (mean, var) for a discrete candidate set."""
from __future__ import annotations

import math
import random as _random


def _phi(z):  # standard normal pdf
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):  # standard normal cdf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


class EI:
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def __call__(self, mean, var, best_y):
        out = []
        for m, v in zip(mean, var):
            s = math.sqrt(max(v, 1e-12))
            z = (m - best_y - self.xi) / s
            out.append((m - best_y - self.xi) * _Phi(z) + s * _phi(z))
        return out


class UCB:
    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def __call__(self, mean, var, best_y):
        return [m + self.kappa * math.sqrt(max(v, 0)) for m, v in zip(mean, var)]


class Thompson:
    """Independent posterior draw per candidate (ignores cross-covariance; fine for ranking)."""

    def __init__(self, seed: int | None = None):
        self.rng = _random.Random(seed)

    def __call__(self, mean, var, best_y):
        return [self.rng.gauss(m, math.sqrt(max(v, 0))) for m, v in zip(mean, var)]


class Mean:
    """Pure exploitation."""

    def __call__(self, mean, var, best_y):
        return list(mean)


# ---- batch acquisitions -------------------------------------------------------------------------
# (mean, cov, best_y, q) -> q candidate indices, chosen jointly. `BOSelector.top(k)` with `batch=` set hands
# the joint posterior over the candidates here when k > 1; k == 1 never reaches this file's batch code.
# The point of a joint choice: sibling candidates (near-identical embeddings) have near-identical marginal EI
# and all rank together under the per-candidate acquisitions above, so a top-k by EI is k copies of one bet.


class KrigingBeliever:
    """Greedy fantasy batch: pick the argmax of `base`, condition the posterior on its mean as if observed
    (Ginsbourger et al. 2010's kriging believer), repeat. Deterministic and cheap: q rank-one updates."""

    def __init__(self, base=None):
        self.base = base or EI(xi=0.0)

    def __call__(self, mean, cov, best_y, q):
        import numpy as np
        mu, C = np.asarray(mean, float).copy(), np.asarray(cov, float).copy()
        picked = []
        for _ in range(min(q, len(mu))):
            acq = np.asarray(self.base(list(mu), list(np.diag(C)), best_y), float)
            acq[picked] = -np.inf
            i = int(np.argmax(acq))
            picked.append(i)
            # Believing f(x_i) = mu_i moves no mean (mu_i is the mean) and removes what x_i explains from the rest.
            c = C[:, i] / max(C[i, i], 1e-12)
            C = C - np.outer(c, C[i, :])
            best_y = max(best_y, float(mu[i]))
        return picked


class QEI:
    """Monte Carlo joint q-EI, greedy batch construction (BoTorch's default). Draw S joint samples of f over the
    candidates once; then add, one at a time, the candidate that most raises E_s[max(0, max_batch f_s - best_y)].
    The samples are shared across the greedy steps, so the batch score is a common-random-numbers comparison."""

    def __init__(self, n_samples: int = 512, seed: int | None = 0):
        self.n_samples, self.seed = n_samples, seed
        self.last: dict = {}

    def __call__(self, mean, cov, best_y, q):
        import numpy as np
        mu, C = np.asarray(mean, float), np.asarray(cov, float)
        rng = np.random.default_rng(self.seed)
        # eigen-decomposition rather than Cholesky: candidates at training inputs give a singular posterior
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        A = V * np.sqrt(np.maximum(w, 0.0))
        F = mu + rng.standard_normal((self.n_samples, len(mu))) @ A.T           # (S, n) joint draws
        picked, cur = [], np.full(self.n_samples, -np.inf)
        for _ in range(min(q, len(mu))):
            gain = np.maximum(np.maximum(cur[:, None], F) - best_y, 0.0).mean(0)   # (n,) batch qEI if i is added
            gain[picked] = -np.inf
            i = int(np.argmax(gain))
            picked.append(i)
            cur = np.maximum(cur, F[:, i])
        self.last = {"qei": float(np.maximum(cur - best_y, 0.0).mean())}
        return picked


class QuantecarloQEI:
    """Joint q-EI by the hosted quantecarlo service, acquisition only: the posterior (mean, cov, best_y) goes
    over the wire, the surrogate does not. Everything `BOSelector` fits - noise, PIT, incumbent, PCA, module
    blocks - stays as it is; this is the third implementation of the same (mean, cov, best_y, q) -> indices
    contract as `QEI` above, with an exhaustive / screened search over subsets instead of greedy construction.
    The service scores improvement on exp of the posterior it is sent (q-EI on a lognormal objective), so the
    posterior must be normal-scale: `BOSelector`'s `transform="pit"` gives exactly that, and `best_y` must be
    on the same scale. A raw-scale `best_y` is not refused; the response carries a warning and the client
    logs it. Picks therefore differ from `QEI` (plain EI on the PIT scale) by design, not by error.
    Requires the `quantecarlo` package (>= 0.7.1); `client` is a `quantecarlo.QEIClient` or None for the default.
    `**select` are per-call `QEIClient.select` fields (pi_floor, seed, ei_budget, dtype, ...)."""

    def __init__(self, client=None, **select):
        self.client, self.select = client, select
        self.last: dict = {}

    def __call__(self, mean, cov, best_y, q):
        if self.client is None:
            from quantecarlo import QEIClient
            self.client = QEIClient()
        out = self.client.select(mean, cov, best_y, q, **self.select)
        self.last = {k: out[k] for k in ("qei", "regime", "n_cands", "n_sampled", "n_batches")}
        return list(out["indices"])
