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
