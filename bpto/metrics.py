"""Metrics are named float vectors. Aggregation happens here; scalarization in scoring."""
from __future__ import annotations

import math

Metrics = dict[str, float]


def mean_metrics(rows: list[Metrics]) -> Metrics:
    if not rows:
        return {}
    keys = {k for r in rows for k in r}
    return {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in sorted(keys)}


def std_metrics(rows: list[Metrics]) -> Metrics:
    if len(rows) < 2:
        return {k: 0.0 for k in (rows[0] if rows else {})}
    mu = mean_metrics(rows)
    return {
        k: math.sqrt(sum((r.get(k, 0.0) - mu[k]) ** 2 for r in rows) / (len(rows) - 1))
        for k in mu
    }
