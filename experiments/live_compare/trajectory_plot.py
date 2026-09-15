"""Where the optimizer was at each point of its budget: the anytime curve (incumbent vs rollouts) per arm.

    python -m experiments.live_compare.trajectory_plot runs/hotpot --metric best_f1 --out trajectory.png
    python -m experiments.live_compare.trajectory_plot runs/compress_v2 --metric best_tokens --lower-better --out trajectory.png

Reads the per-round `curve` rows in results.jsonl (written by compress.py / hotpot.py). Left: mean ± 1 SE over
seeds of the incumbent (best fully-evaluated candidate) as a step function of rollouts spent, one line per arm.
Right: every seed as a faint line, so the spread and the per-run jumps are visible. For a constrained run the
floor schedule is drawn as a panel *below* (same x, its own axis - never a second y-axis on the same plot), and
the sawtooth in the incumbent (a floor raise can knock the incumbent infeasible) is the thing to look at.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

COLORS = {"gepa": "#eb6834", "bo": "#2a78d6", "mipro": "#1baf7a", "gepa3": "#eda100"}


def step_at(curve, x, metric):
    """Incumbent value after the last recorded point with calls <= x (None before the first)."""
    v = None
    for c in curve:
        if c["calls"] <= x and c.get(metric) is not None:
            v = c[metric]
    return v


def main(path: Path, out: Path, metric: str, lower_better: bool, grid_step: int):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(path / "results.jsonl")]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    arms = [a for a in COLORS if a in by] + [a for a in by if a not in COLORS]
    xmax = max(r["rollouts"] for r in rows)
    grid = list(range(0, xmax + 1, grid_step))
    constrained = any("floor" in c for r in rows for c in r["curve"])
    if constrained:
        fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
        ax, axf = axes[0], axes[1]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True); axf = None
    for a in arms:
        runs = sorted(by[a], key=lambda r: r["seed"])
        series = [[step_at(r["curve"], x, metric) for x in grid] for r in runs]
        xs, mu, se = [], [], []
        for i, x in enumerate(grid):
            vals = [s[i] for s in series if s[i] is not None]
            if len(vals) == len(series):  # only where every seed has an incumbent
                xs.append(x); mu.append(statistics.mean(vals)); se.append(statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0)
        ax[0].step(xs, mu, where="post", color=COLORS.get(a, "#52514e"), lw=2, label=f"{a} (n={len(runs)})")
        ax[0].fill_between(xs, [m - s for m, s in zip(mu, se)], [m + s for m, s in zip(mu, se)], step="post", color=COLORS.get(a, "#52514e"), alpha=.15, lw=0)
        if xs:
            ax[0].annotate(f"{mu[-1]:.3g}", (xs[-1], mu[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color="#52514e")
        for r in runs:
            pts = [(c["calls"], c[metric]) for c in r["curve"] if c.get(metric) is not None]
            ax[1].step([p[0] for p in pts], [p[1] for p in pts], where="post", color=COLORS.get(a, "#52514e"), alpha=.4, lw=1.2, label=a if r is runs[0] else None)
    root_key = "root_tokens" if metric == "best_tokens" else metric.replace("best_", "root_")
    roots = [r[root_key] for r in rows if root_key in r]
    for x in ax:
        if roots:
            x.axhline(statistics.mean(roots), color="#0b0b0b", lw=1, ls=":", label="root (mean over seeds)")
        x.grid(alpha=.25); x.legend(loc="upper right" if lower_better else "lower right", fontsize=9)
        x.set_xlim(0, xmax)
    if lower_better and metric == "best_tokens":
        for x in ax:
            x.set_yscale("log")
    ax[0].set(ylabel=metric.replace("_", " ") + (" (lower is better)" if lower_better else ""), title="incumbent vs rollouts, mean ± 1 SE over seeds")
    ax[1].set(title="per seed")
    if axf is not None:
        for j in range(2):
            for r in rows:
                pts = [(c["calls"], c["floor"]) for c in r["curve"] if c.get("floor")]
                axf[j].step([p[0] for p in pts], [p[1] for p in pts], where="post", color="#52514e", alpha=.5, lw=1)
            axf[j].set(xlabel="rollouts (task-model calls)"); axf[j].grid(alpha=.25)
        axf[0].set(ylabel="floor")
    else:
        for x in ax:
            x.set(xlabel="rollouts (task-model calls)")
    fig.suptitle(f"{path.name}: anytime {metric.replace('_', ' ')} per arm (equal rollout budget)")
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--metric", default="best_f1")
    ap.add_argument("--lower-better", action="store_true")
    ap.add_argument("--grid-step", type=int, default=25)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    main(a.path, a.out or a.path / "trajectory.png", a.metric, a.lower_better, a.grid_step)
