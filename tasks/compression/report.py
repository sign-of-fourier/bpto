"""Reporting for compression runs: a text table of the Pareto front and an optional plot."""
from __future__ import annotations

from pathlib import Path

from bpto import Tree

MAXIMIZE = {"f1": True, "template_tokens": False}


def pareto_table(tree: Tree, maximize: dict[str, bool] = MAXIMIZE, width: int = 70) -> str:
    rows = sorted(tree.pareto(maximize), key=lambda n: n.evaluation.metrics.get("template_tokens", 0))
    # ties on the front collapse to one row (the shallowest), with a count
    groups: dict[tuple, list] = {}
    for n in rows:
        m = n.evaluation.metrics
        groups.setdefault((round(m.get("f1", 0), 4), m.get("template_tokens", 0)), []).append(n)
    lines = [f"{'f1':>6} {'tokens':>7} {'ties':>4} {'depth':>5} {'op':8} {'feas':4}  prompt"]
    for (f1, tok), ns in groups.items():
        n = min(ns, key=lambda x: x.depth)
        t = n.prompt.template.replace("\n", "⏎")
        lines.append(f"{f1:6.3f} {tok:7.0f} {len(ns):4d} {n.depth:5d} {n.origin.op:8} "
                     f"{'y' if n.evaluation.feasible else 'n':4}  {t[:width]}{'…' if len(t) > width else ''}")
    return "\n".join(lines)


def plot_pareto(tree: Tree, path: str | Path, maximize: dict[str, bool] = MAXIMIZE) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = tree.evaluated_nodes()
    front = {n.id for n in tree.pareto(maximize)}
    xs = [n.evaluation.metrics.get("template_tokens", 0) for n in ev]
    ys = [n.evaluation.metrics.get("f1", 0) for n in ev]
    depth = [n.depth for n in ev]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(xs, ys, c=depth, cmap="viridis", s=28, alpha=0.75, label="evaluated nodes")
    fx = sorted((x, y) for n, x, y in zip(ev, xs, ys) if n.id in front)
    if fx:
        ax.plot([p[0] for p in fx], [p[1] for p in fx], "r-o", ms=5, lw=1.2, label="Pareto front")
    ax.set_xlabel("template tokens (lower is better)")
    ax.set_ylabel("F1")
    ax.set_title(f"Prompt compression — {len(ev)} evaluated nodes, depth ≤ {tree.max_depth}")
    fig.colorbar(sc, ax=ax, label="depth")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
