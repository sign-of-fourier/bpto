"""Observability: event log, progress printing, and tree visualisation.

Listeners are `fn(event, node)` callables registered on `tree.listeners`; events are
"proposed", "expanded", "evaluated".
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TextIO

from .tree import Node, Tree


def _summary(node: Node) -> dict:
    d = {"id": node.id, "parent": node.parent_id, "depth": node.depth, "op": node.origin.op,
         "state": str(node.state), "template": node.prompt.template}
    if node.evaluation:
        e = node.evaluation
        d.update(score=e.score, feasible=e.feasible, n=e.n, metrics=e.metrics)
    return d


class EventLog:
    """Append-only JSONL: one line per event, with a node summary and client usage so far."""

    def __init__(self, path: str | Path, tree: Tree | None = None):
        self.path = Path(path)
        self._f = open(self.path, "a")
        if tree is not None:
            self.attach(tree)

    def attach(self, tree: Tree) -> "EventLog":
        self._tree = tree
        tree.listeners.append(self)
        return self

    def __call__(self, event: str, node: Node) -> None:
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event,
               "usage": self._tree.task.client.usage.model_dump(), "node": _summary(node)}
        self._f.write(json.dumps(rec, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]


class Progress:
    """One line per evaluated node (and a summary per expansion) to a stream."""

    def __init__(self, tree: Tree, stream: TextIO = sys.stderr, every: int = 1):
        self.tree, self.stream, self.every, self._n, self._t0 = tree, stream, every, 0, time.monotonic()
        tree.listeners.append(self)

    def __call__(self, event: str, node: Node) -> None:
        if event == "evaluated":
            self._n += 1
            if self._n % self.every == 0:
                e = node.evaluation
                best = self.tree.best()
                self.stream.write(
                    f"[{time.monotonic() - self._t0:6.1f}s] eval #{self._n} d={node.depth} {node.origin.op:8} "
                    f"score={e.score:+.4f}{'' if e.feasible else ' (infeasible)'} n={e.n} "
                    f"best={best.score:+.4f} calls={self.tree.task.client.usage.calls}\n")
        elif event == "expanded":
            self.stream.write(f"[{time.monotonic() - self._t0:6.1f}s] expanded {node.id} d={node.depth} "
                              f"-> {len(self.tree.children[node.id])} children (tree={len(self.tree)})\n")


# ---- visualisation ---------------------------------------------------------------------

def _label(node: Node, width: int) -> str:
    t = node.prompt.template.replace("\n", " ")
    t = t[:width] + ("…" if len(t) > width else "")
    s = f"{node.score:+.3f}" if node.evaluated else "  —   "
    flag = "" if (not node.evaluated or node.evaluation.feasible) else "!"
    return f"{s}{flag} [{node.origin.op}] {t}"


def tree_text(tree: Tree, width: int = 60, sort_by_score: bool = True) -> str:
    """Indented text rendering; best-scoring children first."""
    lines: list[str] = []

    def walk(node: Node, prefix: str, last: bool, is_root: bool):
        branch = "" if is_root else ("└─ " if last else "├─ ")
        lines.append(prefix + branch + _label(node, width))
        kids = tree.child_nodes(node)
        if sort_by_score:
            kids.sort(key=lambda n: (n.score if n.score is not None else float("-inf")), reverse=True)
        ext = "" if is_root else ("   " if last else "│  ")
        for i, k in enumerate(kids):
            walk(k, prefix + ext, i == len(kids) - 1, False)

    walk(tree.root, "", True, True)
    return "\n".join(lines)


def tree_dot(tree: Tree, width: int = 40, color_metric: str | None = None) -> str:
    """Graphviz DOT; nodes shaded by score (or `color_metric`). `dot -Tpng tree.dot -o tree.png`."""
    vals = {n.id: (n.evaluation.metrics.get(color_metric) if color_metric else n.score)
            for n in tree if n.evaluated}
    lo, hi = (min(vals.values()), max(vals.values())) if vals else (0, 1)
    out = ["digraph tree {", "  rankdir=TB; node [shape=box, style=filled, fontname=Helvetica, fontsize=9];"]
    for n in tree:
        v = vals.get(n.id)
        fill = "#eeeeee" if v is None else "#%02x%02x%02x" % _ramp((v - lo) / (hi - lo) if hi > lo else 1.0)
        lbl = _label(n, width).replace('"', "'")
        out.append(f'  "{n.id}" [label="{lbl}", fillcolor="{fill}"];')
        if n.parent_id:
            out.append(f'  "{n.parent_id}" -> "{n.id}";')
    out.append("}")
    return "\n".join(out)


def _ramp(x: float) -> tuple[int, int, int]:
    """white -> green."""
    x = max(0.0, min(1.0, x))
    return (int(255 - 155 * x), int(255 - 55 * x), int(255 - 155 * x))


def plot_tree(tree: Tree, path: str | Path, color_metric: str | None = None, figsize=(11, 6)) -> Path:
    """Layered layout (depth on y), coloured by score; Pareto/best node outlined. Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos: dict[str, tuple[float, float]] = {}
    x_counter = [0.0]

    def layout(node: Node) -> float:
        kids = tree.child_nodes(node)
        if not kids:
            x = x_counter[0]; x_counter[0] += 1.0
        else:
            x = sum(layout(k) for k in kids) / len(kids)
        pos[node.id] = (x, -node.depth)
        return x

    layout(tree.root)
    vals = {n.id: (n.evaluation.metrics.get(color_metric) if color_metric else n.score) for n in tree if n.evaluated}
    fig, ax = plt.subplots(figsize=figsize)
    for n in tree:
        if n.parent_id:
            (x0, y0), (x1, y1) = pos[n.parent_id], pos[n.id]
            ax.plot([x0, x1], [y0, y1], color="#bbbbbb", lw=0.8, zorder=1)
    ev = [n for n in tree if n.id in vals]
    un = [n for n in tree if n.id not in vals]
    if un:
        ax.scatter([pos[n.id][0] for n in un], [pos[n.id][1] for n in un], c="#dddddd", s=30, zorder=2, label="unevaluated")
    if ev:
        sc = ax.scatter([pos[n.id][0] for n in ev], [pos[n.id][1] for n in ev], c=[vals[n.id] for n in ev],
                        cmap="viridis", s=40, zorder=3, label="evaluated")
        fig.colorbar(sc, ax=ax, label=color_metric or "score")
        best = tree.best()
        if best:
            ax.scatter(*pos[best.id], s=160, facecolors="none", edgecolors="red", lw=1.5, zorder=4, label="best")
    ax.set_yticks(range(0, -tree.max_depth - 1, -1))
    ax.set_yticklabels(range(tree.max_depth + 1))
    ax.set_ylabel("depth"); ax.set_xticks([])
    ax.set_title(f"{len(tree)} nodes, {len(ev)} evaluated")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def lineage(tree: Tree, node: Node, width: int = 80) -> str:
    """Root-to-node path with scores: how did we get here?"""
    chain = list(reversed(tree.ancestors(node))) + [node]
    return "\n".join(f"{'  ' * i}{_label(n, width)}" for i, n in enumerate(chain))
