"""The compression readout as a graph: template tokens vs train F1, every fully-evaluated prompt, per arm.

    python -m experiments.live_compare.front_plot runs/compress_v2 --out <dir>/fronts.png

Left: all prompts from all seeds pooled (points), each arm's pooled non-dominated front (line), root marked.
Right: per-seed fronts as faint lines so the spread is visible. A region with no points at some accuracy
level means the search found nothing there - the honest answer when the user's bar sits in that region.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .analyze_compress import load_run


def front(points):
    pts = sorted(points, key=lambda p: (p[0], -p[1])); out = []
    for t, f in pts:
        if not out or f > out[-1][1]:
            out.append((t, f))
    return out


def main(path: Path, out: Path, key: str = "f1"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = {}
    for d in sorted(path.iterdir()):
        if d.is_dir() and (d / "tree.json").exists():
            arm, s = d.name.rsplit("-s", 1); runs[(arm, int(s))] = load_run(d)
    arms = sorted({a for a, _ in runs}, reverse=True); col = {"gepa": "tab:orange", "bo": "tab:blue"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    root = next(iter(runs.values()))
    for a in arms:
        pts = [(c["tokens"], c[key]) for (aa, _), r in runs.items() if aa == a for c in r["cands"] if c["depth"] > 0]
        ax[0].scatter([p[0] for p in pts], [p[1] for p in pts], s=14, alpha=.35, color=col[a], label=f"{a}: {len(pts)} prompts")
        fr = front(pts); ax[0].step([p[0] for p in fr], [p[1] for p in fr], where="post", color=col[a], lw=2)
        for (aa, s), r in runs.items():
            if aa != a:
                continue
            fr = front([(c["tokens"], c[key]) for c in r["cands"]])
            ax[1].step([p[0] for p in fr], [p[1] for p in fr], where="post", color=col[a], alpha=.35, label=a if s == 0 else None)
    roots = [r["root_" + key] for r in runs.values()]
    for x in ax:
        x.scatter([root["root_tokens"]], [sum(roots) / len(roots)], marker="*", s=200, color="k", zorder=5, label="root (mean over seeds)")
        x.set(xscale="log", xlabel="template tokens (log)"); x.grid(alpha=.3); x.legend(loc="lower right")
        x.set_xticks([4, 6, 8, 10, 15, 20, 30, 50, 100]); x.set_xticklabels([4, 6, 8, 10, 15, 20, 30, 50, 100])
    ax[0].set(ylabel=f"train {key}", title=f"all seeds pooled: every prompt + each arm's front"); ax[1].set(title="per-seed fronts")
    ax[0].set_ylim(0.6, 1.01)
    fig.suptitle("Compression: shortest prompt at each accuracy level, GEPA vs BO (same mutator, budget, gate)")
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); main(Path(a.path), Path(a.out))
