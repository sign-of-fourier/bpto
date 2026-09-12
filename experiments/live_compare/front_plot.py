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


def mean_front(path: Path, out: Path, key: str = "f1"):
    """Mean ± SE over seeds of the shortest prompt meeting each accuracy bar (relative to the seed's root),
    plus the paired gepa-bo difference with its SE band."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import statistics as st
    from .analyze_compress import shortest_at
    runs = {}
    for d in sorted(path.iterdir()):
        if d.is_dir() and (d / "tree.json").exists():
            arm, s = d.name.rsplit("-s", 1); runs[(arm, int(s))] = load_run(d)
    arms = sorted({a for a, _ in runs}, reverse=True); col = {"gepa": "tab:orange", "bo": "tab:blue"}
    seeds = sorted({s for _, s in runs if all((a, s) in runs for a in arms)})
    gaps = [i / 100 for i in range(0, 21)]
    tok = {a: [] for a in arms}  # per gap: list over seeds
    for g in gaps:
        for a in arms:
            tok[a].append([shortest_at(runs[(a, s)]["cands"], runs[(a, s)]["root_" + key] - g)["tokens"] for s in seeds])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a in arms:
        m = [st.mean(v) for v in tok[a]]; se = [st.stdev(v) / len(v) ** 0.5 for v in tok[a]]
        ax[0].plot(gaps, m, color=col[a], lw=2, label=f"{a} (mean, n={len(seeds)})")
        ax[0].fill_between(gaps, [x - s for x, s in zip(m, se)], [x + s for x, s in zip(m, se)], color=col[a], alpha=.2)
    d = [[g - b for g, b in zip(tok["gepa"][i], tok["bo"][i])] for i in range(len(gaps))]
    md = [st.mean(v) for v in d]; sd = [st.stdev(v) / len(v) ** 0.5 for v in d]
    ax[1].plot(gaps, md, color="k", lw=2, label="gepa − bo tokens (paired by seed)")
    ax[1].fill_between(gaps, [x - s for x, s in zip(md, sd)], [x + s for x, s in zip(md, sd)], color="k", alpha=.15, label="±1 SE")
    ax[1].fill_between(gaps, [x - 2 * s for x, s in zip(md, sd)], [x + 2 * s for x, s in zip(md, sd)], color="k", alpha=.07, label="±2 SE")
    ax[1].axhline(0, color="gray", lw=1)
    ax[0].set(xlabel=f"accuracy bar: train {key} ≥ root − gap", ylabel="shortest prompt found (template tokens)", yscale="log",
              title="mean shortest prompt at each bar, ±1 SE over seeds"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[0].set_yticks([5, 7, 10, 15, 20, 30, 50, 100]); ax[0].set_yticklabels([5, 7, 10, 15, 20, 30, 50, 100])
    ax[1].set(xlabel=f"accuracy bar: train {key} ≥ root − gap", ylabel="tokens", title="paired difference (positive = BO shorter)"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.suptitle(f"Compression, {len(seeds)} seeds: how much shorter, at each accuracy bar")
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); main(Path(a.path), Path(a.out))
    mean_front(Path(a.path), Path(a.out).with_name(Path(a.out).stem + "_mean.png"))
