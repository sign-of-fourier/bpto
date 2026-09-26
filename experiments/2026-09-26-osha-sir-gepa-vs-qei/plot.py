"""Holdout accuracy and wall-clock per run, by arm (evaluations.jsonl -> arms.png). Run from the repo root."""
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ARMS = [("q1", "GEPA q=1"), ("independent4", "GEPA independent q=4"), ("qei4", "GEPA + q-EI q=4")]
COLOR = {"q1": "#2a78d6", "independent4": "#eb6834", "qei4": "#1baf7a"}  # validated categorical slots 1-3
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"

runs = [json.loads(l) for l in (HERE / "evaluations.jsonl").read_text().splitlines()]
ref = json.loads((HERE / "reference_scores.json").read_text())["holdout"]
ho = lambda r: mean(x["acc"] for x in r["reps"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
panels = [("Holdout accuracy (200 rows, mean of 2 scorings)", lambda r: ho(r["holdout"])),
          ("Wall-clock per run (s), B = 5,000 calls", lambda r: r["secs"])]
for ax, (title, get) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    for i, (arm, label) in enumerate(ARMS):
        vals = [get(r) for r in runs if r["arm"] == arm]
        xs = [i + (j - 2) * 0.06 for j in range(len(vals))]
        ax.scatter(xs, vals, s=64, color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=3)
        m = mean(vals)
        ax.hlines(m, i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax.annotate(f"{m:.3f}" if m < 1 else f"{m:.0f} s", (i + 0.28, m), va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(ARMS)), [l for _, l in ARMS], fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlim(-0.5, len(ARMS) - 0.3)
b2 = ho(ref["B2"])
axes[0].axhline(b2, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
axes[0].annotate(f"seed prompt B2 = {b2:.3f}", (-0.45, b2 + 0.002), fontsize=9, color=MUTED, va="bottom")
axes[1].set_ylim(0, None)
fig.text(0.01, 0.01, "Each dot is one seed (5 per arm); bar = mean. Micro task model, Lite reflector, 10-row gate, "
         "200-row validation. 8 of 15 runs returned B2 itself.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(HERE / "arms.png", dpi=150, facecolor=SURFACE)
