"""Accuracy against wall-clock per arm (-> wallclock.png, wallclock.json). Run from the repo root; needs runs/osha_sir/.

Time of each candidate: GEPA records when a program was found in metric calls (`discovery_calls`), not seconds. It
also writes `generated_best_outputs_valset/task_*/iter_*_prog_<j>.json` when program j is best on a val row; the
earliest mtime of those is when j's validation finished (about half the programs have one). Run start = summary.json
mtime - secs. Calls -> seconds is interpolated linearly between those anchors, (0 calls, seed evaluated) and
(total calls, end); throughput is steady within a run (7-8 calls/s at q=1, 10-12 at q=4).
"""
import glob
import json
import os
import re
from pathlib import Path
from statistics import mean

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
GEPA = Path("runs/osha_sir/gepa")
ARMS = [("q1", "GEPA q=1"), ("independent4", "GEPA independent q=4"), ("qei4", "GEPA + q-EI q=4")]
COLOR = {"q1": "#2a78d6", "independent4": "#eb6834", "qei4": "#1baf7a"}  # validated categorical slots 1-3
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"

runs = [json.loads(l) for l in (HERE / "evaluations.jsonl").read_text().splitlines()]
ref = json.loads((HERE / "reference_scores.json").read_text())["holdout"]
ho = lambda h: mean(x["acc"] for x in h["reps"])


def timeline(r):
    """(seconds, best val so far minus the run's own seed val) at each candidate's discovery. The seed prompt's
    val score varies .715-.755 across runs (Micro is not deterministic), so raw levels would compare noise."""
    g = GEPA / f"main_{r['arm']}_s{r['seed']}"
    start = os.path.getmtime(g / "summary.json") - r["secs"]
    first = {}
    for f in glob.glob(str(g / "gepa_state/generated_best_outputs_valset/task_*/*.json")):
        p = int(re.match(r"iter_\d+_prog_(\d+)", os.path.basename(f))[1])
        first[p] = min(first.get(p, 1e18), os.path.getmtime(f) - start)
    calls = r["discovery_calls"]
    anchors = sorted({(0, first[0])} | {(calls[p], t) for p, t in first.items()} | {(r["total_metric_calls"], r["secs"])})
    ac, at = zip(*anchors)
    ts = np.interp(calls, ac, at)
    best, out = 0.0, []
    for t, v in sorted(zip(ts, r["val_scores"])):
        best = max(best, v)
        out.append((float(t), best - r["seed_val"]))
    return out


grid = np.linspace(0, max(r["secs"] for r in runs), 400)
summary = {}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
for arm, label in ARMS:
    rs = [r for r in runs if r["arm"] == arm]
    curves = []
    for r in rs:
        tl = timeline(r)
        ts, vs = zip(*tl)
        a1.step(list(ts) + [r["secs"]], list(vs) + [vs[-1]], where="post", color=COLOR[arm], alpha=0.25, linewidth=1)
        curves.append(np.array([max([v for t, v in tl if t <= x], default=np.nan) for x in grid]))
    m = np.nanmean(curves, axis=0)
    end = mean(r["secs"] for r in rs)
    keep = grid <= end
    a1.plot(grid[keep], m[keep], color=COLOR[arm], linewidth=2.5, label=label)
    a1.scatter([end], [m[keep][-1]], s=64, color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=4)
    summary[arm] = {"mean_secs": round(end, 1), "mean_val_gain_at_end": round(float(m[keep][-1]), 4),
                    "mean_val_gain_at": {str(t): round(float(np.interp(t, grid, m)), 4) for t in (100, 200, 300, 422)}}
    for r in rs:
        a2.scatter(r["secs"], ho(r["holdout"]), s=64, color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=3)
    summary[arm]["holdout_mean"] = round(mean(ho(r["holdout"]) for r in rs), 4)

b2 = ho(ref["B2"])
a2.axhline(b2, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
a2.annotate(f"seed prompt B2 = {b2:.3f}", (530, b2 + 0.002), fontsize=9, color=MUTED, va="bottom")
a1.axhline(0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f} pts" if v else "seed"))
a1.set_title("Best validation accuracy so far, gain over the run's own seed score\n(200 val rows; mean of 5 seeds, thin = each seed)",
             fontsize=10, color=INK, loc="left")
a2.set_title("Holdout accuracy of the returned prompt vs run time", fontsize=10, color=INK, loc="left")
a1.set_xlabel("wall-clock (s)", fontsize=9, color=MUTED)
a2.set_xlabel("wall-clock for the whole run, B = 5,000 calls (s)", fontsize=9, color=MUTED)
a1.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
a1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
for ax in (a1, a2):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
a1.set_xlim(0, None)
fig.text(0.01, 0.01, "Dot on the left = arm's mean finish time. Val is what GEPA selects on; its gains did not transfer to "
         "the holdout (right).\nCandidate times are interpolated from GEPA's call counts (see wallclock.py).",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / "wallclock.png", dpi=150, facecolor=SURFACE)
(HERE / "wallclock.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
