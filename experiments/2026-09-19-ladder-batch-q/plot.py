"""Redraw curves.png from results/*.json. Palette validated (dataviz six checks, light surface)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "results"
load = lambda f, arm: json.load(open(R / f"{f}.json"))["results"][arm]
C = {"gepa": "#2a78d6", "gepa-ei": "#eb6834", "top2": "#1baf7a", "hosted": "#8e5bd6"}
panels = [
    ("6 skills, easy (budget 600, 100 seeds)", [
        ("gepa q=1", "gepa", "-", load("lm_b600_q1", "gepa-weighted")),
        ("gepa q=2", "gepa", "--", load("lm_b600_q2_qei", "gepa-weighted")),
        ("gepa-ei q=1", "gepa-ei", "-", load("lm_b600_q1", "gepaei")),
        ("gepa-ei q=2 (q-EI)", "gepa-ei", "--", load("lm_b600_q2_qei", "gepaei")),
        ("gepa-ei q=2 (top-2)", "top2", "--", load("lm_b600_q2_topk", "gepaei"))]),
    ("6 skills, hard: noise .3, p_informed .25 (budget 2000, 100 seeds)", [
        ("gepa q=1", "gepa", "-", load("lh_q1", "gepa-weighted")),
        ("gepa q=2", "gepa", "--", load("lh_q2_qei", "gepa-weighted")),
        ("gepa-ei q=1", "gepa-ei", "-", load("lh_q1", "gepaei")),
        ("gepa-ei q=2 (q-EI)", "gepa-ei", "--", load("lh_q2_qei", "gepaei")),
        ("gepa-ei q=2 (top-2)", "top2", "--", load("lh_q2_topk", "gepaei"))]),
    ("12 skills (budget 2000, 30 seeds)", [
        ("gepa q=1", "gepa", "-", load("l12_gepa_q1", "gepa-weighted")),
        ("gepa q=2", "gepa", "--", load("l12_q2_qei", "gepa-weighted")),
        ("gepa-ei q=1", "gepa-ei", "-", load("l12_gepaei_q1", "gepaei")),
        ("gepa-ei q=2 (q-EI)", "gepa-ei", "--", load("l12_q2_qei", "gepaei")),
        ("gepa-ei q=2 (hosted q-EI)", "hosted", "--", load("l12_q2_quantecarlo", "gepaei"))]),
    ("12 skills, q = 1 vs 4 (budget 2000, 30 seeds)", [
        ("gepa q=1", "gepa", "-", load("l12_gepa_q1", "gepa-weighted")),
        ("gepa q=4", "gepa", ":", load("l12_q4_qei", "gepa-weighted")),
        ("gepa-ei q=1", "gepa-ei", "-", load("l12_gepaei_q1", "gepaei")),
        ("gepa-ei q=4 (q-EI)", "gepa-ei", ":", load("l12_q4_qei", "gepaei")),
        ("gepa-ei q=4 (top-4)", "top2", ":", load("l12_q4_topk", "gepaei")),
        ("gepa-ei q=4 (hosted q-EI)", "hosted", ":", load("l12_q4_quantecarlo", "gepaei"))]),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharey=True)
for ax, (title, series) in zip(axes.flat, panels):
    for label, col, ls, r in series:
        x, m, se = r["grid"], r["mean"], r["se"]
        ax.plot(x, m, ls, color=C[col], lw=2, label=label)
        ax.fill_between(x, [a - b for a, b in zip(m, se)], [a + b for a, b in zip(m, se)], color=C[col], alpha=.12, lw=0)
    ax.set_title(title, fontsize=11); ax.set_xlabel("rollouts"); ax.grid(alpha=.25); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
for ax in axes[:, 0]: ax.set_ylabel("true score of best candidate")
fig.suptitle("Synthetic ladder, same rollouts: one parent per round (solid), two (dashed), four (dotted). "
             "GEPA's extra draws are blind; the surrogate's are not.", fontsize=11)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "curves.png", dpi=120)
