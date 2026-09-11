"""Schedule-free readout of a compress.py run directory: what each arm found, at what cost, paired by seed.

    python -m experiments.live_compare.analyze_compress runs/compress_v2 --out experiments/<dir>

Reads tree.json + events.jsonl per arm-seed. A *candidate* is a node evaluated on the full training set;
its discovery rollout is the task-client call count when that evaluation was logged. Reports:
  - threshold table: for accuracy gaps g (train F1 >= root F1 - g), the shortest candidate each arm found
    and when; paired gepa-bo differences with wins
  - union-front regret: tokens above the best either arm found for that seed at each gap
  - rollout accounting: root / minibatch / full evals that advanced the (tokens, F1) front vs did not
  - fronts plot
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

GAPS = [0.0, 0.02, 0.05, 0.10]


def load_run(d: Path):
    tree = json.load(open(d / "tree.json"))
    nodes = {n["id"]: n for n in tree["nodes"]}
    root = nodes[tree["root"]]
    n_train = root["evaluation"]["n"]
    found = {}  # node id -> calls at first full evaluation
    mb_calls = 0
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e["event"] != "evaluated":
            continue
        if e["node"]["n"] == n_train:
            found.setdefault(e["node"]["id"], e["usage"]["calls"])
        else:
            mb_calls += e["node"]["n"]
    cands = []
    for i, n in nodes.items():
        ev = n.get("evaluation")
        if ev and ev["n"] == n_train:
            cands.append({"id": i, "tokens": ev["metrics"]["template_tokens"], "f1": ev["metrics"]["f1"],
                          "found": found.get(i, 0), "depth": n["depth"], "template": n["prompt"]["template"]})
    cands.sort(key=lambda c: c["found"])
    # front-advancing = not dominated by any candidate found earlier
    for k, c in enumerate(cands):
        earlier = cands[:k]
        c["advanced"] = not any(e["tokens"] <= c["tokens"] and e["f1"] >= c["f1"] and (e["tokens"] < c["tokens"] or e["f1"] > c["f1"]) for e in earlier)
    return {"root_f1": root["evaluation"]["metrics"]["f1"], "root_tokens": root["evaluation"]["metrics"]["template_tokens"],
            "n_train": n_train, "cands": cands, "mb_calls": mb_calls,
            "full_calls": n_train * (len(cands) - 1), "total_calls": max(c["found"] for c in cands)}


def shortest_at(cands, floor):
    ok = [c for c in cands if c["f1"] >= floor - 1e-9]
    return min(ok, key=lambda c: (c["tokens"], c["found"]), default=None)


def analyze(path: Path):
    runs = {}
    for d in sorted(path.iterdir()):
        if d.is_dir() and (d / "tree.json").exists() and (d / "events.jsonl").exists():
            arm, s = d.name.rsplit("-s", 1)
            runs[(arm, int(s))] = load_run(d)
    arms = sorted({a for a, _ in runs}, reverse=True)  # gepa, bo
    seeds = sorted({s for _, s in runs if all((a, s) in runs for a in arms)})
    out = [f"# {path}: {len(seeds)} complete seeds x {arms}", "",
           "## Threshold table - shortest candidate with train F1 >= root F1 - gap (tokens @ rollout found)", ""]
    hdr = "| gap | " + " | ".join(f"{a} tokens | {a} found@" for a in arms) + " | paired gepa-bo tokens | BO shorter / ties / n | union-front regret gepa / bo |"
    out += [hdr, "|---|" + "---|" * (2 * len(arms) + 3)]
    for g in GAPS:
        per = {a: [] for a in arms}; regret = {a: [] for a in arms}; diffs = []
        for s in seeds:
            picks = {}
            for a in arms:
                r = runs[(a, s)]
                picks[a] = shortest_at(r["cands"], r["root_f1"] - g)
                per[a].append(picks[a])
            union = shortest_at([c for a in arms for c in runs[(a, s)]["cands"]], runs[(arms[0], s)]["root_f1"] - g)
            for a in arms:
                regret[a].append(picks[a]["tokens"] - union["tokens"])
            diffs.append(picks["gepa"]["tokens"] - picks["bo"]["tokens"])
        cells = []
        for a in arms:
            t = [p["tokens"] for p in per[a]]; f = [p["found"] for p in per[a]]
            cells.append(f"{st.mean(t):.1f} ± {st.stdev(t) / len(t) ** 0.5:.1f} | {st.mean(f):.0f}")
        se = st.stdev(diffs) / len(diffs) ** 0.5
        out.append(f"| {g:.2f} | " + " | ".join(cells) + f" | {st.mean(diffs):+.1f} ± {se:.1f} | {sum(d > 0 for d in diffs)} / {sum(d == 0 for d in diffs)} / {len(diffs)}"
                   f" | {st.mean(regret['gepa']):.1f} / {st.mean(regret['bo']):.1f} |")
    out += ["", "## Rollout accounting (mean per run)", "", "| arm | candidates | root | minibatch | full evals | of which advanced front | share of rollouts on non-advancing full evals |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        rs = [runs[(a, s)] for s in seeds]
        adv = st.mean(sum(c["advanced"] for c in r["cands"] if c["depth"] > 0) for r in rs)
        nadv = st.mean(sum(not c["advanced"] for c in r["cands"] if c["depth"] > 0) for r in rs)
        out.append(f"| {a} | {st.mean(len(r['cands']) for r in rs):.1f} | {rs[0]['n_train']} | {st.mean(r['mb_calls'] for r in rs):.0f} | "
                   f"{st.mean(r['full_calls'] for r in rs):.0f} | {adv:.1f} of {adv + nadv:.1f} | {st.mean(r['n_train'] * sum(not c['advanced'] for c in r['cands'] if c['depth'] > 0) / r['total_calls'] for r in rs):.0%} |")
    out += ["", "## Per seed, gap 0.05: shortest candidate", "", "| seed | root F1 | gepa | bo |", "|---|---|---|---|"]
    for s in seeds:
        r = {a: runs[(a, s)] for a in arms}; p = {a: shortest_at(r[a]["cands"], r[a]["root_f1"] - 0.05) for a in arms}
        out.append(f"| {s} | {r['gepa']['root_f1']:.3f} | " + " | ".join(f"{p[a]['tokens']:.0f} tok, F1 {p[a]['f1']:.3f} @{p[a]['found']}: `{p[a]['template'][:70]}`" for a in arms) + " |")
    return "\n".join(out) + "\n", runs, arms, seeds


def plot(runs, arms, seeds, path: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"gepa": "tab:orange", "bo": "tab:blue"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for (a, s), r in runs.items():
        if s not in seeds:
            continue
        cs = sorted(r["cands"], key=lambda c: c["tokens"]); front = []
        for c in cs:
            if not front or c["f1"] > front[-1]["f1"]:
                front.append(c)
        ax[0].plot([c["tokens"] for c in front], [c["f1"] for c in front], "o-", color=col[a], alpha=.4, label=a if s == seeds[0] else None)
        best, xs, ys = None, [], []
        for c in sorted(r["cands"], key=lambda c: c["found"]):
            if c["f1"] >= r["root_f1"] - 0.05 and (best is None or c["tokens"] < best):
                best = c["tokens"]
            xs.append(c["found"]); ys.append(best if best is not None else r["root_tokens"])
        ax[1].step(xs, ys, where="post", color=col[a], alpha=.4, label=a if s == seeds[0] else None)
    ax[0].set(xscale="log", xlabel="template tokens", ylabel="train F1", title="(tokens, train F1) fronts per run"); ax[0].legend()
    ax[1].set(xlabel="rollouts", ylabel="shortest candidate with F1 >= root - 0.05", yscale="log", title="anytime at gap 0.05 (schedule-free)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text, runs, arms, seeds = analyze(Path(a.path))
    out = Path(a.out or a.path); out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.md").write_text(text); plot(runs, arms, seeds, out / "analysis.png")
    print(text)
