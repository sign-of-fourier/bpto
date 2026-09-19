"""$0 replay: at every post-warmup round of a recorded gepa-ei run, refit the surrogate on that round's state
(embeddings are stored on the nodes, so no model calls) and compare which q parents each batch acquisition
would have chosen - independent top-q by EI, KrigingBeliever, local QEI, hosted QuantecarloQEI.

The question it answers before any spend: do the joint picks differ from top-q at all on this pool? If not, a live
q > 1 run measures speed only and says nothing about the acquisition.

    python -m experiments.live_compare.replay_batch runs/hotpot_gepaei2 --q 2 --pca 4 [--no-remote]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from bpto import Dataset, Tree
from bpto.bo import EI, GPR, BOSelector, KrigingBeliever, QEI, QuantecarloQEI
from bpto.gepa import candidates
from bpto.tree import Node

from .hotpot import child_gain_ses, child_gains, f1_of


class NoEmbedder:
    async def embed(self, texts):
        raise RuntimeError("replay must not embed: a node without a stored embedding was never ranked live")


class _Task:  # the only thing the selectors touch is `tree.task.dataset` (for the train ids)
    def __init__(self, ids):
        self.dataset = Dataset([_Ex(i) for i in ids])


class _Ex:
    def __init__(self, i):
        self.id = i


def tree_at(data: dict, upto: str | None, task) -> Tree:
    """The tree as it stood before the reflect child `upto` was created (all nodes if None)."""
    t = Tree.__new__(Tree)
    t.task, t.nodes, t.children, t.listeners, t.meta = task, {}, {}, [], {}
    cut = next(n["created_at"] for n in data["nodes"] if n["id"] == upto) if upto else None
    for rec in data["nodes"]:
        if cut is None or rec["created_at"] < cut:
            t._add(Node.model_validate(rec))
    t.root = t.nodes[data["root"]]
    return t


async def replay_seed(path: Path, q: int, pca: int, remote: bool, seed: int):
    data = json.loads((path / "tree.json").read_text())
    root = next(n for n in data["nodes"] if n["id"] == data["root"])
    ids = set(root["evaluation"]["dataset_ids"])
    task = _Task(sorted(ids))
    kids = sorted((n for n in data["nodes"] if n["origin"]["op"] == "reflect"), key=lambda n: n["created_at"])
    fits = [f for f in data["meta"]["bo_fits"] if not f["warmup"]]
    # round r's child is the first reflect node created after round r-1's child whose source is r's parent
    rows, ki = [], 0
    for f in fits:
        while ki < len(kids) and kids[ki]["origin"]["params"]["source"] != f["parent"]:
            ki += 1
        if ki >= len(kids):
            break
        child = kids[ki]; ki += 1
        tree = tree_at(data, child["id"], task)
        cands = candidates(tree, ids)
        if len(cands) < q + 1 or any(n.embedding is None for n in cands):
            continue
        methods = {"topk": None, "kb": KrigingBeliever(), "qei": QEI(seed=seed)}
        if remote:
            methods["quantecarlo"] = QuantecarloQEI(seed=seed)
        picks, info = {}, {}
        for name, batch in methods.items():
            sel = BOSelector(NoEmbedder(), GPR(), EI(), value=child_gains, noise=child_gain_ses, transform="pit",
                             pca=pca or None, batch=batch)
            chosen = await sel.top(q, among=lambda t: cands)(tree)
            picks[name] = [n.id for n in chosen]
            info[name] = sel.last_fit.get("batch", {})
        pred = sel.last_fit["pred"]
        order = sorted(cands, key=lambda n: -pred[n.id][2])
        rows.append({"round": f["round"], "n_cands": len(cands), "n_train": sel.last_fit["n_train"], "flat": sel.last_fit["flat"],
                     "ell": sel.last_fit["ell"], "ei_spread": float(pred[order[0].id][2] - pred[order[-1].id][2]),
                     "live_parent": f["parent"], "picks": picks,
                     "ranks": {m: [order.index(tree.nodes[i]) for i in p] for m, p in picks.items()},
                     "siblings": {m: len({tree.nodes[i].parent_id for i in p}) < len(p) for m, p in picks.items()},
                     "f1": {m: [round(f1_of(tree.nodes[i]), 3) for i in p] for m, p in picks.items()},
                     "report": {m: {k: v for k, v in d.items() if k not in ("ids", "top_k_ids", "method")} for m, d in info.items()}})
    return rows


def summarise(rows: list[dict], q: int) -> str:
    if not rows:
        return "no replayable rounds"
    ms = list(rows[0]["picks"])
    out = [f"{len(rows)} rounds replayed (pool > {q}); flat-fit rounds: {sum(r['flat'] for r in rows)}; "
           f"median ℓ {statistics.median(r['ell'] for r in rows):.2f}; median EI spread {statistics.median(r['ei_spread'] for r in rows):.4f}"]
    for m in ms:
        if m == "topk":
            continue
        diff = [set(r["picks"][m]) != set(r["picks"]["topk"]) for r in rows]
        sib = [r["siblings"][m] for r in rows]
        sib0 = [r["siblings"]["topk"] for r in rows]
        rk = [max(r["ranks"][m]) for r in rows]
        out.append(f"  {m:12s} differs from top-{q} in {sum(diff)}/{len(rows)} rounds; picks share a parent in {sum(sib)} "
                   f"(top-{q}: {sum(sib0)}); worst EI rank chosen: median {statistics.median(rk)}, max {max(rk)}")
    hit = [r["live_parent"] in r["picks"]["topk"] for r in rows]
    out.append(f"  live q=1 parent inside the replayed top-{q}: {sum(hit)}/{len(rows)}")
    return "\n".join(out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runs/<name> with gepa-ei-s*/tree.json (or the old gepaei-s*)")
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--pca", type=int, default=4)
    ap.add_argument("--no-remote", action="store_true", help="skip the hosted quantecarlo endpoint")
    ap.add_argument("--out", default=None, help="write per-round rows as jsonl")
    args = ap.parse_args()
    run = Path(args.run)
    allrows = []
    for d in sorted([*run.glob("gepa-ei-s*"), *run.glob("gepaei-s*")]):
        seed = int(d.name.split("-s")[1])
        rows = await replay_seed(d, args.q, args.pca, not args.no_remote, seed)
        for r in rows:
            r["seed"] = seed
        allrows += rows
        print(f"{d.name}: {summarise(rows, args.q)}")
    print(f"all seeds: {summarise(allrows, args.q)}")
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in allrows) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
