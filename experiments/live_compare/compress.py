"""Compression comparison at equal rollouts: GEPA vs BO, objective = template tokens (minimise) subject to an
F1 floor that starts loose and tightens with rollouts spent (constraint annealing / continuation).

    python -m experiments.live_compare.compress --mock --seeds 1 --rollouts 150 --out runs/compress_mock   # plumbing
    python -m experiments.live_compare.compress --seeds 3 --rollouts 600 --max-usd 0.5

Floor schedule (same for both arms, keyed on the task client's call count, not tree depth):
    floor(calls) = min(root_f1 - start_gap + step * (calls // every), root_f1 - end_gap)
Infeasible nodes stay in the tree and the Pareto pool and may be parents; only the "best" readout (shortest
feasible candidate) requires feasibility. Everything evaluated is re-scored whenever the floor moves (free).

Per (arm, seed): runs/<out>/<arm>-s<seed>/{tree.json,events.jsonl,trace.jsonl} + a row in results.jsonl
(anytime curve of best-feasible tokens, final (tokens, f1) front, held-out metrics of root and best).
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import statistics
import time
from pathlib import Path

from bpto import (Budget, CompletionCache, ConstrainedObjective, EventLog, LinearObjective, ModelConfig, Stop, Tree,
                  evaluate, pareto_front, run, select)
from bpto.bo import EI, GPR, BOSelector, HashEmbedder
from bpto.gepa import ReflectiveExpander, candidates, pareto_sample
from bpto.gepa.loop import minibatch_for
from bpto.gepa.select import example_scores
from bpto.scoring import ObjectiveContext
from bpto.search import step
from tasks.compression import generate_dataset, make_task
from tasks.compression.feedback import COMPRESS_REFLECT_PROMPT, feedback, passed

from .compare import load_env

ARMS = ["gepa", "bo"]


class Floor:
    """Mutable F1 floor read by the objective's bound; the harness advances it from rollouts spent."""

    def __init__(self, start_gap: float, end_gap: float, step: float, every: int):
        self.start_gap, self.end_gap, self.step, self.every = start_gap, end_gap, step, every
        self.root_f1: float | None = None
        self.value = 0.0

    def update(self, calls: int) -> float:
        if self.root_f1 is not None:
            self.value = min(self.root_f1 - self.start_gap + self.step * (calls // self.every), self.root_f1 - self.end_gap)
        return self.value


def make_objective(floor: Floor):
    return ConstrainedObjective(LinearObjective(template_tokens=-1.0), metric="f1", bound=lambda ctx: floor.value, sense=">=")


def rescore(tree: Tree) -> None:
    """Re-apply the (moved) objective to every stored evaluation. No model calls."""
    n = len(tree.evaluated_nodes())
    for node in tree.evaluated_nodes():
        ev = node.evaluation
        ev.score, ev.feasible = tree.task.objective(ev.metrics, ObjectiveContext(depth=node.depth, n_evaluated=n))


def tokens(n) -> float:
    return n.evaluation.metrics.get("template_tokens", 0.0)


def gain_of(root_tokens: float):
    """Value of a node: tokens saved vs the root if it meets the floor, else 0 (an infeasible prompt saved nothing)."""
    def gain(node, tree=None) -> float | None:
        if node.evaluation is None:
            return None
        return max(0.0, root_tokens - tokens(node)) if node.evaluation.feasible else 0.0
    return gain


def best_children_gain(gain):
    def value(node, tree) -> float | None:
        vals = [gain(c) for c in tree.child_nodes(node) if c.evaluation is not None]
        return max(vals) if vals else None
    return value


def best_feasible(tree, ids):
    feas = [n for n in candidates(tree, ids) if n.evaluation.feasible]
    return min(feas, key=tokens, default=None)


def accepted(tree, child, floor: float) -> bool:
    """Minibatch gate under the constrained order: feasibility (mb f1 >= floor) first, then shorter (both feasible)
    or more accurate (both infeasible)."""
    parent = tree.nodes.get(child.parent_id) if child.parent_id else None
    if child.evaluation is None or parent is None or parent.evaluation is None:
        return False
    ids = child.evaluation.dataset_ids
    pf1 = example_scores(tree, parent, "f1")
    if not all(i in pf1 for i in ids):
        return False
    p_f1 = sum(pf1[i] for i in ids) / max(1, len(ids))
    c_f1 = child.evaluation.metrics.get("f1", 0.0)
    cf, pf = c_f1 >= floor, p_f1 >= floor
    if cf and pf:
        return tokens(child) < tokens(parent)
    if cf != pf:
        return cf
    return c_f1 > p_f1


def schedule_for(arm, seed, args, floor: Floor, embedder, client, trace: list, screened: set[str]):
    """gepa: Pareto pool on per-example F1, sample ∝ wins, 1 parent, 1 child.
    bo: EI over candidates with value = best child's gain (GEPA sampler until `warmup`), 3 children, surrogate keeps 1."""
    gain = None
    parent_bo = child_bo = None

    def schedule(tree, r):
        nonlocal gain, parent_bo, child_bo
        full = tree.task.dataset
        ids = {ex.id for ex in full}
        if not tree.evaluated_nodes():
            return [step(evaluate(dataset=full), select.root, name="root")]
        if floor.root_f1 is None:
            floor.root_f1 = tree.root.evaluation.metrics["f1"]
            gain = gain_of(tokens(tree.root))
            if arm == "bo":
                parent_bo = BOSelector(embedder, GPR(), EI(), value=best_children_gain(gain))
                child_bo = BOSelector(embedder, GPR(), EI(), value=gain)
        before = floor.value
        floor.update(client.usage.calls)
        rescore(tree)  # stale scores/feasibility after the floor moved; also the first pass after the root eval
        if floor.value != before:
            bf = best_feasible(tree, ids)
            trace.append({"round": r, "event": "floor", "calls": client.usage.calls, "floor": round(floor.value, 4),
                          "best_tokens": tokens(bf) if bf else None, "best_f1": bf.evaluation.metrics["f1"] if bf else None})

        mb = minibatch_for(r, full, args.minibatch, seed)
        is_new = lambda n: n.origin.op == "reflect" and n.evaluation is None and n.id not in screened
        on_mb = lambda n: n.origin.op == "reflect" and n.evaluation is not None and not ids.issubset(set(n.evaluation.dataset_ids))
        sampler = pareto_sample(1, mode="weighted", seed=seed * 7919 + r, ids=ids, metric="f1")
        mutator = ReflectiveExpander(feedback, minibatch=args.minibatch, n=1 if arm == "gepa" else 3, seed=seed + r,
                                     meta_prompt=COMPRESS_REFLECT_PROMPT, passed=passed)

        if arm == "gepa":
            parent, to_minibatch = sampler, select.where(is_new)
        else:
            async def parent(tree):
                trained = [n for n in tree if n.state == "expanded" and parent_bo.value(n, tree) is not None]
                if len(trained) < args.warmup:
                    return sampler(tree)
                ranked = await parent_bo.rank(tree, candidates(tree, ids))
                tree.meta.setdefault("bo_fits", []).append({"round": r, **{k: v for k, v in parent_bo.last_fit.items() if k != "pred"}})
                return [n for _, n in ranked[:1]]

            async def to_minibatch(tree):
                fresh = select.where(is_new)(tree)
                if len(fresh) <= 1:
                    return fresh
                ranked = await child_bo.rank(tree, fresh)
                for _, n in ranked[1:]:
                    screened.add(n.id); n.origin.params["screened_out"] = True
                return [ranked[0][1]]

        def gate(tree):
            new = select.where(on_mb)(tree)
            ok = [n for n in new if accepted(tree, n, floor.value)]
            for n in new:
                n.origin.params["accepted"] = n in ok
                par = tree.nodes[n.parent_id]
                n.origin.params["diff_ratio"] = round(difflib.SequenceMatcher(None, par.prompt.template, n.prompt.template).ratio(), 3)
            return ok

        return [
            step(mutator, parent, name=f"r{r}/reflect"),
            step(evaluate(dataset=mb), to_minibatch, name=f"r{r}/minibatch"),
            step(evaluate(dataset=full), gate, name=f"r{r}/full"),
        ]
    return schedule


def make_clients(args, cache, global_budget):
    if args.mock:
        from tasks.compression.run import _mock_client
        c = _mock_client(); c.budget = Budget(max_calls=args.rollouts + args.n_train + 2 * args.holdout, parent=global_budget)
        return c, c
    from bpto import BedrockClient
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = BedrockClient(args.model, region=region, max_concurrency=args.concurrency, cache=cache,
                           budget=Budget(max_calls=args.rollouts + args.n_train + 2 * args.holdout, parent=global_budget))
    reflect = BedrockClient(args.reflect_model, region=region, max_concurrency=4, cache=cache, budget=global_budget)
    return client, reflect


async def run_one(arm, seed, args, full_data, global_budget, cache, embedder):
    out = Path(args.out) / f"{arm}-s{seed}"; out.mkdir(parents=True, exist_ok=True)
    train, rest = full_data.split(args.n_train / len(full_data), seed=seed)
    held = rest.sample(args.holdout, seed=seed)
    # cache per arm-seed: a shared file would hand the second arm its root evaluation for free
    client, reflect_client = make_clients(args, None if args.mock else CompletionCache(out / "cache.jsonl"), global_budget)
    floor = Floor(args.start_gap, args.end_gap, args.step, args.every)
    task = make_task(client, train, objective=make_objective(floor), config=ModelConfig(max_tokens=512, temperature=args.eval_temperature),
                     expander_client=reflect_client, expander_config=ModelConfig(max_tokens=2048, temperature=args.reflect_temperature))
    tree = Tree(task)
    EventLog(out / "events.jsonl", tree)
    ids = {ex.id for ex in train}
    trace, curve, screened = [], [], set()
    t0 = time.time()

    def record(tree, r, st):
        if not st.name.endswith("/full") and st.name != "root":
            return
        bf = best_feasible(tree, ids)
        cands = candidates(tree, ids)
        curve.append({"calls": client.usage.calls, "round": r, "floor": round(floor.value, 4),
                      "best_tokens": tokens(bf) if bf else None, "best_f1": bf.evaluation.metrics["f1"] if bf else None,
                      "feasible": sum(n.evaluation.feasible for n in cands), "infeasible": sum(not n.evaluation.feasible for n in cands),
                      "nodes": len(tree)})
    idle = {"calls": -1, "rounds": 0}

    def done(t):  # rollouts spent, or the mutator has produced nothing for 25 rounds (no calls advancing)
        idle["rounds"] = idle["rounds"] + 1 if client.usage.calls == idle["calls"] else 0
        idle["calls"] = client.usage.calls
        return client.usage.calls >= args.rollouts or idle["rounds"] >= 25 * 3
    stop = Stop(rounds=10_000, until=done)
    res = await run(tree, schedule_for(arm, seed, args, floor, embedder, client, trace, screened), stop=stop,
                    checkpoint=out / "tree.json", on_step=record)
    rescore(tree)
    search_calls = client.usage.calls  # held-out passes below are not rollouts
    best = best_feasible(tree, ids)
    root_ev = await evaluate(dataset=held).score(tree, tree.root)
    best_ev = await evaluate(dataset=held).score(tree, best)
    cands = candidates(tree, ids)
    pts = {n.id: {"template_tokens": tokens(n), "f1": n.evaluation.metrics["f1"]} for n in cands}
    front = sorted(({"id": i, **pts[i]} for i in pareto_front(pts, {"template_tokens": False, "f1": True})), key=lambda p: p["template_tokens"])
    reflect_nodes = [n for n in tree if n.origin.op == "reflect"]
    gated = [n for n in reflect_nodes if "accepted" in n.origin.params]
    row = {"arm": arm, "seed": seed, "rollouts": search_calls, "held_calls": client.usage.calls - search_calls, "stopped": res.stopped_because,
           "seconds": round(time.time() - t0, 1), "nodes": len(tree), "candidates": len(cands),
           "final_floor": round(floor.value, 4), "root_f1": tree.root.evaluation.metrics["f1"], "root_tokens": tokens(tree.root),
           "best_id": best.id, "best_depth": best.depth, "best_tokens": tokens(best), "best_train_f1": best.evaluation.metrics["f1"],
           "root_held": root_ev.metrics, "best_held": best_ev.metrics, "best_prompt": best.prompt.template,
           "proposed": len(reflect_nodes), "screened_out": len(screened), "gated": len(gated),
           "accepted": sum(n.origin.params["accepted"] for n in gated),
           "mean_diff_ratio": round(statistics.mean(n.origin.params["diff_ratio"] for n in gated), 3) if gated else None,
           "front": front, "curve": curve, "floor_trace": trace, "reflect_calls": reflect_client.usage.calls,
           "spent_usd_so_far": round(global_budget.spent_usd, 4)}
    (out / "trace.jsonl").write_text("\n".join(json.dumps(x) for x in trace + curve) + "\n")
    with open(Path(args.out) / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{arm} s{seed}: rollouts={search_calls} nodes={len(tree)} cands={len(cands)} floor={floor.value:.3f} "
          f"tokens {tokens(tree.root):.0f}->{tokens(best):.0f}  train f1 {row['root_f1']:.3f}->{row['best_train_f1']:.3f}  "
          f"held f1 {root_ev.metrics['f1']:.3f}->{best_ev.metrics['f1']:.3f}  accepted {row['accepted']}/{row['gated']} "
          f"${global_budget.spent_usd:.3f}, {row['seconds']}s", flush=True)
    return row


def summarize(path: Path):
    rows = [json.loads(l) for l in open(path / "results.jsonl")]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    arms = list(by); seeds = sorted({s for a in by.values() for s in a})
    lines = ["| seed | root tok | root held f1 | " + " | ".join(f"{a} tok / held f1 / train f1" for a in arms) + " | floor |",
             "|---|---|---|" + "---|" * (len(arms) + 1)]
    for s in seeds:
        any_ = next(by[a][s] for a in arms if s in by[a])
        cells = [f"{by[a][s]['best_tokens']:.0f} / {by[a][s]['best_held']['f1']:.3f} / {by[a][s]['best_train_f1']:.3f}" if s in by[a] else "—" for a in arms]
        lines.append(f"| {s} | {any_['root_tokens']:.0f} | {any_['root_held']['f1']:.3f} | " + " | ".join(cells) + f" | {any_['final_floor']:.3f} |")
    means = []
    for a in arms:
        v = [r["best_tokens"] for r in by[a].values()]; h = [r["best_held"]["f1"] for r in by[a].values()]
        means.append(f"{a}: tokens {statistics.mean(v):.1f} ± {statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0:.1f}, "
                     f"held f1 {statistics.mean(h):.3f} (n={len(v)})")
    paired = [by["gepa"][s]["best_tokens"] - by["bo"][s]["best_tokens"] for s in seeds if s in by.get("bo", {}) and s in by.get("gepa", {})]
    if paired:
        means.append(f"paired tokens gepa-bo (positive = BO shorter): {statistics.mean(paired):+.1f} ± "
                     f"{statistics.stdev(paired) / len(paired) ** 0.5 if len(paired) > 1 else 0:.1f}, BO shorter {sum(d > 0 for d in paired)}/{len(paired)}")
    text = "\n".join(lines) + "\n\n" + "\n".join(means) + "\n"
    (path / "summary.md").write_text(text)
    print(text)


async def main(args):
    load_env()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    global_budget = Budget(max_usd=args.max_usd)
    cache = None
    if args.mock:
        embedder = HashEmbedder()
    else:
        from bpto.bo import BedrockEmbedder
        embedder = BedrockEmbedder(region=os.environ.get("AWS_REGION", "us-east-1"), budget=global_budget)
    full_data = generate_dataset(args.n_train + args.holdout + 40, seed=args.data_seed)
    done = set()
    if (out / "results.jsonl").exists():
        done = {(r["arm"], r["seed"]) for r in map(json.loads, open(out / "results.jsonl"))}
    try:
        for seed in range(args.seeds):
            for arm in args.arms:
                if (arm, seed) in done:
                    continue
                await run_one(arm, seed, args, full_data, global_budget, cache, embedder)
    except Exception as e:
        print(f"stopped: {type(e).__name__}: {e}")
        if args.mock:
            raise
    print(f"total spent: ${global_budget.spent_usd:.4f}  ({global_budget.spent})")
    if (out / "results.jsonl").exists():
        summarize(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=600, help="task-model calls per arm-seed (search only)")
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--start-gap", type=float, default=0.15, help="initial floor = root_f1 - start_gap")
    ap.add_argument("--end-gap", type=float, default=0.05, help="final floor = root_f1 - end_gap")
    ap.add_argument("--step", type=float, default=0.02, help="floor increase per `every` rollouts")
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--reflect-temperature", type=float, default=1.0)
    ap.add_argument("--eval-temperature", type=float, default=0.0, help="task-model temperature for rollouts (Nova default is 0.7; runs before 2026-09-16 used it)")
    ap.add_argument("--model", default="us.amazon.nova-micro-v1:0")
    ap.add_argument("--reflect-model", default="us.amazon.nova-lite-v1:0")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0.5, help="hard cap on the whole experiment")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="runs/compress")
    asyncio.run(main(ap.parse_args()))
