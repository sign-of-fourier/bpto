"""Live two-arm comparison at equal rollouts: faithful GEPA vs BO (ancestor-attributed parent choice +
surrogate child pre-screen). Everything else identical: task, split, reflective mutator, minibatch gate,
rollout budget, held-out set. One shared dollar cap shuts the whole experiment down.

    python -m experiments.live_compare.compare --task ifbench --seeds 5 --rollouts 1200 --max-usd 0.75
    python -m experiments.live_compare.compare --task ifbench --seeds 1 --rollouts 200 --out runs/compare_smoke   # plumbing

Per (arm, seed): runs/<out>/<arm>-s<seed>/{tree.json,events.jsonl} + a row in results.jsonl with the
train curve, the best candidate and its held-out metrics. Rollouts = unique model calls on the task client.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

from bpto import (Budget, CompletionCache, EventLog, ModelConfig, Stop, Tree, evaluate, run, select)
from bpto.bo import EI, GPR, BedrockEmbedder, BOSelector
from bpto.gepa import ReflectiveExpander, candidates, pareto_sample
from bpto.gepa.loop import _accepted, minibatch_for
from bpto.search import step
from bpto.value import best_child, own_score

ARMS = ["gepa", "bo"]


def load_env():
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def make_task(args, train, client, expander_client):
    if args.task == "ifbench":
        from tasks.ifbench import make_task as mk
        from tasks.ifbench.feedback import feedback
        task = mk(client, train, config=ModelConfig(max_tokens=1024), expander_client=expander_client,
                  expander_config=ModelConfig(max_tokens=2048))
        return task, feedback, "strict"
    from tasks.compression import make_task as mk
    from tasks.compression.feedback import feedback
    task = mk(client, train, config=ModelConfig(max_tokens=512), expander_client=expander_client,
              expander_config=ModelConfig(max_tokens=2048))
    return task, feedback, "f1"


def load_data(args):
    if args.task == "ifbench":
        from tasks.ifbench import load_or_fetch
        return load_or_fetch()
    from tasks.compression import generate_dataset
    return generate_dataset(args.n_train + args.holdout + 40, seed=args.data_seed)


def schedule_for(arm: str, feedback, seed: int, minibatch: int, warmup: int, embedder):
    """gepa: Pareto pool, sample ∝ wins, 1 parent, 1 child.  bo: EI over all candidates with value=best child
    (GEPA sampler until `warmup` expansions), 3 children, surrogate keeps 1 for the minibatch."""
    parent_bo = BOSelector(embedder, GPR(), EI(), value=best_child) if arm == "bo" else None
    child_bo = BOSelector(embedder, GPR(), EI(), value=own_score) if arm == "bo" else None
    screened: set[str] = set()

    def schedule(tree, r):
        full = tree.task.dataset
        ids = {ex.id for ex in full}
        if not tree.evaluated_nodes():
            return [step(evaluate(dataset=full), select.root, name="root")]
        mb = minibatch_for(r, full, minibatch, seed)
        is_new = lambda n: n.origin.op == "reflect" and n.evaluation is None and n.id not in screened
        on_mb = lambda n: n.origin.op == "reflect" and n.evaluation is not None and not ids.issubset(set(n.evaluation.dataset_ids))
        sampler = pareto_sample(1, mode="weighted", seed=seed * 7919 + r, ids=ids)

        if arm == "gepa":
            parent, to_minibatch, n_children = sampler, select.where(is_new), 1
        else:
            async def parent(tree):
                expanded = [n for n in tree if n.state == "expanded" and best_child(n, tree) is not None]
                if len(expanded) < warmup:
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
            n_children = 3
        return [
            step(ReflectiveExpander(feedback, minibatch=minibatch, n=n_children, seed=seed + r), parent, name=f"r{r}/reflect"),
            step(evaluate(dataset=mb), to_minibatch, name=f"r{r}/minibatch"),
            step(evaluate(dataset=full), lambda t: _accepted(t, select.where(on_mb)(t)), name=f"r{r}/full"),
        ]
    return schedule


async def run_one(arm: str, seed: int, args, full_data, global_budget, cache, embedder):
    from bpto import BedrockClient
    out = Path(args.out) / f"{arm}-s{seed}"; out.mkdir(parents=True, exist_ok=True)
    train, rest = full_data.split(args.n_train / len(full_data), seed=seed)
    held = rest.sample(args.holdout, seed=seed)

    region = os.environ.get("AWS_REGION", "us-east-1")
    # per-arm-seed rollout cap on the task client; the shared dollar cap is checked by every client
    client = BedrockClient(args.model, region=region, max_concurrency=args.concurrency, cache=cache,
                           budget=Budget(max_calls=args.rollouts + args.n_train + 2 * args.holdout, parent=global_budget))  # slack: last step overshoot + held-out
    reflect_client = BedrockClient(args.reflect_model, region=region, max_concurrency=4, cache=cache, budget=global_budget)
    task, feedback, key = make_task(args, train, client, reflect_client)

    tree = Tree(task)
    EventLog(out / "events.jsonl", tree)
    curve = []
    t0 = time.time()

    def record(tree, r, st):
        best = max(candidates(tree), key=lambda n: n.score, default=None)
        curve.append({"calls": client.usage.calls, "round": r, "step": st.name, "best_train": best.score if best else None,
                      "pool": len(candidates(tree)), "nodes": len(tree)})
    stop = Stop(rounds=10_000, until=lambda t: client.usage.calls >= args.rollouts)
    res = await run(tree, schedule_for(arm, feedback, seed, args.minibatch, args.warmup, embedder), stop=stop,
                    checkpoint=out / "tree.json", on_step=record)
    best = max(candidates(tree), key=lambda n: n.score)
    root_ev = await evaluate(dataset=held).score(tree, tree.root)
    best_ev = await evaluate(dataset=held).score(tree, best)
    row = {"arm": arm, "seed": seed, "task": args.task, "rollouts": client.usage.calls, "stopped": res.stopped_because,
           "seconds": round(time.time() - t0, 1), "nodes": len(tree), "pool": len(candidates(tree)),
           "best_id": best.id, "best_depth": best.depth, "best_train": best.score,
           "root_train": tree.root.score, "root_held": root_ev.metrics, "best_held": best_ev.metrics,
           "best_prompt": best.prompt.template, "curve": curve, "reflect_calls": reflect_client.usage.calls,
           "spent_usd_so_far": round(global_budget.spent_usd, 4)}
    with open(Path(args.out) / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{arm} s{seed}: rollouts={client.usage.calls} pool={row['pool']} nodes={len(tree)} "
          f"train {tree.root.score:.3f}->{best.score:.3f}  held {key}: {root_ev.metrics[key]:.3f}->{best_ev.metrics[key]:.3f}"
          f"  ${global_budget.spent_usd:.3f} spent, {row['seconds']}s", flush=True)
    return row


def summarize(path: Path, key: str):
    rows = [json.loads(l) for l in open(path / "results.jsonl")]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    seeds = sorted(set(s for a in by.values() for s in a))
    lines = [f"| seed | " + " | ".join(f"{a} held {key}" for a in by) + " | root held | " + " | ".join(f"{a} train" for a in by) + " |",
             "|---|" + "---|" * (2 * len(by) + 1)]
    for s in seeds:
        held = [f"{by[a][s]['best_held'][key]:.3f}" if s in by[a] else "—" for a in by]
        train = [f"{by[a][s]['best_train']:.3f}" if s in by[a] else "—" for a in by]
        root = next((f"{by[a][s]['root_held'][key]:.3f}" for a in by if s in by[a]), "—")
        lines.append(f"| {s} | " + " | ".join(held) + f" | {root} | " + " | ".join(train) + " |")
    means = []
    for a in by:
        v = [r["best_held"][key] for r in by[a].values()]
        means.append(f"{a}: {statistics.mean(v):.3f} ± {statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0:.3f} (n={len(v)})")
    paired = [by["bo"][s]["best_held"][key] - by["gepa"][s]["best_held"][key] for s in seeds if s in by.get("bo", {}) and s in by.get("gepa", {})]
    if paired:
        means.append(f"paired bo-gepa: {statistics.mean(paired):+.3f} ± {statistics.stdev(paired) / len(paired) ** 0.5 if len(paired) > 1 else 0:.3f}, wins {sum(d > 0 for d in paired)}/{len(paired)}")
    text = "\n".join(lines) + "\n\n" + "\n".join(means) + "\n"
    (path / "summary.md").write_text(text)
    print(text)


async def main(args):
    load_env()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    global_budget = Budget(max_usd=args.max_usd)
    cache = CompletionCache(out / "cache.jsonl")
    embedder = BedrockEmbedder(region=os.environ.get("AWS_REGION", "us-east-1"), budget=global_budget)
    full_data = load_data(args)
    key = "strict" if args.task == "ifbench" else "f1"
    done = set()
    if (out / "results.jsonl").exists():
        done = {(r["arm"], r["seed"]) for r in map(json.loads, open(out / "results.jsonl"))}
    try:
        for seed in range(args.seeds):
            for arm in args.arms:
                if (arm, seed) in done:
                    continue
                await run_one(arm, seed, args, full_data, global_budget, cache, embedder)
    except Exception as e:  # BudgetExceeded from a held-out pass, or anything else: keep what we have
        print(f"stopped: {type(e).__name__}: {e}")
    print(f"total spent: ${global_budget.spent_usd:.4f}  ({global_budget.spent})")
    if (out / "results.jsonl").exists():
        summarize(out, key)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ifbench", "compression"], default="ifbench")
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=1200, help="task-model calls per arm-seed (search only)")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--model", default="us.amazon.nova-micro-v1:0")
    ap.add_argument("--reflect-model", default="us.amazon.nova-lite-v1:0")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0.75, help="hard cap on the whole experiment")
    ap.add_argument("--out", default="runs/compare")
    asyncio.run(main(ap.parse_args()))
