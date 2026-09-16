"""HotpotQA (distractor) comparison at equal rollouts: GEPA vs BO vs MIPRO (0-shot), objective = train F1 (maximise).

    python -m experiments.live_compare.hotpot --mock --seeds 1 --rollouts 400 --n-train 40 --holdout 40 --out runs/hotpot_mock
    python -m experiments.live_compare.hotpot --seeds 3 --max-usd 4      # phase 1 (25% of seeds), then look
    python -m experiments.live_compare.hotpot --seeds 12 --max-usd 14    # continues: finished (arm, seed) pairs are skipped

Pausing: the loop is seed-major and skips (arm, seed) rows already in results.jsonl, so "run 25%, analyse, resume"
is `--seeds 3` followed by `--seeds 12` with the same --out. Every arm-seed has its own cache, so a re-run of an
interrupted arm-seed pays only for what it had not already done.

Arms (same root, train split, reflector model, minibatch, gate and rollout budget):
    gepa   Pareto pool on per-example F1, sample ∝ wins, 1 reflected child/round, minibatch gate (child > parent).
    gepa3  control: as gepa but 3 children/round, keep 1 at random - isolates "more proposals" from "surrogate picks".
    bo     no minibatch. EI over the candidates with value = the candidate's own full-train F1 (GEPA sampler until
           `warmup` candidates have been expanded), 3 reflected children, surrogate keeps 1 and it gets the full train set.
           The GP trains on full evaluations only (never on minibatch scores); targets PIT-transformed
           (`--bo-transform`), per-node noise = F1 SE. `--bo-gate minibatch` restores the old two-filter schedule.
    mipro  MIPROv2 (0-shot): N grounded instruction candidates proposed once from the root, then categorical TPE
           trials on minibatches (evaluations accumulate, so a candidate's coverage grows toward the full set),
           and every `full_every` trials the best-by-mean candidate gets the full train set. No feedback, no tree growth.

Per (arm, seed): runs/<out>/<arm>-s<seed>/{tree.json,events.jsonl,trace.jsonl,cache.jsonl} + a row in results.jsonl
(anytime curve of best train F1, held-out EM/F1 of root and best, proposal/gate counts).

Programs: `--program --modules selector,answerer` makes every node a two-module Program. Mutation is GEPA's
round-robin (module r mod M, one module rewritten per child, both arms). The bo arm then uses `AdditiveGPR` (one RBF
per module, summed) with per-node noise (F1 SE), ranks parents on the round's module component, and at the end
evaluates the additive model's "recombination" (per-module argmax of posterior mean) on train and held-out.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import random as _random
import statistics
import time
from pathlib import Path

from bpto import Budget, CompletionCache, Dataset, EventLog, ModelConfig, Origin, Stop, Tree, evaluate, run, select
from bpto.bo import EI, GPR, AdditiveGPR, BOSelector, HashEmbedder
from bpto.gepa import ReflectiveExpander, beats_parent, candidates, pareto_sample
from bpto.gepa.loop import minibatch_for
from bpto.mipro import CategoricalTPE, GroundedProposer, dataset_summary
from bpto.search import step
from tasks.hotpotqa import load_or_fetch, make_task
from tasks.hotpotqa.feedback import FEEDBACK, PASSED, feedback, passed, program_feedback, program_passed
from tasks.hotpotqa.program import make_program_task

from .compare import load_env

ARMS = ["gepa", "bo", "mipro"]


def f1_of(n) -> float | None:
    return n.evaluation.metrics.get("f1") if n.evaluation is not None else None


def is_full(node, tree) -> bool:
    ids = {ex.id for ex in tree.task.dataset}
    return node.evaluation is not None and ids.issubset(set(node.evaluation.dataset_ids))


def node_value(node, tree=None) -> float | None:
    """BO training target for a child: its F1 on whatever it was evaluated on (minibatch or full). Old gate only."""
    return f1_of(node)


def best_child_value(node, tree) -> float | None:
    vals = [f1_of(c) for c in tree.child_nodes(node) if c.evaluation is not None]
    return max(vals) if vals else None


def full_value(node, tree) -> float | None:
    """Own F1, only when measured on the full train set - minibatch scores never enter the GP."""
    return f1_of(node) if is_full(node, tree) else None


def best_full_child_value(node, tree) -> float | None:
    vals = [f1_of(c) for c in tree.child_nodes(node) if is_full(c, tree)]
    return max(vals) if vals else None


def best_full_child_se(node, tree) -> float | None:
    kids = [c for c in tree.child_nodes(node) if is_full(c, tree)]
    return f1_se(max(kids, key=f1_of)) if kids else None


def best_candidate(tree, ids):
    return max(candidates(tree, ids), key=lambda n: n.evaluation.metrics["f1"], default=None)


def f1_se(node, tree=None) -> float | None:
    """Per-node measurement noise for the surrogate: SE of the node's own F1 (minibatch-5 nodes ≈ .2, full-200 ≈ .03)."""
    ev = node.evaluation
    return ev.metrics_std.get("f1", 0.0) / max(1, ev.n) ** 0.5 if ev is not None else None


def best_child_se(node, tree) -> float | None:
    kids = [c for c in tree.child_nodes(node) if c.evaluation is not None]
    return f1_se(max(kids, key=f1_of)) if kids else None


def modules_of(args) -> list[str]:
    return [m for m in args.modules.split(",") if m] if args.program else []


def template_of(node, module: str | None) -> str:
    return node.prompt.modules[module].template if module else node.prompt.template


def make_bo(args, embedder, value, noise):
    multi = len(modules_of(args)) > 1
    return BOSelector(embedder, AdditiveGPR() if multi else GPR(), EI(), value=value, noise=None if args.no_noise else noise,
                      transform=None if args.bo_transform == "none" else args.bo_transform)


def schedule_for(arm, seed, args, embedder, client, screened: set[str], mipro_state: dict, bos: dict):
    modules = modules_of(args)
    multi = len(modules) > 1

    def schedule(tree, r):
        full = tree.task.dataset
        ids = {ex.id for ex in full}
        if not tree.evaluated_nodes():
            return [step(evaluate(dataset=full), select.root, name="root")]
        if arm == "mipro":
            return mipro_round(tree, r, full, ids)
        pure = arm == "bo" and args.bo_gate == "none"
        if arm == "bo" and "parent" not in bos:
            if pure:
                # parent target = the candidate's OWN full-train F1 (not its best child's: that rewards a parent for
                # having been expanded and locks EI onto it - ladder 0.50 vs 0.995, 2026-09-16)
                bos["parent"] = make_bo(args, embedder, full_value, f1_se)
                bos["child"] = make_bo(args, embedder, full_value, f1_se)
            else:
                bos["parent"] = make_bo(args, embedder, best_child_value, best_child_se)
                bos["child"] = make_bo(args, embedder, node_value, f1_se)
        parent_bo, child_bo = bos.get("parent"), bos.get("child")

        module = modules[r % len(modules)] if multi else None  # GEPA's round-robin, same r for every arm
        mb = minibatch_for(r, full, args.minibatch, seed)
        is_new = lambda n: n.origin.op == "reflect" and n.evaluation is None and n.id not in screened
        on_mb = lambda n: n.origin.op == "reflect" and n.evaluation is not None and not ids.issubset(set(n.evaluation.dataset_ids))
        sampler = pareto_sample(1, mode="weighted", seed=seed * 7919 + r, ids=ids, metric="f1")
        if multi:
            fb, ok = FEEDBACK, PASSED
        else:
            fb, ok = (program_feedback, program_passed) if args.program else (feedback, passed)
        mutator = ReflectiveExpander(fb, minibatch=args.minibatch, n=1 if arm == "gepa" else 3, seed=seed + r, passed=ok,
                                     module=module)

        if arm == "gepa":
            parent, to_minibatch = sampler, select.where(is_new)
        elif arm == "gepa3":
            parent = sampler

            def to_minibatch(tree):
                fresh = select.where(is_new)(tree)
                if len(fresh) <= 1:
                    return fresh
                keep = _random.Random(seed * 104_729 + r).choice(fresh)
                for n in fresh:
                    if n is not keep:
                        screened.add(n.id); n.origin.params["screened_out"] = True
                return [keep]
        else:
            async def parent(tree):
                trained = ([n for n in candidates(tree, ids) if n.state == "expanded"] if pure
                           else [n for n in tree if n.state == "expanded" and parent_bo.value(n, tree) is not None])
                if len(trained) < args.warmup:
                    return sampler(tree)
                ranked = await parent_bo.rank(tree, candidates(tree, ids), module=module if args.parent_rank == "module" else None)
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

        def tag(n):
            par = tree.nodes[n.parent_id]
            m = n.origin.params.get("module")
            n.origin.params["diff_ratio"] = round(difflib.SequenceMatcher(None, template_of(par, m), template_of(n, m)).ratio(), 3)

        def gate(tree):
            new = select.where(on_mb)(tree)
            ok = [n for n in new if beats_parent(tree, n)]
            for n in new:
                n.origin.params["accepted"] = n in ok
                tag(n)
            return ok

        if pure:
            # one full evaluation per round: the surrogate's pick among the fresh children, nothing else
            async def to_full(tree):
                keep = await to_minibatch(tree)
                for n in keep:
                    tag(n)
                    n.origin.params["accepted"] = None  # decided after the evaluation, by `record`
                return keep
            return [
                step(mutator, parent, name=f"r{r}/reflect"),
                step(evaluate(dataset=full), to_full, name=f"r{r}/full"),
            ]

        return [
            step(mutator, parent, name=f"r{r}/reflect"),
            step(evaluate(dataset=mb), to_minibatch, name=f"r{r}/minibatch"),
            step(evaluate(dataset=full), gate, name=f"r{r}/full"),
        ]

    def mipro_round(tree, r, full, ids):
        st = mipro_state
        if "tpe" not in st:  # round 1: propose once from the root
            summary = dataset_summary(full, k=3, seed=seed, max_chars=300)
            calls = max(1, -(-args.mipro_n // 4))
            proposer = GroundedProposer(summary, n=4, calls=calls, seed=seed)
            st["tpe"] = None
            return [step(proposer, select.root, name="propose")]
        cands = [n for n in tree if n.origin.op == "mipro_propose"]
        if st["tpe"] is None:
            st["tpe"] = CategoricalTPE(len(cands), seed=seed)
            st["cands"] = [n.id for n in cands]
            st["trials"] = 0
        tpe: CategoricalTPE = st["tpe"]
        if not cands:
            return []
        by_id = {ex.id: ex for ex in full}
        st["trials"] += 1
        steps = []
        if st["trials"] % args.full_every == 0:
            means = tpe.scores()
            pending = [c for c in sorted(means, key=means.get, reverse=True)
                       if not ids.issubset(set(tree.nodes[st["cands"][c]].evaluation.dataset_ids or []))]
            if pending:
                node = tree.nodes[st["cands"][pending[0]]]
                steps.append(step(evaluate(dataset=full), lambda t, n=node: [n], name=f"r{r}/full"))
                return steps
        c = tpe.suggest()
        node = tree.nodes[st["cands"][c]]
        mb = minibatch_for(r, full, args.mipro_minibatch, seed)
        seen = set(node.evaluation.dataset_ids) if node.evaluation is not None else set()
        grown = Dataset([by_id[i] for i in list(seen) + [ex.id for ex in mb if ex.id not in seen]])
        st["last"] = (c, [ex.id for ex in mb])
        steps.append(step(evaluate(dataset=grown), lambda t, n=node: [n], name=f"r{r}/trial"))
        return steps

    return schedule


def make_clients(args, cache, global_budget, data):
    k = 2 if args.program else 1  # the program makes two task-model calls per example (selector + answerer)
    extra = k * (args.n_train + args.holdout) if len(modules_of(args)) > 1 else 0  # recombination readout (bo arm)
    per_run = Budget(max_calls=args.rollouts + k * (args.n_train + 2 * args.holdout) + extra + 50, parent=global_budget)
    if args.mock:
        from tasks.hotpotqa.mock import _mock_client
        c = _mock_client(data, max_concurrency=8); c.budget = per_run
        return c, c
    from bpto import BedrockClient
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = BedrockClient(args.model, region=region, max_concurrency=args.concurrency, cache=cache, budget=per_run)
    reflect = BedrockClient(args.reflect_model, region=region, max_concurrency=4, cache=cache, budget=global_budget)
    return client, reflect


async def run_one(arm, seed, args, full_data, global_budget, embedder):
    out = Path(args.out) / f"{arm}-s{seed}"; out.mkdir(parents=True, exist_ok=True)
    train, rest = full_data.split(args.n_train / len(full_data), seed=seed)
    held = rest.sample(args.holdout, seed=seed)
    client, reflect_client = make_clients(args, None if args.mock else CompletionCache(out / "cache.jsonl"), global_budget, full_data)
    cfg = ModelConfig(max_tokens=args.max_tokens, temperature=args.eval_temperature)
    ek = dict(expander_client=reflect_client, expander_config=ModelConfig(max_tokens=2048, temperature=args.reflect_temperature))
    task = (make_program_task(client, train, config=cfg, modules=tuple(modules_of(args) or ["selector"]), **ek) if args.program
            else make_task(client, train, config=cfg, reasoning=not args.no_reasoning, **ek))
    tree = Tree(task)
    EventLog(out / "events.jsonl", tree)
    ids = {ex.id for ex in train}
    curve, screened, mipro_state, bos = [], set(), {}, {}
    t0 = time.time()

    def record(tree, r, st):
        if not (st.name.endswith("/full") or st.name == "root" or st.name.endswith("/trial")):
            return
        if st.name.endswith("/trial"):  # MIPRO: feed the TPE the trial's minibatch mean and log it
            c, mb_ids = mipro_state["last"]
            node = tree.nodes[mipro_state["cands"][c]]
            per = {x.example_id: x.metrics.get("f1", 0.0) for x in node.evaluation.per_example}
            mipro_state["tpe"].observe(c, sum(per.get(i, 0.0) for i in mb_ids) / max(1, len(mb_ids)))
        for n in tree:  # pure bo: "accepted" = beat the parent on the full set (bookkeeping only, no gate)
            if n.origin.params.get("accepted") is None and "accepted" in n.origin.params and n.evaluation is not None:
                n.origin.params["accepted"] = beats_parent(tree, n)
        bc = best_candidate(tree, ids)
        curve.append({"calls": client.usage.calls, "round": r, "step": st.name.split("/")[-1],
                      "best_f1": f1_of(bc) if bc else None, "best_em": bc.evaluation.metrics.get("em") if bc else None,
                      "best_sel_recall": bc.evaluation.metrics.get("sel_recall") if bc else None,
                      "best_id": bc.id if bc else None, "candidates": len(candidates(tree, ids)), "nodes": len(tree)})
    idle = {"calls": -1, "rounds": 0}

    def done(t):
        idle["rounds"] = idle["rounds"] + 1 if client.usage.calls == idle["calls"] else 0
        idle["calls"] = client.usage.calls
        return client.usage.calls >= args.rollouts or idle["rounds"] >= 25 * 3
    res = await run(tree, schedule_for(arm, seed, args, embedder, client, screened, mipro_state, bos),
                    stop=Stop(rounds=100_000, until=done), checkpoint=out / "tree.json", on_step=record)
    search_calls = client.usage.calls
    best = best_candidate(tree, ids)
    root_ev = await evaluate(dataset=held).score(tree, tree.root)
    best_ev = root_ev if best is tree.root else await evaluate(dataset=held).score(tree, best)
    cands = candidates(tree, ids)
    proposed = [n for n in tree if n.origin.op in ("reflect", "mipro_propose")]
    gated = [n for n in proposed if "accepted" in n.origin.params]
    modules = modules_of(args)
    recomb = None
    if arm == "bo" and len(modules) > 1 and "child" in bos and len(cands) > 1:
        # the additive model's best program: per-module argmax of posterior mean over the candidates
        await bos["child"].rank(tree, cands[:1])  # (re)fit on every evaluated node
        pick = await bos["child"].argmax_modules(tree, cands)
        prog = tree.root.prompt
        for m, n in pick.items():
            prog = prog.with_module(m, n.prompt.modules[m])
        same = next((n for n in cands if n.prompt == prog), None)
        node = same or tree.add_child(best, prog, Origin(op="recombine", params={"modules": {m: n.id for m, n in pick.items()}}))
        if same is None:
            await tree.apply(evaluate(dataset=train), lambda t: [node])
            tree.save(out / "tree.json")
        held_r = await evaluate(dataset=held).score(tree, node)
        recomb = {"id": node.id, "existing": same is not None, "parts": {m: n.id for m, n in pick.items()},
                  "train": node.evaluation.metrics, "held": held_r.metrics, "prompt": str(node.prompt)}
    row = {"arm": arm, "seed": seed, "rollouts": search_calls, "held_calls": client.usage.calls - search_calls,
           "stopped": res.stopped_because, "seconds": round(time.time() - t0, 1), "nodes": len(tree), "candidates": len(cands),
           "root_f1": f1_of(tree.root), "root_em": tree.root.evaluation.metrics["em"], "root_sel_recall": tree.root.evaluation.metrics.get("sel_recall"),
           "best_train_sel_recall": best.evaluation.metrics.get("sel_recall"),
           "best_id": best.id, "best_depth": best.depth, "best_train_f1": f1_of(best), "best_train_em": best.evaluation.metrics["em"],
           "root_held": root_ev.metrics, "best_held": best_ev.metrics, "best_prompt": str(best.prompt),
           "modules": modules or None,
           "accepted_by_module": {m: sum(n.origin.params["accepted"] for n in gated if n.origin.params.get("module") == m) for m in modules} or None,
           "best_lineage_modules": [n.origin.params.get("module") for n in [best, *tree.ancestors(best)] if n.origin.op == "reflect"] or None,
           "recombination": recomb, "bo_fits": tree.meta.get("bo_fits"),
           "proposed": len(proposed), "screened_out": len(screened), "gated": len(gated),
           "accepted": sum(n.origin.params["accepted"] for n in gated),
           "mean_diff_ratio": round(statistics.mean(n.origin.params["diff_ratio"] for n in gated), 3) if gated else None,
           "mipro_trials": mipro_state.get("trials"),
           "all_candidates": [{"id": n.id, "depth": n.depth, "f1": f1_of(n), "em": n.evaluation.metrics["em"]} for n in cands],
           "curve": curve, "reflect_calls": reflect_client.usage.calls, "spent_usd_so_far": round(global_budget.spent_usd, 4)}
    (out / "trace.jsonl").write_text("\n".join(json.dumps(x) for x in curve) + "\n")
    with open(Path(args.out) / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{arm} s{seed}: rollouts={search_calls} nodes={len(tree)} cands={len(cands)} "
          f"train f1 {row['root_f1']:.3f}->{row['best_train_f1']:.3f}  held f1 {root_ev.metrics['f1']:.3f}->{best_ev.metrics['f1']:.3f} "
          f"em {root_ev.metrics['em']:.3f}->{best_ev.metrics['em']:.3f}  accepted {row['accepted']}/{row['gated']} "
          f"${global_budget.spent_usd:.3f}, {row['seconds']}s", flush=True)
    return row


def summarize(path: Path):
    rows = [json.loads(l) for l in open(path / "results.jsonl")]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    arms = list(by); seeds = sorted({s for a in by.values() for s in a})
    lines = ["| seed | root train f1 | root held f1 | " + " | ".join(f"{a} train / held f1 (held em)" for a in arms) + " |",
             "|---|---|---|" + "---|" * len(arms)]
    for s in seeds:
        any_ = next(by[a][s] for a in arms if s in by[a])
        cells = [f"{by[a][s]['best_train_f1']:.3f} / {by[a][s]['best_held']['f1']:.3f} ({by[a][s]['best_held']['em']:.3f})" if s in by[a] else "—" for a in arms]
        lines.append(f"| {s} | {any_['root_f1']:.3f} | {any_['root_held']['f1']:.3f} | " + " | ".join(cells) + " |")
    means = []
    for a in arms:
        d_tr = [r["best_train_f1"] - r["root_f1"] for r in by[a].values()]
        d_he = [r["best_held"]["f1"] - r["root_held"]["f1"] for r in by[a].values()]
        se = lambda v: statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
        means.append(f"{a}: gain over root, train {statistics.mean(d_tr):+.3f} ± {se(d_tr):.3f}, held {statistics.mean(d_he):+.3f} ± {se(d_he):.3f} (n={len(d_tr)})")
    text = "\n".join(lines) + "\n\n" + "\n".join(means) + "\n"
    (path / "summary.md").write_text(text)
    print(text)


async def main(args):
    load_env()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    global_budget = Budget(max_usd=args.max_usd)
    if args.mock:
        embedder = HashEmbedder()
    else:
        from bpto.bo import BedrockEmbedder
        embedder = BedrockEmbedder(region=os.environ.get("AWS_REGION", "us-east-1"), budget=global_budget)
    full_data = load_or_fetch()
    if args.n_train + args.holdout > len(full_data):
        raise SystemExit(f"n_train + holdout > {len(full_data)} rows")
    done = set()
    if (out / "results.jsonl").exists():
        done = {(r["arm"], r["seed"]) for r in map(json.loads, open(out / "results.jsonl"))}
    try:
        for seed in range(args.seed_start, args.seeds):
            for arm in args.arms:
                if (arm, seed) in done:
                    continue
                await run_one(arm, seed, args, full_data, global_budget, embedder)
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
    ap.add_argument("--seeds", type=int, default=3, help="run seeds [seed_start, seeds)")
    ap.add_argument("--seed-start", type=int, default=0, help="first seed (parallel processes take disjoint ranges)")
    ap.add_argument("--rollouts", type=int, default=3000, help="task-model calls per arm-seed (search only)")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--minibatch", type=int, default=5, help="reflection traces and gate size (gepa/bo)")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--mipro-n", type=int, default=16, help="instruction candidates proposed once (16 x 200 train > 2,800 search rollouts)")
    ap.add_argument("--mipro-minibatch", type=int, default=10)
    ap.add_argument("--full-every", type=int, default=10, help="MIPRO: full-evaluate the best-by-mean candidate every k trials")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reflect-temperature", type=float, default=1.0)
    ap.add_argument("--eval-temperature", type=float, default=0.0, help="task-model temperature for rollouts (Nova default is 0.7; runs before 2026-09-16 used it)")
    ap.add_argument("--model", default="us.amazon.nova-micro-v1:0")
    ap.add_argument("--reflect-model", default="us.amazon.nova-lite-v1:0")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=4.0, help="hard cap on the whole experiment")
    ap.add_argument("--no-reasoning", action="store_true", help="answer-only schema (default: reasoning field before the answer)")
    ap.add_argument("--program", action="store_true", help="two-stage program (tasks.hotpotqa.program): search the paragraph selector; rollouts count both stages' calls")
    ap.add_argument("--modules", default="selector", help="with --program: comma-separated modules under search (selector,answerer -> Program nodes, round-robin)")
    ap.add_argument("--no-noise", action="store_true", help="bo: do not pass per-node F1 SE to the surrogate")
    ap.add_argument("--bo-gate", choices=["none", "minibatch"], default="none", help="bo: none = surrogate pick gets the full set directly (GP trains on full evals only); minibatch = old two-filter schedule")
    ap.add_argument("--bo-transform", choices=["pit", "none"], default="pit", help="bo: target transform for the surrogate")
    ap.add_argument("--parent-rank", choices=["full", "module"], default="full", help="bo, multi-module: EI on the full posterior or on the round's module component")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="runs/hotpot")
    asyncio.run(main(ap.parse_args()))
