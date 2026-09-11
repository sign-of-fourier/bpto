"""Synthetic ladder: does a surrogate beat GEPA's Pareto-weighted sampler, and where does it plug in?

$0, offline, runs the *real* bpto stack (Tree, evaluate, gepa loop, BOSelector) against a planted
landscape served by a MockClient:

  * K hidden "skills". A prompt has skill j iff it contains keyword_j. The vocabulary also holds
    distractor words. Each example needs a subset of skills; its score is the fraction present,
    with deterministic per-(prompt, example) noise so no two evaluations agree perfectly.
  * The mutation operator is the same for every arm: a ReflectiveExpander whose mock "reflection"
    reads the feedback ("missing: s3") and adds a missing skill with probability p_informed,
    otherwise a random vocabulary word; sometimes drops a word. Every arm gets identical children
    for identical (parent, minibatch) - only *selection* differs.

Arms (parent selection | child pre-screen):
  gepa-weighted   Pareto pool, sample ∝ examples won (GEPA)        | none
  gepa-uniform    Pareto pool, uniform                              | none
  gepa-best       argmax mean (greedy, no stochasticity)            | none
  gepa-all        every candidate, uniform (no Pareto)              | none
  bo-replace      BOSelector over all candidates (value=best child) | none
  bo-in-pareto    BOSelector restricted to the Pareto pool          | none
  gepa+screen     GEPA sampler                                      | propose 3, surrogate keeps 1
  bo+screen       BOSelector over all candidates                    | propose 3, surrogate keeps 1

All arms: same minibatch gate, same rollout budget (unique model calls). BO arms fall back to the GEPA
sampler until `warmup` nodes have been expanded (nothing to fit before that).

    python -m experiments.synthetic_ladder.ladder --seeds 30 --budget 600 --out experiments/<date>-synthetic-ladder
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

from bpto import (Budget, BudgetExceeded, Dataset, LinearObjective, MockClient, Stop, Task, Tree, evaluate, run,
                  select)
from bpto.bo import EI, GPR, BOSelector, HashEmbedder
from bpto.gepa import ReflectiveExpander, candidates, pareto_pool, pareto_sample
from bpto.gepa.loop import _accepted, minibatch_for
from bpto.ops import Variants
from bpto.search import step
from bpto.value import best_child, own_score

K = 6
SKILLS = [f"skill{j}" for j in range(K)]
DISTRACTORS = [f"word{j}" for j in range(24)]
TRAPS = ["trap0", "trap1", "trap2"]  # --trap: a trap word lifts every example to >= 0.5 but caps it at 0.6
VOCAB = SKILLS + DISTRACTORS
TRAP_CAP = (0.5, 0.6)


def _skill_frac(words: set[str], need: list[str]) -> float:
    frac = len(set(need) & words) / len(need)
    if any(t in words for t in TRAPS):
        frac = min(max(frac, TRAP_CAP[0]), TRAP_CAP[1])
    return frac


def _h(*parts) -> int:
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


NEEDS: dict[str, list[str]] = {}  # "task <seed>-<i>" -> required skills (the mock model looks them up)


def make_dataset(n: int, seed: int) -> Dataset:
    rnd = random.Random(seed)
    recs = []
    for i in range(n):
        need = sorted(rnd.sample(SKILLS, rnd.choice([1, 2, 2, 3])))
        key = f"task {seed}-{i}"
        NEEDS[key] = need
        recs.append({"id": f"e{i}", "inputs": {"x": key}, "answer": need})
    return Dataset.from_records(recs)


def make_client(seed: int, noise: float, p_informed: float, budget: int, trap: bool = False) -> MockClient:
    vocab = VOCAB + (TRAPS * 3 if trap else [])  # traps are "attractive": over-represented among random additions

    def handler(prompt, cfg, schema):
        if schema is Variants:
            n = int(re.search(r"write (\d+) improved", prompt).group(1))
            base = re.search(r"<prompt>\n(.*?)\n</prompt>", prompt, re.S).group(1)
            missing = sorted(set(re.findall(r"missing: ([\w, ]+)", prompt)) and
                             {w for m in re.findall(r"missing: ([\w, ]+)", prompt) for w in m.replace(",", " ").split()})
            words = base.replace("{x}", "").split()
            out = []
            for i in range(n):
                rnd = random.Random(_h(seed, prompt, i))
                w = list(words)
                if missing and rnd.random() < p_informed:
                    w.append(rnd.choice(sorted(missing)))
                else:
                    w.append(rnd.choice(vocab))
                if len(w) > 2 and rnd.random() < 0.3:
                    w.pop(rnd.randrange(len(w)))
                out.append(" ".join(dict.fromkeys(w)) + " {x}")
            return Variants(prompts=out)
        # evaluation: the "model" answers with the skills it has (keywords in the template) that the task needs
        key = re.search(r"task \d+-\d+", prompt).group(0)
        words = set(prompt.replace(key, "").split())
        have = {s for s in NEEDS[key] if s in words}
        # the scorer only sees text, so encode the (possibly trap-capped) fraction as the answer list
        return " ".join(sorted(have)) + f" |frac={_skill_frac(words, NEEDS[key]):.4f}"
    return MockClient(handler, max_concurrency=16, budget=Budget(max_calls=budget))


def scorer(noise: float, seed: int):
    def _score(prompt, example, completion, ctx):
        frac = float(completion.text.rsplit("|frac=", 1)[-1])
        rnd = random.Random(_h(seed, "noise", prompt.template, example.id))
        if rnd.random() < noise:
            frac = max(0.0, min(1.0, frac + rnd.choice([-0.5, 0.5])))
        return {"acc": frac}
    return _score


def true_score(node, dataset) -> float:
    words = set(node.prompt.template.split())
    return statistics.mean(_skill_frac(words, ex.answer) for ex in dataset)


def feedback(example, result):
    need, got = set(example.answer), set((result.output or "").split())
    miss = sorted(need - got)
    return f"missing: {', '.join(miss)}" if miss else "correct"


# ---------------------------------------------------------------- arms
def parent_selector(arm: str, seed: int, r: int, ids: set[str], bo: BOSelector | None, warmup: int):
    mode = {"gepa-weighted": "weighted", "gepa-uniform": "uniform", "gepa-best": "best", "gepa-all": "all"}.get(arm, "weighted")
    base = pareto_sample(1, mode=mode, seed=seed * 7919 + r, ids=ids)
    if not arm.startswith("bo"):
        return base

    async def _sel(tree):
        expanded = [n for n in tree if n.state == "expanded" and best_child(n, tree) is not None]
        if len(expanded) < warmup:
            return base(tree)
        among = pareto_pool(tree, ids)[0] if arm == "bo-in-pareto" else candidates(tree, ids)
        ranked = await bo.rank(tree, among)
        return [n for _, n in ranked[:1]]
    return _sel


def make_schedule(arm: str, seed: int, minibatch: int, warmup: int):
    screen = arm.endswith("+screen")
    parent_bo = BOSelector(HashEmbedder(dim=128), GPR(), EI(), value=best_child) if arm.startswith("bo") else None
    child_bo = BOSelector(HashEmbedder(dim=128), GPR(), EI(), value=own_score) if screen else None

    def schedule(tree, r):
        full = tree.task.dataset
        ids = {ex.id for ex in full}
        if not tree.evaluated_nodes():
            return [step(evaluate(dataset=full), select.root, name="root")]
        mb = minibatch_for(r, full, minibatch, seed)
        is_new = lambda n: n.origin.op == "reflect" and n.evaluation is None
        on_mb = lambda n: n.origin.op == "reflect" and n.evaluation is not None and not ids.issubset(set(n.evaluation.dataset_ids))
        if screen:
            # keep only the surrogate's favourite among this round's proposals; the rest stay unevaluated forever
            async def pick(tree):
                fresh = [n for n in select.where(is_new)(tree) if not n.origin.params.get("screened_out")]
                if len(fresh) <= 1:
                    return fresh
                ranked = await child_bo.rank(tree, fresh)
                for _, n in ranked[1:]:
                    n.origin.params["screened_out"] = True
                return [ranked[0][1]]
            to_minibatch = pick
        else:
            to_minibatch = select.where(is_new)
        return [
            step(ReflectiveExpander(feedback, minibatch=minibatch, n=3 if screen else 1, seed=seed + r),
                 parent_selector(arm.replace("+screen", ""), seed, r, ids, parent_bo, warmup), name="reflect"),
            step(evaluate(dataset=mb), to_minibatch, name="minibatch"),
            step(evaluate(dataset=full), lambda t: _accepted(t, select.where(on_mb)(t)), name="full"),
        ]
    return schedule


async def run_arm(arm: str, seed: int, args) -> list[tuple[int, float]]:
    dataset = make_dataset(args.n_examples, seed)
    client = make_client(seed, args.noise, args.p_informed, args.budget, trap=args.trap)
    task = Task(root="Please answer the request. {x}", description="answers requests that need certain skills",
                dataset=dataset, scorer=scorer(args.noise, seed), objective=LinearObjective(acc=1.0), client=client)
    tree = Tree(task)
    curve: list[tuple[int, float]] = []

    def record(tree, r, st):
        best = max((n for n in candidates(tree)), key=lambda n: n.score, default=None)
        curve.append((client.usage.calls, true_score(best, dataset) if best else 0.0))
    await run(tree, make_schedule(arm, seed, args.minibatch, args.warmup), stop=Stop(rounds=10_000), on_step=record)
    curve.append((client.usage.calls, curve[-1][1] if curve else 0.0))
    return curve


def best_at(curve, calls: int) -> float:
    v = 0.0
    for c, s in curve:
        if c <= calls:
            v = s
        else:
            break
    return v


ARMS = ["gepa-weighted", "gepa-uniform", "gepa-best", "gepa-all", "bo-replace", "bo-in-pareto", "gepa+screen", "bo+screen"]


async def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    grid = list(range(0, args.budget + 1, args.budget // 20))
    results = {}
    for arm in (args.arms or ARMS):
        curves = [await run_arm(arm, s, args) for s in range(args.seeds)]
        mat = [[best_at(c, g) for g in grid] for c in curves]
        mean = [statistics.mean(col) for col in zip(*mat)]
        se = [statistics.stdev(col) / len(col) ** 0.5 if len(col) > 1 else 0.0 for col in zip(*mat)]
        reach = [next((c for c, s in cv if s >= args.target), None) for cv in curves]
        results[arm] = {"grid": grid, "mean": mean, "se": se, "final": mean[-1], "final_se": se[-1],
                        "reached_target": sum(r is not None for r in reach) / len(reach),
                        "median_calls_to_target": statistics.median([r for r in reach if r is not None]) if any(r is not None for r in reach) else None}
        print(f"{arm:14s} final {mean[-1]:.3f} ± {se[-1]:.3f}   reached {args.target:.2f}: {results[arm]['reached_target']:.0%}"
              f"   median calls to target: {results[arm]['median_calls_to_target']}", flush=True)
    (out / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    lines = [f"| arm | final (mean ± SE, n={args.seeds}) | reached {args.target} | median rollouts to target |", "|---|---|---|---|"]
    for arm, r in results.items():
        lines.append(f"| {arm} | {r['final']:.3f} ± {r['final_se']:.3f} | {r['reached_target']:.0%} | {r['median_calls_to_target']} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm, r in results.items():
            ax.plot(r["grid"], r["mean"], label=arm, lw=2 if arm in ("gepa-weighted", "bo-replace") else 1)
            ax.fill_between(r["grid"], [m - s for m, s in zip(r["mean"], r["se"])], [m + s for m, s in zip(r["mean"], r["se"])], alpha=0.12)
        ax.set_xlabel("rollouts (unique model calls)"); ax.set_ylabel("true score of best candidate")
        ax.set_title(f"synthetic ladder, {args.seeds} seeds, noise={args.noise}, p_informed={args.p_informed}, trap={args.trap}")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "curves.png", dpi=120)
        print("plot:", out / "curves.png")
    except ImportError:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=int, default=600, help="rollouts per arm per seed")
    ap.add_argument("--n-examples", type=int, default=40)
    ap.add_argument("--minibatch", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--p-informed", type=float, default=0.5, help="prob. the mock reflection adds a *missing* skill")
    ap.add_argument("--warmup", type=int, default=4, help="expanded nodes before BO arms stop using the GEPA sampler")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--trap", action="store_true", help="deceptive landscape: trap words lift the mean to 0.5-0.6 and cap it there")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--out", default="experiments/synthetic-ladder")
    asyncio.run(main(ap.parse_args()))
