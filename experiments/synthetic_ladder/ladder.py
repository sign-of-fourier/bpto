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
  bo-pure         no minibatch: BOSelector over candidates, propose 3, surrogate keeps 1 and it is evaluated on the
                  full set. The GP trains on full evaluations only; targets PIT-transformed; per-node noise.

All arms: same minibatch gate, same rollout budget (unique model calls). BO arms fall back to the GEPA
sampler until `warmup` nodes have been expanded (nothing to fit before that).

    python -m experiments.synthetic_ladder.ladder --seeds 30 --budget 600 --out experiments/<date>-synthetic-ladder

Two-module variant (`--modules 2`): the node is a Program {A, B}; skills 0-2 only count when in A, 3-5 only in B;
mutation is round-robin (module r mod 2), the mock reflector rewrites the module it is shown, and per-module
feedback lists that module's missing skills. `--interaction` adds a non-additive bonus (A has skill0 and B has
skill3). Arms: gepa-rr | bo-concat+screen (one RBF over the concatenated embeddings - the ablation) |
bo-additive+screen (AdditiveGPR, full-posterior EI for parents) | bo-additive-comp+screen (parent EI on the round's
module component alone). BO arms get per-node noise unless --no-noise.

    python -m experiments.synthetic_ladder.ladder --modules 2 --seeds 30 --budget 600 --out experiments/<date>-synthetic-ladder-2mod
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

from bpto import (Budget, BudgetExceeded, Dataset, LinearObjective, MockClient, Program, Prompt, Stop, Task, Tree,
                  evaluate, run, select)
from bpto.bo import EI, GPR, QEI, UCB, AdditiveGPR, BOSelector, HashEmbedder, KrigingBeliever, QuantecarloQEI, Thompson
from bpto.gepa import ReflectiveExpander, candidates, pareto_pool, pareto_sample
from bpto.gepa.loop import _accepted, minibatch_for
from bpto.gepa.select import example_scores
from bpto.ops import Variants
from bpto.search import step
from bpto.value import best_child, own_score

K = 6
SKILLS = [f"skill{j}" for j in range(K)]
DISTRACTORS = [f"word{j}" for j in range(24)]
TRAPS = ["trap0", "trap1", "trap2"]  # --trap: a trap word lifts every example to >= 0.5 but caps it at 0.6
VOCAB = SKILLS + DISTRACTORS


def set_skills(k: int) -> None:
    """--skills k: a deeper ladder (2026-09-19). Distractors scale 4:1 so a random addition hits a skill at the same
    rate; examples need 1..k/3 skills so the climb is k accepted mutations long and the pool is wide mid-search."""
    global K
    K = k
    SKILLS[:] = [f"skill{j}" for j in range(K)]
    DISTRACTORS[:] = [f"word{j}" for j in range(4 * K)]
    VOCAB[:] = SKILLS + DISTRACTORS
    OWNER.clear(); OWNER.update({s: MODULES[0] if j < K // 2 else MODULES[1] for j, s in enumerate(SKILLS)})


TRAP_CAP = (0.5, 0.6)
MODULES = ["A", "B"]
OWNER = {s: MODULES[0] if j < K // 2 else MODULES[1] for j, s in enumerate(SKILLS)}  # --modules 2: which module a skill counts in
INTERACTION = {"A": "skill0", "B": "skill3"}  # --interaction: +0.25 when both present (non-additive)


def module_words(prompt) -> dict[str, set[str]]:
    if isinstance(prompt, Program):
        return {m: set(p.template.split()) for m, p in prompt.modules.items()}
    return {MODULES[0]: set(prompt.template.split())}


def _skill_frac2(mw: dict[str, set[str]], need: list[str], interaction: bool) -> float:
    frac = sum(s in mw[OWNER[s]] for s in need) / len(need)
    if interaction and all(INTERACTION[m] in mw[m] for m in MODULES):
        frac = min(1.0, frac + 0.25)
    return frac


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
        need = sorted(rnd.sample(SKILLS, rnd.choice([1, 2, 2, 3] if K <= 6 else list(range(1, K // 3 + 1)) + [2])))
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


def scorer(noise: float, seed: int, modules: int = 1, interaction: bool = False):
    def _score(prompt, example, completion, ctx):
        if modules > 1:  # both modules' contributions come from the templates; one call per example is still charged
            mw = module_words(prompt)
            for m in MODULES:
                have = sorted(s for s in example.answer if OWNER[s] == m and s in mw[m])
                ctx.trace[m] = {"input": example.inputs["x"], "output": " ".join(have)}
            frac, key = _skill_frac2(mw, example.answer, interaction), str(prompt)
        else:
            frac, key = float(completion.text.rsplit("|frac=", 1)[-1]), prompt.template
        rnd = random.Random(_h(seed, "noise", key, example.id))
        if rnd.random() < noise:
            frac = max(0.0, min(1.0, frac + rnd.choice([-0.5, 0.5])))
        return {"acc": frac}
    return _score


def true_score(node, dataset, interaction: bool = False) -> float:
    if isinstance(node.prompt, Program):
        mw = module_words(node.prompt)
        return statistics.mean(_skill_frac2(mw, ex.answer, interaction) for ex in dataset)
    words = set(node.prompt.template.split())
    return statistics.mean(_skill_frac(words, ex.answer) for ex in dataset)


def feedback(example, result):
    need, got = set(example.answer), set((result.output or "").split())
    miss = sorted(need - got)
    return f"missing: {', '.join(miss)}" if miss else "correct"


def module_feedback(m: str):
    """Only the skills this module owns: the reflector for A is never told about B's misses."""
    def _fb(example, result):
        need = {s for s in example.answer if OWNER[s] == m}
        got = set(((result.trace or {}).get(m) or {}).get("output", "").split())
        miss = sorted(need - got)
        return f"missing: {', '.join(miss)}" if miss else "correct"
    return _fb


FEEDBACK2 = {m: module_feedback(m) for m in MODULES}


def se_of(metric):
    return lambda n, t: n.evaluation.metrics_std.get(metric, 0.0) / max(1, n.evaluation.n) ** 0.5 if n.evaluation else None


def best_child_se(n, tree):
    kids = [c for c in tree.child_nodes(n) if c.evaluation is not None]
    return se_of("acc")(max(kids, key=lambda c: c.score), tree) if kids else None


# ---------------------------------------------------------------- arms
def parent_selector(arm: str, seed: int, r: int, ids: set[str], bo: BOSelector | None, warmup: int, module: str | None = None,
                    q: int = 1, log: list | None = None):
    """`q` > 1 (gepa-weighted and gepaei only, 2026-09-18): q parents per round once the pool has more than one member -
    q Pareto draws without replacement for gepa, `bo.top(q)` (joint batch acquisition if `bo.batch` is set) for gepaei.
    The speed question: the same rollouts in fewer rounds, and does a joint pick choose a more useful pair than two draws."""
    mode = {"gepa-weighted": "weighted", "gepa-uniform": "uniform", "gepa-best": "best", "gepa-all": "all"}.get(arm, "weighted")
    base = pareto_sample(1, mode=mode, seed=seed * 7919 + r, ids=ids)
    if arm == "gepa-weighted" and q > 1:
        def _gepa_q(tree):
            cands = candidates(tree, ids)
            k = min(q, len(cands)) if len(cands) > 1 else 1
            picks = pareto_sample(k, mode=mode, seed=seed * 7919 + r, ids=ids)(tree)
            if log is not None:
                log.append({"round": r, "n_cands": len(cands), "q": k, "sibling": len({n.parent_id for n in picks}) < len(picks)})
            return picks
        return _gepa_q
    if not (arm.startswith("bo") or arm == "gepaei"):
        return base

    async def _sel(tree):
        if arm == "gepaei":
            # the pool grows one gate-passer at a time: warm up until every current candidate (up to `warmup`) has been tried
            cands = candidates(tree, ids)
            trained = [n for n in cands if bo.value(n, tree) is not None]
            k = min(q, len(cands)) if len(cands) > 1 else 1
            if len(trained) < min(warmup, len(cands)):
                picks = pareto_sample(k, mode=mode, seed=seed * 7919 + r, ids=ids)(tree)
                rec = {"warmup": True}
            else:
                picks = await bo.top(k, among=lambda t: cands)(tree)
                b = bo.last_fit.get("batch")
                rec = {"warmup": False, "flat": bo.last_fit.get("flat"), "batch_differs": None if not b else set(b["ids"]) != set(b["top_k_ids"])}
            if log is not None:
                log.append({"round": r, "n_cands": len(cands), "q": k, "sibling": len({n.parent_id for n in picks}) < len(picks), **rec})
            return picks
        expanded = [n for n in tree if n.state == "expanded" and best_child(n, tree) is not None]
        if len(expanded) < warmup:
            return base(tree)
        among = pareto_pool(tree, ids)[0] if arm == "bo-in-pareto" else candidates(tree, ids)
        # "-comp": rank parents on the round's module component alone (ignores the modules the child keeps)
        ranked = await bo.rank(tree, among, module=module if arm.endswith("-comp") else None)
        return [n for _, n in ranked[:1]]
    return _sel


def child_est(n, tree):
    """gepaei (v2, 2026-09-17): every evaluated child's estimated full score at the parent's input =
    parent full score + (child − parent on the child's own rows). One observation per proposal, gate pass or fail."""
    if not _is_full(n, tree):
        return None
    ps = example_scores(tree, n)
    out = []
    for c in tree.child_nodes(n):
        if c.evaluation is None or not all(i in ps for i in c.evaluation.dataset_ids):
            continue
        ids = c.evaluation.dataset_ids
        out.append(n.score + c.score - sum(ps[i] for i in ids) / max(1, len(ids)))
    return out or None


def child_est_se(n, tree):
    if not _is_full(n, tree):
        return None
    ps = example_scores(tree, n)
    return [se_of("acc")(c, tree) for c in tree.child_nodes(n)
            if c.evaluation is not None and all(i in ps for i in c.evaluation.dataset_ids)] or None


def _is_full(n, tree):
    return n.evaluation is not None and {ex.id for ex in tree.task.dataset}.issubset(set(n.evaluation.dataset_ids))


def full_score(n, tree):
    return n.score if _is_full(n, tree) else None


def best_full_child(n, tree):
    v = [c.score for c in tree.child_nodes(n) if _is_full(c, tree)]
    return max(v) if v else None


def best_full_child_se(n, tree):
    kids = [c for c in tree.child_nodes(n) if _is_full(c, tree)]
    return se_of("acc")(max(kids, key=lambda c: c.score), tree) if kids else None


def make_schedule(arm: str, seed: int, minibatch: int, warmup: int, modules: int = 1, use_noise: bool = True,
                  q: int = 1, batch=None, log: list | None = None):
    screen = arm.endswith("+screen")
    if arm.startswith("bo-pure"):
        # bo-pure[-<value>-<acq>]: value in {child (best full child), own}; acq in {ei, ucb, ts}
        parts = arm.split("-")[2:]
        val = {"child": best_full_child, "own": full_score}[parts[0] if parts else "child"]
        vse = best_full_child_se if val is best_full_child else se_of("acc")
        acq = {"ei": EI, "ucb": UCB, "ts": lambda: Thompson(seed=seed)}[parts[1] if len(parts) > 1 else "ei"]
        surrogate = AdditiveGPR if modules > 1 else GPR
        parent_bo = BOSelector(HashEmbedder(dim=128), surrogate(), acq(), value=val, noise=vse, transform="pit")
        child_bo = BOSelector(HashEmbedder(dim=128), surrogate(), EI(), value=full_score, noise=se_of("acc"), transform="pit")

        def schedule(tree, r):
            full = tree.task.dataset
            ids = {ex.id for ex in full}
            if not tree.evaluated_nodes():
                return [step(evaluate(dataset=full), select.root, name="root")]
            module = MODULES[r % len(MODULES)] if modules > 1 else None
            is_new = lambda n: n.origin.op == "reflect" and n.evaluation is None and not n.origin.params.get("screened_out")

            async def pick(tree):
                fresh = select.where(is_new)(tree)
                if len(fresh) <= 1:
                    return fresh
                ranked = await child_bo.rank(tree, fresh)
                for _, n in ranked[1:]:
                    n.origin.params["screened_out"] = True
                return [ranked[0][1]]

            async def parent(tree):
                base = pareto_sample(1, mode="weighted", seed=seed * 7919 + r, ids=ids)
                expanded = [n for n in tree if n.state == "expanded" and best_full_child(n, tree) is not None]
                if len(expanded) < warmup:
                    return base(tree)
                ranked = await parent_bo.rank(tree, candidates(tree, ids))
                tree.meta.setdefault("parents", []).append(ranked[0][1].id)
                return [n for _, n in ranked[:1]]
            fb = FEEDBACK2 if modules > 1 else feedback
            return [
                step(ReflectiveExpander(fb, minibatch=minibatch, n=3, seed=seed + r, module=module), parent, name="reflect"),
                step(evaluate(dataset=full), pick, name="full"),
            ]
        return schedule
    if modules > 1:
        surrogate = AdditiveGPR if arm.startswith("bo-additive") else GPR
        pn, cn = (best_child_se, se_of("acc")) if use_noise else (None, None)
        parent_bo = BOSelector(HashEmbedder(dim=128), surrogate(), EI(), value=best_child, noise=pn) if arm.startswith("bo") else None
        child_bo = BOSelector(HashEmbedder(dim=128), surrogate(), EI(), value=own_score, noise=cn) if screen else None
    else:
        if arm == "gepaei":
            parent_bo = BOSelector(HashEmbedder(dim=128), GPR(), EI(), value=child_est, noise=child_est_se if use_noise else None,
                                   transform="pit", pca=4, batch=batch)
        else:
            parent_bo = BOSelector(HashEmbedder(dim=128), GPR(), EI(), value=best_child) if arm.startswith("bo") else None
        child_bo = BOSelector(HashEmbedder(dim=128), GPR(), EI(), value=own_score) if screen else None

    def schedule(tree, r):
        module = MODULES[r % len(MODULES)] if modules > 1 else None
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
        fb = FEEDBACK2 if modules > 1 else feedback
        return [
            step(ReflectiveExpander(fb, minibatch=minibatch, n=3 if screen else 1, seed=seed + r, module=module),
                 parent_selector(arm.replace("+screen", ""), seed, r, ids, parent_bo, warmup, module, q=q, log=log), name="reflect"),
            step(evaluate(dataset=mb), to_minibatch, name="minibatch"),
            step(evaluate(dataset=full), lambda t: _accepted(t, select.where(on_mb)(t)), name="full"),
        ]
    return schedule


async def run_arm(arm: str, seed: int, args) -> list[tuple[int, float]]:
    dataset = make_dataset(args.n_examples, seed)
    client = make_client(seed, args.noise, args.p_informed, args.budget, trap=args.trap)
    if args.modules > 1:
        root = {"A": "Please answer the request. {x}", "B": "Then complete the second part. {x}"}
        description = {"A": "handles the first half of a request (module A)", "B": "handles the second half (module B)"}
    else:
        root, description = "Please answer the request. {x}", "answers requests that need certain skills"
    task = Task(root=root, description=description, dataset=dataset,
                scorer=scorer(args.noise, seed, args.modules, args.interaction), objective=LinearObjective(acc=1.0), client=client)
    tree = Tree(task)
    curve: list[tuple[int, float]] = []
    rounds: list[tuple[int, int]] = []  # (calls, round) - the speed readout: rounds to reach a score, not rollouts
    log: list[dict] = []

    def record(tree, r, st):
        best = max((n for n in candidates(tree)), key=lambda n: n.score, default=None)
        curve.append((client.usage.calls, true_score(best, dataset, args.interaction) if best else 0.0))
        rounds.append((client.usage.calls, r))
    batch = make_batch(args, seed) if arm == "gepaei" else None
    await run(tree, make_schedule(arm, seed, args.minibatch, args.warmup, args.modules, not args.no_noise, q=args.q, batch=batch, log=log),
              stop=Stop(rounds=10_000), on_step=record)
    curve.append((client.usage.calls, curve[-1][1] if curve else 0.0))
    return {"curve": curve, "rounds": rounds, "log": log, "pool": len(candidates(tree)), "nodes": len(tree)}


def make_batch(args, seed):
    if args.q <= 1 or args.batch == "topk":
        return None
    if args.batch == "kb":
        return KrigingBeliever()
    if args.batch == "qei":
        return QEI(seed=seed)
    return QuantecarloQEI(seed=seed)


def best_at(curve, calls: int) -> float:
    v = 0.0
    for c, s in curve:
        if c <= calls:
            v = s
        else:
            break
    return v


ARMS = ["gepa-weighted", "gepa-uniform", "gepa-best", "gepa-all", "bo-replace", "bo-in-pareto", "gepa+screen", "bo+screen", "bo-pure", "gepaei"]
ARMS2 = ["gepa-rr", "bo-concat+screen", "bo-additive+screen", "bo-additive-comp+screen"]


async def main(args):
    if getattr(args, "skills", 6) != 6:
        set_skills(args.skills)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    grid = list(range(0, args.budget + 1, args.budget // 20))
    results = {}
    for arm in (args.arms or (ARMS2 if args.modules > 1 else ARMS)):
        runs = [await run_arm(arm, s, args) for s in range(args.seeds)]
        curves = [x["curve"] for x in runs]
        mat = [[best_at(c, g) for g in grid] for c in curves]
        mean = [statistics.mean(col) for col in zip(*mat)]
        se = [statistics.stdev(col) / len(col) ** 0.5 if len(col) > 1 else 0.0 for col in zip(*mat)]
        reach = [next((c for c, s in cv if s >= args.target), None) for cv in curves]
        # rounds at which the target was reached (the round of the first step whose call count is >= reach)
        reach_rounds = [next((rr for c, rr in x["rounds"] if c >= rc), None) if rc is not None else None for x, rc in zip(runs, reach)]
        total_rounds = [x["rounds"][-1][1] if x["rounds"] else 0 for x in runs]
        logs = [e for x in runs for e in x["log"]]
        multi = [e for e in logs if e["q"] > 1]
        results[arm] = {"grid": grid, "mean": mean, "se": se, "final": mean[-1], "final_se": se[-1],
                        "reached_target": sum(r is not None for r in reach) / len(reach),
                        "median_calls_to_target": statistics.median([r for r in reach if r is not None]) if any(r is not None for r in reach) else None,
                        "median_rounds_to_target": statistics.median([r for r in reach_rounds if r is not None]) if any(r is not None for r in reach_rounds) else None,
                        "median_rounds_total": statistics.median(total_rounds), "mean_pool": statistics.mean(x["pool"] for x in runs),
                        "max_pool": max(x["pool"] for x in runs), "q": args.q, "batch": args.batch if args.q > 1 else None,
                        "q_rounds": len(multi), "q_rounds_share": len(multi) / max(1, len(logs)),
                        "sibling_pairs": sum(e["sibling"] for e in multi), "batch_differs": sum(bool(e.get("batch_differs")) for e in multi),
                        "flat_fits": sum(bool(e.get("flat")) for e in multi),
                        "per_seed": {"final": [c[-1][1] for c in curves], "calls_to_target": reach, "rounds_to_target": reach_rounds,
                                     "rounds_total": total_rounds, "pool": [x["pool"] for x in runs]}}
        rr = results[arm]
        print(f"{arm:14s} final {mean[-1]:.3f} ± {se[-1]:.3f}   reached {args.target:.2f}: {rr['reached_target']:.0%}"
              f"   median calls to target: {rr['median_calls_to_target']}   median rounds to target: {rr['median_rounds_to_target']}"
              f"   rounds total {rr['median_rounds_total']}   pool {rr['mean_pool']:.1f} (max {rr['max_pool']})"
              + (f"   q-rounds {rr['q_rounds_share']:.0%}, sibling pairs {rr['sibling_pairs']}/{rr['q_rounds']}, batch≠top-q {rr['batch_differs']}, flat {rr['flat_fits']}" if args.q > 1 else ""),
              flush=True)
    (out / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    lines = [f"| arm | q | batch | final (mean ± SE, n={args.seeds}) | reached {args.target} | median rollouts to target | median rounds to target | rounds total | pool (mean/max) | sibling pairs | batch ≠ top-q |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in results.items():
        lines.append(f"| {arm} | {r['q']} | {r['batch'] or '-'} | {r['final']:.3f} ± {r['final_se']:.3f} | {r['reached_target']:.0%} | {r['median_calls_to_target']} | {r['median_rounds_to_target']} | {r['median_rounds_total']} | {r['mean_pool']:.1f}/{r['max_pool']} | {r['sibling_pairs']}/{r['q_rounds']} | {r['batch_differs']} |")
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
        ax.set_title(f"synthetic ladder, {args.seeds} seeds, noise={args.noise}, p_informed={args.p_informed}, trap={args.trap}"
                     + (f", modules={args.modules}, interaction={args.interaction}" if args.modules > 1 else ""))
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
    ap.add_argument("--modules", type=int, default=1, choices=[1, 2], help="2: Program nodes {A, B}, round-robin mutation")
    ap.add_argument("--no-noise", action="store_true", help="--modules 2: BO arms without per-node noise")
    ap.add_argument("--interaction", action="store_true", help="--modules 2: non-additive bonus when A has skill0 and B has skill3")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--trap", action="store_true", help="deceptive landscape: trap words lift the mean to 0.5-0.6 and cap it there")
    ap.add_argument("--skills", type=int, default=6, help="planted skills K (6 = the original ladder; 12 = twice the climb)")
    ap.add_argument("--q", type=int, default=1, help="gepa-weighted / gepaei: parents per round once the pool has > 1 member")
    ap.add_argument("--batch", choices=["topk", "kb", "qei", "quantecarlo"], default="qei", help="gepaei, q > 1: batch acquisition")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--out", default="experiments/synthetic-ladder")
    asyncio.run(main(ap.parse_args()))
