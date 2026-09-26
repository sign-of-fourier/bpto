"""OSHA SIR benchmark runner. Every live call goes through a bpto client with a Budget and a CompletionCache file.

    python -m tasks.osha_sir.run author                 # B1 / B2: two strong models write the seed prompts (2 calls)
    python -m tasks.osha_sir.run onestep                # one-step optimizer: one rewrite of B2, Lite and strong (2 calls)
    python -m tasks.osha_sir.run gepa --arm q1 --seed 0 --B 600 --val-n 50      # one optimizer run
        --arm q1 | independent4 | qei4

Prompts live in runs/osha_sir/prompts/*.txt (verbatim model output + a .json with how it was made).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from bpto import Budget, CompletionCache, ModelConfig
from bpto.llm.bedrock import BedrockClient

from .common import SLOT
from .official_gepa import LogOnlyCache
from .data import labels as load_labels, read

REGION = "us-east-1"
MICRO, LITE = "us.amazon.nova-micro-v1:0", "us.amazon.nova-lite-v1:0"
STRONG = {"B1": "us.openai.gpt-6-sol", "B2": "us.anthropic.claude-opus-5-5"}
ONESTEP_STRONG = STRONG["B2"]
RUNS = Path("runs/osha_sir")
PROMPTS = RUNS / "prompts"

AUTHOR_PROMPT = ("Write a prompt that classifies OSHA severe injury narratives into these event categories: {labels}. "
                 "Output only the category.\n\nThe prompt must contain the placeholder " + SLOT + " exactly once, where "
                 "the narrative will be inserted. Return only the prompt.")
ONESTEP_PROMPT = ("Improve this prompt. Keep the placeholder " + SLOT + " and the category list.\n\n"
                  "<prompt>\n{prompt}\n</prompt>\n\nReturn only the improved prompt.")


def _env():
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def client(model: str, budget: Budget, *, temperature: float | None = None, max_tokens: int = 4096,
           concurrency: int = 16, cache: str = "cache.jsonl", replay: bool = True) -> BedrockClient:
    _env()
    (RUNS / cache).parent.mkdir(parents=True, exist_ok=True)
    return BedrockClient(model, region=REGION, max_concurrency=concurrency, budget=budget,
                         cache=CompletionCache(RUNS / cache) if replay else LogOnlyCache(RUNS / cache),
                         default_config=ModelConfig(model=model, temperature=temperature, max_tokens=max_tokens))


def unfence(text: str) -> tuple[str, bool]:
    """The prompt inside a reply: a <prompt>...</prompt> block (what ONESTEP_PROMPT shows) or a reply that is one code
    fence; else the reply itself. The same rule for every rewrite; `extracted` is logged."""
    m = re.search(r"<prompt>\s*\n?(.*?)\n?\s*</prompt>", text, re.S) or re.fullmatch(r"\s*```[^\n]*\n(.*?)\n```\s*", text, re.S)
    return (m.group(1).strip(), True) if m else (text.strip(), False)


def save_prompt(name: str, raw: str, meta: dict) -> str:
    PROMPTS.mkdir(parents=True, exist_ok=True)
    text, fenced = unfence(raw)
    meta = {**meta, "extracted": fenced, "has_slot": SLOT in text, "slot_count": text.count(SLOT),
            "all_labels_present": all(l in text for l in load_labels()), "chars": len(text),
            "date": time.strftime("%Y-%m-%d")}
    (PROMPTS / f"{name}.raw.txt").write_text(raw)
    (PROMPTS / f"{name}.txt").write_text(text)
    (PROMPTS / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: {len(text)} chars, slot x{meta['slot_count']}, all labels {meta['all_labels_present']}, "
          f"extracted {fenced}")
    return text


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


async def _one(model: str, prompt: str, budget: Budget) -> str:
    c = await client(model, budget, cache="authoring_cache.jsonl", max_tokens=9000 if "nova" in model else 16000).complete(prompt)
    assert c.stop_reason != "max_tokens", f"{model} hit max_tokens"
    return c.text


def author():
    import asyncio

    budget = Budget(max_calls=4)
    lab = "; ".join(load_labels())
    prompt = AUTHOR_PROMPT.replace("{labels}", lab)
    for name, model in STRONG.items():
        raw = asyncio.run(_one(model, prompt, budget))
        save_prompt(name, raw, {"op": "author", "model": model, "meta_prompt": prompt})


def onestep():
    import asyncio

    b2 = load_prompt("B2")
    prompt = ONESTEP_PROMPT.replace("{prompt}", b2)
    for name, model, temp in (("onestep_lite", LITE, 1.0), ("onestep_strong", ONESTEP_STRONG, None)):
        budget = Budget(max_calls=2, max_usd=0.05) if model == LITE else Budget(max_calls=2)
        raw = asyncio.run(_one_cfg(model, prompt, budget, temp))
        save_prompt(name, raw, {"op": "onestep", "model": model, "temperature": temp, "source": "B2",
                                "meta_prompt": ONESTEP_PROMPT})


async def _one_cfg(model, prompt, budget, temp):
    c = await client(model, budget, temperature=temp, cache="authoring_cache.jsonl", max_tokens=9000 if "nova" in model else 16000).complete(prompt)
    assert c.stop_reason != "max_tokens", f"{model} hit max_tokens"
    return c.text


def b0():
    """The control: bare instruction + label list, written by hand, no model involved."""
    text = ("Classify this injury narrative into one of: " + "; ".join(load_labels())
            + ". Answer with the label only.\n\nNarrative: " + SLOT)
    if not (PROMPTS / "B0.txt").exists():
        save_prompt("B0", text, {"op": "control", "model": None})


def score(args):
    """Score prompts on a split, `repeats` times. Each repeat has its own cache file so a repeat is a fresh call
    (temperature 0 is still not deterministic on Nova). Per-row predictions + a summary per prompt."""
    import asyncio
    from collections import Counter

    from .common import parse_label, render

    labs = load_labels()
    rows = read(args.split)
    b0()
    spend = Budget(max_usd=args.max_usd)
    outdir = RUNS / "scores" / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}

    async def one_prompt(name, r):
        c = client(MICRO, Budget(parent=spend), temperature=0.0, max_tokens=1024, concurrency=args.concurrency,
                   cache=f"score_rep{r}.jsonl")
        text = (PROMPTS / f"{name}.txt").read_text() if (PROMPTS / f"{name}.txt").exists() else Path(name).read_text()

        async def row(x):
            prompt, restored = render(text, x["narrative"])
            try:
                comp = await c.complete(prompt)
                return {"id": x["id"], "gold": x["label"], "pred": parse_label(comp.text, labs), "raw": comp.text,
                        "in": comp.input_tokens, "out": comp.output_tokens, "restored": restored, "error": None}
            except Exception as e:
                if type(e).__name__ == "BudgetExceeded":
                    raise
                return {"id": x["id"], "gold": x["label"], "pred": None, "raw": "", "in": 0, "out": 0,
                        "restored": restored, "error": f"{type(e).__name__}: {e}"}
        return await asyncio.gather(*(row(x) for x in rows))

    def metrics(res):
        n = len(res)
        acc = sum(r["pred"] == r["gold"] for r in res) / n
        f1s, recs = [], []
        for l in labs:
            tp = sum(r["pred"] == l and r["gold"] == l for r in res)
            fp = sum(r["pred"] == l and r["gold"] != l for r in res)
            fn = sum(r["pred"] != l and r["gold"] == l for r in res)
            p_, r_ = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
            recs.append(r_)
        return {"acc": round(acc, 4), "macro_f1": round(sum(f1s) / len(f1s), 4),
                "balanced_acc": round(sum(recs) / len(recs), 4),
                "parse_fail": sum(r["pred"] is None and r["error"] is None for r in res),
                "errors": sum(r["error"] is not None for r in res), "slot_restored": res[0]["restored"],
                "mean_in_tokens": round(sum(r["in"] for r in res) / n), "mean_out_tokens": round(sum(r["out"] for r in res) / n, 1)}

    for name in args.prompts:
        reps = []
        for r in range(args.repeats):
            res = asyncio.run(one_prompt(name, r))
            (outdir / f"{Path(name).stem}_r{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in res))
            reps.append(res)
        m = [metrics(res) for res in reps]
        flips = [sum(a["pred"] != b["pred"] for a, b in zip(reps[0], reps[k])) for k in range(1, len(reps))]
        summary[Path(name).stem] = {"reps": m, "pred_flips_vs_rep0": flips}
        print(Path(name).stem, " | ".join(f"acc {x['acc']} mF1 {x['macro_f1']} bal {x['balanced_acc']} pf {x['parse_fail']} "
                                          f"err {x['errors']} in {x['mean_in_tokens']}" for x in m), "| flips", flips)
    prev = json.loads((outdir / "summary.json").read_text()) if (outdir / "summary.json").exists() else {}
    (outdir / "summary.json").write_text(json.dumps({**prev, **summary}, indent=2))
    print(f"spent ${spend.spent_usd:.4f}")


def gepa_run(args):
    from bpto.bo import BedrockEmbedder

    from .official_gepa import Row, run_official
    from .qei_sampling import QEISampling

    labs = load_labels()
    train = [Row(r["id"], r["narrative"], r["label"]) for r in read("train")]
    val = [Row(r["id"], r["narrative"], r["label"]) for r in read("val")][: args.val_n]
    seed_prompt = load_prompt(args.seed_prompt)
    spend = Budget(max_usd=args.max_usd)                       # the whole run's bill: task + reflection + embeddings
    # one task cache per run: runs execute concurrently (appends to one file could interleave) and a cache hit from
    # another run's calls would be free and instant, which would flatter that run's wall-clock
    task = client(MICRO, Budget(max_calls=args.task_cap or int(args.B * 1.5), parent=spend), temperature=0.0,
                  max_tokens=args.task_max_tokens, concurrency=args.concurrency,
                  cache=f"task_cache/{args.tag}{args.arm}_s{args.seed}.jsonl")
    refl = client(LITE, Budget(parent=spend), temperature=1.0, max_tokens=4096, concurrency=args.concurrency,
                  cache=f"reflect_log/{args.tag}{args.arm}_s{args.seed}.jsonl", replay=False)
    sampling, emb = None, None
    if args.arm == "independent4":
        from gepa.strategies.proposal_sampling import IndependentSampling
        sampling = IndependentSampling(4)
    elif args.arm == "qei4":
        _env()
        emb = BedrockEmbedder(region=REGION, budget=spend)
        sampling = QEISampling(4, emb, seed=args.seed)
    out = RUNS / "gepa" / f"{args.tag}{args.arm}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    err = None
    try:
        res, adapter, lm = run_official(seed_prompt, train, val, labs, task_client=task, reflect_client=refl,
                                        max_metric_calls=args.B, minibatch=args.minibatch, seed=args.seed,
                                        run_dir=str(out / "gepa_state"), sampling_strategy=sampling)
    except Exception as e:  # BudgetExceeded included: record what was spent, then re-raise
        err = e
        raise
    finally:
        secs = time.time() - t0
        summary = {"arm": args.arm, "seed": args.seed, "B": args.B, "val_n": len(val), "minibatch": args.minibatch,
                   "secs": round(secs, 1), "spent_usd": round(spend.spent_usd, 4),
                   "task_calls_billed": task.usage.calls, "task_cache_hits": task.usage.cache_hits,
                   "task_tokens": [task.usage.input_tokens, task.usage.output_tokens],
                   "reflect_calls": refl.usage.calls, "reflect_cache_hits": refl.usage.cache_hits, "embed_calls": getattr(emb, "calls", 0),
                   "error": f"{type(err).__name__}: {err}" if err else None}
        if err is None:
            best = res.best_idx
            summary.update(total_metric_calls=res.total_metric_calls, candidates=len(res.candidates),
                           val_scores=[round(s, 4) for s in res.val_aggregate_scores], best_idx=best,
                           best_val=round(res.val_aggregate_scores[best], 4), seed_val=round(res.val_aggregate_scores[0], 4),
                           discovery_calls=res.discovery_eval_counts, parents=res.parents,
                           rows_requested=adapter.stats.requested, rows_errors=adapter.stats.errors,
                           evals_slot_restored=adapter.stats.slot_restored, stalled=adapter.stall.fired,
                           reflection_failures=sum("reflection failed" in l.lower() for l in adapter.logger.lines))
            (out / "best_prompt.txt").write_text(res.best_candidate["instruction"])
            (out / "candidates.json").write_text(json.dumps(res.candidates, indent=1))
            (out / "gepa_log.txt").write_text("\n".join(adapter.logger.lines))
            if args.arm == "qei4":
                (out / "qei_log.json").write_text(json.dumps(sampling.log, indent=1, default=str))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k not in ("discovery_calls", "parents", "val_scores")}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("author")
    sub.add_parser("onestep")
    sc = sub.add_parser("score")
    sc.add_argument("prompts", nargs="+")
    sc.add_argument("--split", default="val", choices=["train", "val", "holdout"])
    sc.add_argument("--repeats", type=int, default=2)
    sc.add_argument("--max-usd", type=float, default=0.50)
    sc.add_argument("--concurrency", type=int, default=16)
    g = sub.add_parser("gepa")
    g.add_argument("--arm", choices=["q1", "independent4", "qei4"], required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--B", type=int, default=5000)
    g.add_argument("--val-n", type=int, default=200)
    g.add_argument("--minibatch", type=int, default=10)
    g.add_argument("--seed-prompt", default="B2")
    g.add_argument("--max-usd", type=float, default=0.50)
    g.add_argument("--task-cap", type=int, default=None)
    g.add_argument("--task-max-tokens", type=int, default=1024)
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    {"author": lambda: author(), "onestep": lambda: onestep(), "gepa": lambda: gepa_run(args), "score": lambda: score(args)}[args.cmd]()


if __name__ == "__main__":
    main()
