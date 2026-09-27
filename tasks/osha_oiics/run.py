"""Detailed-OIICS two-step runner. Every live call goes through a bpto client with a Budget and a CompletionCache file.

    python -m tasks.osha_oiics.run control                    # C0: bare group list / bare code list, by hand ($0)
    python -m tasks.osha_oiics.run author                     # A: Opus 5.5 writes both modules (1 call)
    python -m tasks.osha_oiics.run score C0 A --split val     # pilot: both steps, 2 repeats
    python -m tasks.osha_oiics.run score A --oracle           # step 2 alone, handed the gold group (its ceiling)
    python -m tasks.osha_oiics.run gepa --arm q1 --seed 0     # one optimizer run (B=5000 rows, minibatch 15)

Prompt sets live in runs/osha_oiics/prompts/<name>.json: {"route": ..., "code": ..., "meta": {...}}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from bpto import Budget, CompletionCache, ModelConfig
from bpto.llm.base import BudgetExceeded
from bpto.llm.bedrock import BedrockClient
from tasks.osha_sir.official_gepa import LogOnlyCache
from tasks.osha_sir.run import LITE, MICRO, REGION, _env

from .data import codebook, read
from .program import SLOTS, Row, TwoStepAdapter, run_official

STRONG = "us.anthropic.claude-opus-5-5"
RUNS = Path("runs/osha_oiics")
PROMPTS = RUNS / "prompts"


def client(model: str, budget: Budget, *, temperature: float | None = None, max_tokens: int = 4096,
           concurrency: int = 16, cache: str = "cache.jsonl", replay: bool = True, **kw) -> BedrockClient:
    _env()
    (RUNS / cache).parent.mkdir(parents=True, exist_ok=True)
    return BedrockClient(model, region=REGION, max_concurrency=concurrency, budget=budget, **kw,
                         cache=CompletionCache(RUNS / cache) if replay else LogOnlyCache(RUNS / cache),
                         default_config=ModelConfig(model=model, temperature=temperature, max_tokens=max_tokens))


def rows(split: str) -> list[Row]:
    return [Row(r["id"], r["narrative"], r["group"], r["code"]) for r in read(split)]


def check(p: dict) -> dict:
    book = codebook()
    return {"slots": {k: {s: p[k].count("{" + s + "}") for s in SLOTS[k]} for k in SLOTS},
            "route_names_all_groups": all(t in p["route"] or re.search(rf"(?<!\d){g}(?!\d)", p["route"])
                                          for g, t in book["groups"].items()),
            "chars": {k: len(p[k]) for k in SLOTS}}


def save(name: str, route: str, code: str, meta: dict) -> dict:
    PROMPTS.mkdir(parents=True, exist_ok=True)
    p = {"route": route.strip(), "code": code.strip()}
    p["meta"] = {**meta, **check(p), "date": time.strftime("%Y-%m-%d")}
    (PROMPTS / f"{name}.json").write_text(json.dumps(p, indent=1, ensure_ascii=False))
    print(name, json.dumps({k: p["meta"][k] for k in ("slots", "route_names_all_groups", "chars")}))
    return p


def load(name: str) -> dict[str, str]:
    p = json.loads((PROMPTS / f"{name}.json").read_text()) if (PROMPTS / f"{name}.json").exists() \
        else json.loads(Path(name).read_text())
    return {"route": p["route"], "code": p["code"]}


def control():
    book = codebook()
    groups = "\n".join(f"{g} {t}" for g, t in book["groups"].items())
    route = ("Classify this workplace injury narrative into one OIICS event or exposure major group:\n" + groups
             + "\n\nAnswer with the major group only.\n\nNarrative: {narrative}")
    code = ("This workplace injury narrative is in OIICS event major group {group}. Choose its detailed event code "
            "from:\n{codes}\n\nAnswer with the code only.\n\nNarrative: {narrative}")
    save("C0", route, code, {"op": "control", "model": None})


AUTHOR_PROMPT = """Write the two prompts of a two-step program that assigns the detailed OIICS 2.01 event or exposure \
code to an OSHA Severe Injury Report narrative. A small, fast model runs each prompt.

Step 1 (the "route" prompt) reads the narrative and names one of these 18 major groups:
{groups}
It must contain the placeholder {{narrative}} exactly once and name every major group.

Step 2 (the "code" prompt) reads the narrative and the major group step 1 chose, and picks one detailed code of that \
group. At run time the program fills {{group}} with the chosen group ("43 Falls to lower level") and {{codes}} with \
that group's codes, one "code: title" per line. It must contain {{narrative}}, {{group}} and {{codes}} exactly once \
each and work for every group. Guidance for particular groups is allowed.

The detailed codes that occur, by group (for your reference):
{codes}

Each prompt should end by asking for the answer alone (step 1: the major group; step 2: the four-digit code). Keep \
each prompt under about 1,200 words. Return exactly:
<route_prompt>
...
</route_prompt>
<code_prompt>
...
</code_prompt>"""


def author():
    book = codebook()
    groups = "\n".join(f"{g} {t}" for g, t in book["groups"].items())
    codes = "\n".join(f"{g} {book['groups'][g]}\n" + "\n".join(f"  {c}: {t}" for c, t in cs.items())
                      for g, cs in book["codes"].items())
    prompt = AUTHOR_PROMPT.replace("{groups}", groups).replace("{codes}", codes).replace("{{", "{").replace("}}", "}")

    async def go():
        c = client(STRONG, Budget(max_calls=2), cache="authoring_cache.jsonl", max_tokens=16000,
                   read_timeout=900, max_retries=1)
        return await c.complete(prompt)
    comp = asyncio.run(go())
    assert comp.stop_reason != "max_tokens", "hit max_tokens"
    (PROMPTS).mkdir(parents=True, exist_ok=True)
    (PROMPTS / "A.raw.txt").write_text(comp.text)
    r = re.search(r"<route_prompt>\s*(.*?)\s*</route_prompt>", comp.text, re.S)
    k = re.search(r"<code_prompt>\s*(.*?)\s*</code_prompt>", comp.text, re.S)
    assert r and k, "reply lacks the two blocks; see A.raw.txt"
    save("A", r.group(1), k.group(1), {"op": "author", "model": STRONG, "meta_prompt": prompt})


def metrics(traces) -> dict:
    n = len(traces)
    in_g = [t for t in traces if t.group == t.row.group]
    tok = lambda step, i: [t.tokens[step][i] for t in traces if step in t.tokens]
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else 0
    return {"code_acc": round(sum(t.code == t.row.code for t in traces) / n, 4),
            "group_acc": round(len(in_g) / n, 4),
            "code_acc_given_group": round(sum(t.code == t.row.code for t in in_g) / len(in_g), 4) if in_g else None,
            "route_parse_fail": sum(t.group is None and not t.error for t in traces),
            "code_parse_fail": sum(t.code is None and t.group is not None and not t.error for t in traces),
            "errors": sum(t.error is not None for t in traces), "slot_restored": any(t.slot_restored for t in traces),
            "route_in": mean(tok("route", 0)), "route_out": mean(tok("route", 1)),
            "code_in": mean(tok("code", 0)), "code_out": mean(tok("code", 1)), "code_calls": len(tok("code", 0))}


def score(args):
    book = codebook()
    data = rows(args.split)
    spend = Budget(max_usd=args.max_usd)
    outdir = RUNS / "scores" / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    mode = "oracle" if args.oracle else "both"
    for name in args.prompts:
        cand = load(name)
        reps = []
        for r in range(args.repeats):
            c = client(MICRO, Budget(max_calls=2 * len(data) + 10, parent=spend), temperature=0.0, max_tokens=1024,
                       concurrency=args.concurrency, cache=f"score_rep{r}.jsonl")
            ad = TwoStepAdapter(c, book, None, oracle_group=args.oracle)

            async def go():
                return await asyncio.gather(*(ad.run_row(cand, x) for x in data))
            traces = [t for t, _, _ in asyncio.run(go())]
            stem = Path(name).stem
            (outdir / f"{stem}_{mode}_r{r}.jsonl").write_text("".join(json.dumps(
                {"id": t.row.id, "gold": t.row.code, "group": t.group, "code": t.code, "route_out": t.route_out,
                 "code_out": t.code_out, "error": t.error, "tokens": t.tokens}) + "\n" for t in traces))
            reps.append(traces)
        m = [metrics(tr) for tr in reps]
        flips = [sum(a.code != b.code for a, b in zip(reps[0], reps[k])) for k in range(1, len(reps))]
        summary[f"{Path(name).stem}_{mode}"] = {"reps": m, "code_flips_vs_rep0": flips}
        print(f"{Path(name).stem} [{mode}]", " | ".join(
            f"code {x['code_acc']} group {x['group_acc']} code|group {x['code_acc_given_group']} "
            f"pf {x['route_parse_fail']}/{x['code_parse_fail']} err {x['errors']} in {x['route_in']}+{x['code_in']}"
            for x in m), "| flips", flips)
    prev = json.loads((outdir / "summary.json").read_text()) if (outdir / "summary.json").exists() else {}
    (outdir / "summary.json").write_text(json.dumps({**prev, **summary}, indent=2))
    print(f"spent ${spend.spent_usd:.4f}")


def gepa_run(args):
    from bpto.bo import BedrockEmbedder
    from tasks.osha_sir.qei_sampling import QEISampling

    book = codebook()
    train, val = rows("train"), rows("val")[: args.val_n]
    seed = load(args.seed_prompt)
    spend = Budget(max_usd=args.max_usd)
    task = client(MICRO, Budget(max_calls=args.task_cap or int(args.B * 2.4), parent=spend), temperature=0.0,
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
    t0, err = time.time(), None
    try:
        res, adapter, lm = run_official(seed, train, val, book, task_client=task, reflect_client=refl,
                                        max_metric_calls=args.B, minibatch=args.minibatch, seed_value=args.seed,
                                        run_dir=str(out / "gepa_state"), sampling_strategy=sampling)
    except Exception as e:  # BudgetExceeded included: record what was spent, then re-raise
        err = e
        raise
    finally:
        summary = {"arm": args.arm, "seed": args.seed, "seed_prompt": args.seed_prompt, "B": args.B,
                   "val_n": len(val), "minibatch": args.minibatch, "secs": round(time.time() - t0, 1),
                   "spent_usd": round(spend.spent_usd, 4), "task_calls_billed": task.usage.calls,
                   "task_cache_hits": task.usage.cache_hits, "task_tokens": [task.usage.input_tokens, task.usage.output_tokens],
                   "reflect_calls": refl.usage.calls, "embed_calls": getattr(emb, "calls", 0),
                   "error": f"{type(err).__name__}: {err}" if err else None}
        if err is None:
            best = res.best_idx
            st = adapter.stats
            summary.update(total_metric_calls=res.total_metric_calls, candidates=len(res.candidates),
                           val_scores=[round(s, 4) for s in res.val_aggregate_scores], best_idx=best,
                           best_val=round(res.val_aggregate_scores[best], 4), seed_val=round(res.val_aggregate_scores[0], 4),
                           discovery_calls=res.discovery_eval_counts, parents=res.parents,
                           rows_requested=st.requested, rows_errors=st.errors, evals_slot_restored=st.slot_restored,
                           group_acc_all_rows=round(st.group_correct / max(st.requested, 1), 4), stalled=adapter.stall.fired,
                           reflection_failures=sum("reflection failed" in l.lower() for l in adapter.logger.lines))
            (out / "best_prompt.json").write_text(json.dumps(res.best_candidate, indent=1, ensure_ascii=False))
            (out / "candidates.json").write_text(json.dumps(res.candidates, indent=1, ensure_ascii=False))
            (out / "gepa_log.txt").write_text("\n".join(adapter.logger.lines))
            if args.arm == "qei4":
                (out / "qei_log.json").write_text(json.dumps(sampling.log, indent=1, default=str))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k not in ("discovery_calls", "parents", "val_scores")}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("control")
    sub.add_parser("author")
    sc = sub.add_parser("score")
    sc.add_argument("prompts", nargs="+")
    sc.add_argument("--split", default="val", choices=["train", "val", "holdout"])
    sc.add_argument("--repeats", type=int, default=2)
    sc.add_argument("--oracle", action="store_true", help="skip step 1; step 2 gets the gold group")
    sc.add_argument("--max-usd", type=float, default=0.50)
    sc.add_argument("--concurrency", type=int, default=16)
    g = sub.add_parser("gepa")
    g.add_argument("--arm", choices=["q1", "independent4", "qei4"], required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--B", type=int, default=5000)
    g.add_argument("--val-n", type=int, default=200)
    g.add_argument("--minibatch", type=int, default=15)
    g.add_argument("--seed-prompt", default="A")
    g.add_argument("--max-usd", type=float, default=1.00)
    g.add_argument("--task-cap", type=int, default=None)
    g.add_argument("--task-max-tokens", type=int, default=1024)
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    {"control": control, "author": author, "gepa": lambda: gepa_run(args), "score": lambda: score(args)}[args.cmd]()


if __name__ == "__main__":
    main()
