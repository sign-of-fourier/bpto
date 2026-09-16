"""Headroom pilot for the two-stage HotpotQA program (`tasks/hotpotqa/program.py`): root selector + hand variants,
each scored through the fixed answerer, plus two reference points that bypass the selector: ORACLE (answerer on
the gold paragraphs = the ceiling a perfect selector could reach) and ALL10 (answerer on the full context = the
single-prompt baseline). Headroom for selector search = oracle - root.

    python -m experiments.live_compare.hotpot_program_pilot --rows 300 --out runs/hotpot_program_pilot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from pathlib import Path

from bpto import Budget, CompletionCache, ModelConfig, Origin, Prompt, Tree, evaluate
from bpto.metrics import mean_metrics
from tasks.hotpotqa import AnswerWithReasoning, em_f1, load_or_fetch
from tasks.hotpotqa.program import ANSWER_PROMPT, SELECT_ROOT, make_program_task, paragraphs

from .compare import load_env

VARIANTS = {
    "root": SELECT_ROOT,
    "minimal": "{context}\n\nQuestion: {question}\nWhich of the paragraphs above are needed to answer the question? Return their titles.",
    "two_hop": (
        "Ten Wikipedia paragraphs (each begins with its title) and a question that needs two of them.\n"
        "Step 1: find the paragraph about the entity named in the question.\nStep 2: in that paragraph, find the "
        "bridging entity (a person, place, work, organisation or date) the question is really asking about, and "
        "find the paragraph about it.\nFor questions that compare two things, take the paragraph about each.\n"
        "Return the titles of the paragraphs you chose, exactly as written.\n\n"
        "Paragraphs:\n{context}\n\nQuestion: {question}"),
    "generous": (
        "Read the question, then pick every paragraph below that could contain a fact needed to answer it - err on "
        "the side of including a paragraph (up to four). Return their titles exactly as written.\n\n"
        "Paragraphs:\n{context}\n\nQuestion: {question}"),
    "strict_two": (
        "Exactly two of the paragraphs below are needed to answer the question; the other eight are distractors "
        "chosen to look relevant. Identify the two and return their titles exactly as written - no more, no fewer.\n\n"
        "Paragraphs:\n{context}\n\nQuestion: {question}"),
}


async def reference(client, cfg, data, which: str):
    """Run only the answerer on gold paragraphs (oracle) or the full context (all10)."""
    ans = Prompt(template=ANSWER_PROMPT)
    async def one(ex):
        if which == "oracle":
            paras = paragraphs(ex.inputs["context"]); sub = "\n\n".join(paras[t] for t in ex.meta["supporting_titles"] if t in paras)
        else:
            sub = ex.inputs["context"]
        comp = await client.complete(ans.render(context=sub, question=ex.inputs["question"]), config=cfg, schema=AnswerWithReasoning)
        a = comp.parsed; pred = (a.get("answer") if isinstance(a, dict) else getattr(a, "answer", "")) or ""
        em, f1 = em_f1(pred, ex.answer or "")
        return {"em": em, "f1": f1}
    rows = await asyncio.gather(*(one(ex) for ex in data))
    return mean_metrics(rows), [r["f1"] for r in rows]


async def main(args):
    load_env()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = load_or_fetch().sample(args.rows, seed=0)
    if args.mock:
        from tasks.hotpotqa.mock import _mock_client
        client = _mock_client(data)
    else:
        from bpto import BedrockClient
        client = BedrockClient(args.model, region=os.environ.get("AWS_REGION", "us-east-1"), max_concurrency=16,
                               cache=CompletionCache(out / "cache.jsonl"),
                               budget=Budget(max_calls=(2 * len(VARIANTS) + 2) * args.rows + 200, max_usd=args.max_usd))
    cfg = ModelConfig(max_tokens=512, temperature=0.0)
    task = make_program_task(client, data, config=cfg)
    tree = Tree(task)
    rows = []
    for name, tmpl in VARIANTS.items():
        node = tree.root if name == "root" else tree.add_child(tree.root, Prompt(template=tmpl), Origin(op="hand", params={"name": name}))
        ev = await evaluate(dataset=data).score(tree, node)
        per = [r.metrics.get("f1", 0.0) for r in ev.per_example]
        rows.append({"name": name, **{k: ev.metrics.get(k) for k in ("f1", "em", "sel_recall", "sel_precision", "n_selected")},
                     "errors": sum(1 for r in ev.per_example if r.error), "per": per})
        print(f"{name:11s} f1 {ev.metrics['f1']:.3f} em {ev.metrics['em']:.3f}  sel recall {ev.metrics['sel_recall']:.3f} prec {ev.metrics['sel_precision']:.3f} "
              f"n_sel {ev.metrics['n_selected']:.2f}  calls {client.usage.calls}", flush=True)
    for which in ("oracle", "all10"):
        m, per = await reference(client, cfg, data, which)
        rows.append({"name": which, "f1": m["f1"], "em": m["em"], "sel_recall": None, "sel_precision": None, "n_selected": None, "errors": 0, "per": per})
        print(f"{which:11s} f1 {m['f1']:.3f} em {m['em']:.3f}  calls {client.usage.calls}", flush=True)
    root = rows[0]
    lines = ["| variant | answer f1 | em | paired Δf1 vs root ± SE | sel recall | sel precision | n selected |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        d = [a - b for a, b in zip(r["per"], root["per"])]
        r["delta"], r["delta_se"] = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
        f = lambda v: "—" if v is None else f"{v:.3f}"
        lines.append(f"| {r['name']} | {r['f1']:.3f} | {r['em']:.3f} | {r['delta']:+.3f} ± {r['delta_se']:.3f} | {f(r['sel_recall'])} | {f(r['sel_precision'])} | {f(r['n_selected'])} |")
    text = "\n".join(lines) + "\n"
    (out / "pilot.md").write_text(text)
    (out / "pilot.jsonl").write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "per"}) for r in rows) + "\n")
    print(text)
    if not args.mock:
        print(f"spent ${client.budget.spent_usd:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--model", default="us.amazon.nova-micro-v1:0")
    ap.add_argument("--max-usd", type=float, default=0.5)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="runs/hotpot_program_pilot")
    asyncio.run(main(ap.parse_args()))
