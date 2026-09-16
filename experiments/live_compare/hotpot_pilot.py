"""Headroom pilot for HotpotQA on Nova Micro before spending on optimizers: root + hand-written variants on the
same rows. If the spread between variants is inside noise there is nothing for a prompt search to find (the
IFBench lesson). ~6 x 300 = 1,800 calls, capped.

    python -m experiments.live_compare.hotpot_pilot --rows 300 --out runs/hotpot_pilot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from pathlib import Path

from bpto import Budget, CompletionCache, ModelConfig, Prompt, Tree, evaluate
from tasks.hotpotqa import ROOT_PROMPT, load_or_fetch, make_task

from .compare import load_env

VARIANTS = {
    "root": ROOT_PROMPT,
    "minimal": "{context}\n\nQuestion: {question}\nGive only the exact answer.",
    "cot": (
        "Read the paragraphs and answer the question. Before answering, briefly identify which two paragraphs "
        "are relevant and quote the key fact from each; then give the final answer as the shortest exact span "
        "from the text (or yes / no).\n\nParagraphs:\n{context}\n\nQuestion: {question}"),
    "two_hop": (
        "You will answer a two-hop question from ten paragraphs, eight of which are irrelevant distractors.\n"
        "Step 1: find the paragraph about the entity named in the question.\nStep 2: find the second paragraph "
        "that it points to (a person, place, work or date mentioned in step 1).\nStep 3: combine the two facts "
        "into the answer. For questions comparing two things, look up both and compare.\nWrite the steps briefly, "
        "then the final answer: a short exact span copied from the text, or yes / no.\n\n"
        "Paragraphs:\n{context}\n\nQuestion: {question}"),
    "persona": (
        "You are a meticulous encyclopedia fact-checker. A wrong answer will be published, so verify every claim "
        "against the paragraphs below and never guess from memory. Use only the paragraphs. Answer with the "
        "shortest exact phrase from the text, a date or number exactly as written, or yes / no.\n\n"
        "Paragraphs:\n{context}\n\nQuestion: {question}"),
    "span_rules": (
        "Answer the question from the paragraphs. Rules: (1) the answer is a span copied verbatim from a "
        "paragraph, not a sentence - no explanation, no 'the answer is'; (2) questions starting with is/are/was/"
        "were/do/does/did get exactly 'yes' or 'no'; (3) for names give the full name as written; for dates and "
        "numbers copy the text's form; (4) ignore paragraphs unrelated to the question.\n\n"
        "Paragraphs:\n{context}\n\nQuestion: {question}"),
}


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
                               budget=Budget(max_calls=len(VARIANTS) * args.rows + 100, max_usd=args.max_usd))
    task = make_task(client, data, config=ModelConfig(max_tokens=args.max_tokens, temperature=0.0), reasoning=args.reasoning)
    tree = Tree(task)
    rows = []
    for name, tmpl in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        node = tree.root if name == "root" else tree.add_child(tree.root, Prompt(template=tmpl), __import__("bpto").Origin(op="hand", params={"name": name}))
        ev = await evaluate(dataset=data).score(tree, node)
        per = [r.metrics.get("f1", 0.0) for r in ev.per_example]
        errs = sum(1 for r in ev.per_example if r.error)
        se = statistics.stdev(per) / len(per) ** 0.5
        rows.append({"name": name, "f1": ev.metrics["f1"], "em": ev.metrics["em"], "se": se, "errors": errs,
                     "out_tokens": ev.metrics.get("output_tokens"), "per_example": {r.example_id: r.metrics.get("f1", 0.0) for r in ev.per_example}})
        print(f"{name:11s} f1 {ev.metrics['f1']:.3f} ± {se:.3f}  em {ev.metrics['em']:.3f}  out_tok {ev.metrics.get('output_tokens', 0):.0f}  "
              f"errors {errs}  calls {client.usage.calls}  ${client.budget.spent_usd:.3f}" if not args.mock else f"{name} f1 {ev.metrics['f1']:.3f}", flush=True)
    root = rows[0]
    lines = ["| variant | f1 | em | paired Δf1 vs root ± SE | out tokens | errors |", "|---|---|---|---|---|---|"]
    for r in rows:
        d = [r["per_example"][i] - root["per_example"][i] for i in root["per_example"]]
        r["delta"], r["delta_se"] = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
        lines.append(f"| {r['name']} | {r['f1']:.3f} | {r['em']:.3f} | {r['delta']:+.3f} ± {r['delta_se']:.3f} | {r['out_tokens']:.0f} | {r['errors']} |")
    text = "\n".join(lines) + "\n"
    (out / "pilot.md").write_text(text)
    (out / "pilot.jsonl").write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "per_example"}) for r in rows) + "\n")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--model", default="us.amazon.nova-micro-v1:0")
    ap.add_argument("--max-usd", type=float, default=0.3)
    ap.add_argument("--reasoning", action="store_true", help="schema with a reasoning field before the answer")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="runs/hotpot_pilot")
    asyncio.run(main(ap.parse_args()))
