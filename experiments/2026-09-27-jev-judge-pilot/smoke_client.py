"""Live smoke of bpto.JevClient: re-ask the pilot's rep-0 questions through the client, compare to the SDK answers."""
import asyncio, json, os, sys
from pathlib import Path

from bpto import Budget, CompletionCache, JevClient
from bpto.llm.jev import noul, score

HERE = Path(__file__).parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 43
for line in Path("~/projects/jev/.env").expanduser().read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

QA = {"correct": noul("Does the model answer give the same answer to the question as the reference answer? "
                      "Extra correct detail or different wording is fine; a different or incomplete answer is not."),
      "grade": score("How well does the model answer match the reference answer to the question?",
                     ["Wrong or unrelated answer.",
                      "Partly right: overlaps the reference but misses or contradicts part of it.",
                      "Right answer, buried in extra text or missing minor detail.",
                      "Exactly the reference answer, possibly reworded."])}
NAME = {"same": noul("Do these two names refer to the same real person?")}


async def main():
    items = json.load(open(HERE / "pilot_items.json"))[:N]
    sdk = {r["key"]: r for r in map(json.loads, open(HERE / "jev_log.jsonl")) if r["rep"] == 0}
    c = JevClient("jev-1.13", base_url="https://openrouter.ai/api", api_key=os.environ["OPENROUTER_API_KEY"],
                  budget=Budget(max_calls=45), cache=CompletionCache("runs/jev_smoke/cache.jsonl"))

    async def one(it):
        if it["kind"] == "qa":
            a = await c.ask({"question": it["question"], "reference_answer": it["gold"], "model_answer": it["pred"]}, QA)
        else:
            a = await c.ask({"name_a": it["gold"], "name_b": it["pred"]}, NAME)
        return it["key"], a

    res = await asyncio.gather(*(one(it) for it in items))
    diffs = [abs(a[q]["noul"] - sdk[k]["answers"][q]["noul"]) for k, a in res for q in a if q != "grade"]
    print(f"calls {c.usage.calls} cache_hits {c.usage.cache_hits} tokens {c.usage.input_tokens}/{c.usage.output_tokens}")
    print(f"noul |client - sdk| max {max(diffs):.3f} mean {sum(diffs)/len(diffs):.4f} over {len(diffs)}")
    await c.aclose()


asyncio.run(main())
