"""Incumbent baseline: bpto's llm_judge on Nova Micro, same 43 items x 2 repeats, Budget(max_calls=90), no cache
(the repeat is the point), temperature 0 as every evaluation call. Every result logged; rerun skips logged keys."""
import asyncio, json, os, time
from pathlib import Path

from bpto import BedrockClient, Budget
from bpto.data import Example
from bpto.llm.base import ModelConfig
from bpto.scoring import ScoreContext, llm_judge

HERE = Path(__file__).parent
LOG = HERE / "nova_log.jsonl"
REPS = 2

for line in Path("~/projects/bpto/.env").expanduser().read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

budget = Budget(max_calls=90)
cfg = ModelConfig(model="us.amazon.nova-micro-v1:0", temperature=0.0, max_tokens=300)
qa_judge = llm_judge("1 if the model answer gives the same answer to the question as the reference (extra correct detail "
                     "or different wording is fine), 0 if it is a different or incomplete answer; partial credit between.",
                     config=cfg)
name_judge = llm_judge("1 if the model answer names the same real person as the reference, 0 if a different person.",
                       config=cfg)

done = set()
if LOG.exists():
    done = {(r["key"], r["rep"]) for r in map(json.loads, LOG.open())}


class _C:  # the scorer reads only .text
    def __init__(self, text): self.text = text


async def one(item, rep, client):
    if item["kind"] == "qa":
        ex, judge = Example(id=item["key"], inputs={"question": item["question"]}, answer=item["gold"]), qa_judge
    else:
        ex, judge = Example(id=item["key"], inputs={}, answer=item["gold"]), name_judge
    t0 = time.perf_counter()
    try:
        m = await judge(None, ex, _C(item["pred"]), ScoreContext(task=None, client=client, rendered_prompt=""))
        rec = {"key": item["key"], "rep": rep, "latency": time.perf_counter() - t0, "score": m["judge"]}
    except Exception as e:
        if type(e).__name__ == "BudgetExceeded":
            raise
        rec = {"key": item["key"], "rep": rep, "error": f"{type(e).__name__}: {e}"[:300]}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


async def main():
    items = json.load(open(HERE / "pilot_items.json"))
    for rep in range(REPS):  # a fresh client per repeat: ModelClient's default in-memory cache would replay rep 0
        client = BedrockClient("us.amazon.nova-micro-v1:0", region="us-east-1", budget=budget)
        await asyncio.gather(*(one(it, rep, client) for it in items if (it["key"], rep) not in done))
    print(f"calls {budget.spent.calls}, ${budget.spent_usd:.4f}")


asyncio.run(main())
