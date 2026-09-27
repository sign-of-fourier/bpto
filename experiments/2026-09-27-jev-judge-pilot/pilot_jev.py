"""Jev pilot: 43 items x 2 repeats, hard cap 90 calls, every response logged (rerun skips logged keys)."""
import asyncio, json, os, time
from pathlib import Path
from typesafe_sdk import AsyncTypeSafeClient, Noul, Score

HERE = Path(__file__).parent
LOG = HERE / "jev_log.jsonl"
MAX_CALLS = int(os.environ.get("PILOT_MAX", 90))
REPS = 2

for line in Path("~/projects/jev/.env").expanduser().read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

QA_QUESTIONS = {
    "correct": Noul(instructions="Does the model answer give the same answer to the question as the reference answer? "
                                 "Extra correct detail or different wording is fine; a different or incomplete answer is not."),
    "grade": Score(instructions="How well does the model answer match the reference answer to the question?",
                   criteria=["Wrong or unrelated answer.",
                             "Partly right: overlaps the reference but misses or contradicts part of it.",
                             "Right answer, buried in extra text or missing minor detail.",
                             "Exactly the reference answer, possibly reworded."]),
}
NAME_QUESTIONS = {
    "same": Noul(instructions="Do these two names refer to the same real person?"),
}

done = set()
if LOG.exists():
    done = {(r["key"], r["rep"]) for r in map(json.loads, LOG.open())}
calls = 0


async def one(client, sem, item, rep):
    global calls
    async with sem:
        if calls >= MAX_CALLS:
            return
        calls += 1
        if item["kind"] == "qa":
            state = {"question": item["question"], "reference_answer": item["gold"], "model_answer": item["pred"]}
            qs = QA_QUESTIONS
        else:
            state = {"name_a": item["gold"], "name_b": item["pred"]}
            qs = NAME_QUESTIONS
        t0 = time.perf_counter()
        try:
            r = await client.system_one(state=state, questions=qs)
            rec = {"key": item["key"], "rep": rep, "latency": time.perf_counter() - t0, "model": r.model,
                   "usage": r.usage.model_dump(), "answers": {k: a.model_dump() for k, a in r.answers.items()}}
        except Exception as e:  # record, don't retry
            rec = {"key": item["key"], "rep": rep, "error": f"{type(e).__name__}: {e}"[:300]}
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")


async def main():
    items = json.load(open(HERE / "pilot_items.json"))
    client = AsyncTypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api")
    sem = asyncio.Semaphore(8)
    todo = [(it, rep) for rep in range(REPS) for it in items if (it["key"], rep) not in done]
    print(f"{len(todo)} calls to make (cap {MAX_CALLS})")
    # rep 0 then rep 1, so the repeats are separate requests, not concurrent duplicates
    for rep in range(REPS):
        await asyncio.gather(*(one(client, sem, it, r) for it, r in todo if r == rep))
    print(f"made {calls} calls")


asyncio.run(main())
