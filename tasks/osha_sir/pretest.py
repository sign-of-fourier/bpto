"""Phase 0a: does official GEPA run cleanly through bpto's clients? Offline, $0.

    python -m tasks.osha_sir.pretest [--out runs/osha_sir/pretest]

A mock 13-class task where the "model" only gets a class right once the instruction carries a rule for that class's
cue word, and a mock reflector that reads GEPA's feedback and adds one rule per rewrite (so the search can climb).
Faults are injected deterministically (hash of the prompt): transient Bedrock-style errors on task calls, reflector
replies without a ``` block, rewrites that drop the {narrative} slot. Checks, each PASS/FAIL/NOTE:

  C1 runs to completion and climbs          C5 q > 1 (IndependentSampling) fans out: in-flight, batched reflection
  C2 GEPA's metric count == our billed count C6 transient errors score 0 and the run continues
  C3 overshoot past max_metric_calls         C7 BudgetExceeded from the task client stops the run (propagates)
  C4 same seed -> same run; resume           C8 dropped {narrative} slots are restored and counted
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from bpto import Budget, MockClient, ModelConfig
from bpto.llm.base import BudgetExceeded

from .common import SLOT
from .official_gepa import Row, run_official

LABELS = ["Falls to lower level", "Falls on same level", "Struck by object or equipment",
          "Struck against object or equipment", "Caught in or compressed by equipment or objects",
          "Pedestrian vehicular incident", "Roadway incident involving motorized land vehicle",
          "Exposure to temperature extremes", "Exposure to electricity", "Fires", "Overexertion",
          "Repetitive motion", "Other"]
CUES = [f"zq{i}" for i in range(len(LABELS))]
SEED_INSTRUCTION = ("Classify this injury narrative into one of: " + "; ".join(LABELS)
                    + ". Answer with the label only.\n\nNarrative: " + SLOT)


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def make_rows(n: int, offset: int) -> list[Row]:
    rows = []
    for i in range(n):
        k = (i * 7 + offset) % len(LABELS)
        rows.append(Row(id=f"r{offset + i}", label=LABELS[k],
                        narrative=f"Employee {offset + i} was hurt at the site; the report notes {CUES[k]} near the work area."))
    return rows


class Faults:
    def __init__(self, task_err_mod: int = 41, garbage_mod: int = 9, drop_slot_mod: int = 7):
        self.task_err_mod, self.garbage_mod, self.drop_slot_mod = task_err_mod, garbage_mod, drop_slot_mod
        self.task_errors = self.garbage = self.dropped = 0


def task_handler(faults: Faults, inflight: dict):
    async def handler(prompt, cfg, schema):
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        try:
            await asyncio.sleep(0.002)
            if faults.task_err_mod and _h(prompt) % faults.task_err_mod == 0:
                faults.task_errors += 1
                raise RuntimeError("ResourceNotFoundException: Inference Profile ARN not found")
            cue = re.findall(r"zq\d+", prompt)[-1]   # the narrative is rendered last
            k = CUES.index(cue)
            if f"mentions {cue}," in prompt:
                ans = LABELS[k]
            else:
                ans = "Other"
            style = _h(prompt) % 3                   # exercise the parser
            return [ans, f"Category: {ans}", f"The narrative describes an incident.\n{ans}"][style]
        finally:
            inflight["now"] -= 1
    return handler


def reflect_handler(faults: Faults, sizes: list):
    def handler(prompt, cfg, schema):
        current = re.search(r"```\n(.*?)\n```", prompt, re.S).group(1)
        golds = re.findall(r"gold was ([^;.\n]+)", prompt)
        cues_by_gold = {}
        for block in re.split(r"# Example \d+", prompt)[1:]:
            g = re.search(r"gold was ([^;.\n]+)", block)
            c = re.search(r"zq\d+", block)
            if g and c:
                cues_by_gold[g.group(1).strip()] = c.group(0)
        missing = [g.strip() for g in golds if g.strip() in cues_by_gold and f"mentions {cues_by_gold[g.strip()]},"
                   not in current]
        new = current
        if missing:
            g = max(set(missing), key=missing.count)
            new = current + f"\nIf the narrative mentions {cues_by_gold[g]}, the category is {g}."
        h = _h(prompt)
        if faults.drop_slot_mod and h % faults.drop_slot_mod == 0:
            faults.dropped += 1
            new = new.replace("\n\nNarrative: " + SLOT, "").replace(SLOT, "")
        if faults.garbage_mod and h % faults.garbage_mod == 0:
            faults.garbage += 1
            return "Here is my improved instruction: " + new.replace("\n", " ")   # no ``` block
        return f"Sure.\n```\n{new}\n```"
    return handler


def one_run(*, B: int, seed: int, minibatch: int = 5, n_train: int = 60, n_val: int = 40, run_dir: str | None = None,
            sampling=None, count: str = "requested", task_cap: int | None = None, max_concurrency: int = 16,
            faults: Faults | None = None):
    faults = faults or Faults()
    inflight = {"now": 0, "max": 0}
    sizes: list[int] = []
    task = MockClient(task_handler(faults, inflight), max_concurrency=max_concurrency,
                      budget=Budget(max_calls=task_cap) if task_cap else None,
                      default_config=ModelConfig(model="mock-task", temperature=0.0))
    refl = MockClient(reflect_handler(faults, sizes), max_concurrency=max_concurrency,
                      default_config=ModelConfig(model="mock-reflect", temperature=1.0))
    t0 = time.perf_counter()
    err = None
    try:
        result, adapter, lm = run_official(SEED_INSTRUCTION, make_rows(n_train, 0), make_rows(n_val, 1000), LABELS,
                                           task_client=task, reflect_client=refl, max_metric_calls=B,
                                           minibatch=minibatch, seed=seed, run_dir=run_dir,
                                           sampling_strategy=sampling, count=count)
    except BudgetExceeded as e:
        result = adapter = lm = None
        err = e
    return {"result": result, "adapter": adapter, "lm": lm, "task": task, "refl": refl, "faults": faults,
            "inflight": inflight, "secs": time.perf_counter() - t0, "error": err}


def summary(r) -> dict:
    res, ad = r["result"], r["adapter"]
    out = {"secs": round(r["secs"], 2), "task_calls_billed_by_client": r["task"].usage.calls,
           "task_cache_hits": r["task"].usage.cache_hits, "reflect_calls": r["refl"].usage.calls,
           "max_inflight": r["inflight"]["max"], "faults": {k: v for k, v in vars(r["faults"]).items()
                                                            if not k.endswith("_mod")}}
    if res is not None:
        out.update(total_metric_calls=res.total_metric_calls, candidates=len(res.candidates),
                   full_val_evals=res.num_full_val_evals, seed_val=round(res.val_aggregate_scores[0], 3),
                   best_val=round(max(res.val_aggregate_scores), 3), rows_requested=ad.stats.requested,
                   rows_billed=ad.stats.billed, rows_cached=ad.stats.cached, row_errors=ad.stats.errors,
                   evals_slot_restored=ad.stats.slot_restored, stalled=ad.stall.fired,
                   reflection_failures=sum("reflection failed" in l.lower() for l in getattr(ad.logger, "lines", [])),
                   max_rows_per_engine_step=max(sum(e["rows"] for e in ad.stats.evaluations if e["batch"] == b)
                                                for b in {e["batch"] for e in ad.stats.evaluations}))
    if r["error"] is not None:
        out["error"] = f"{type(r['error']).__name__}: {r['error']}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/osha_sir/pretest")
    ap.add_argument("--B", type=int, default=1500)
    args = ap.parse_args(argv)
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    from gepa.strategies.proposal_sampling import IndependentSampling

    B = args.B
    checks, runs = [], {}

    def check(name, ok, detail):
        status = "PASS" if ok is True else "FAIL" if ok is False else "NOTE"
        checks.append({"check": name, "status": status, "detail": detail})
        print(f"[{status}] {name}: {detail}")

    base = one_run(B=B, seed=0)
    runs["base"] = s = summary(base)
    check("C1 completes and climbs", s["best_val"] > s["seed_val"],
          f"val {s['seed_val']} -> {s['best_val']}, {s['candidates']} candidates, {s['reflect_calls']} reflections")
    check("C2 GEPA count == rows requested; billed == client", s["total_metric_calls"] == s["rows_requested"]
          and s["rows_billed"] == s["task_calls_billed_by_client"],
          f"GEPA {s['total_metric_calls']} = requested {s['rows_requested']}; billed {s['rows_billed']} = client "
          f"{s['task_calls_billed_by_client']} ({s['rows_cached']} cache hits)")
    check("C3 overshoot", None, f"{s['total_metric_calls'] - B:+d} calls past max_metric_calls={B} "
          "(engine checks the budget only between iterations)")

    bil = one_run(B=B, seed=0, count="billed")
    runs["billed"] = sb = summary(bil)
    check("C2b count=billed is a mixed unit", None,
          f"GEPA {sb['total_metric_calls']} vs billed {sb['rows_billed']}: the engine takes num_metric_calls on "
          f"minibatches but charges validation passes per row, cached or not")

    again = one_run(B=B, seed=0)
    runs["same_seed"] = s2 = summary(again)
    same = (again["result"].candidates == base["result"].candidates
            and s2["total_metric_calls"] == s["total_metric_calls"])
    check("C4a same seed -> same run", same, f"candidates identical: {again['result'].candidates == base['result'].candidates}, "
          f"calls {s['total_metric_calls']} vs {s2['total_metric_calls']}")

    rd = str(out / "resume_dir")
    first = one_run(B=B // 2, seed=0, run_dir=rd)
    second = one_run(B=B, seed=0, run_dir=rd)
    runs["resume_first"], runs["resume_second"] = summary(first), summary(second)
    check("C4b resume from run_dir", second["result"].total_metric_calls >= first["result"].total_metric_calls
          and len(second["result"].candidates) >= len(first["result"].candidates),
          f"first leg {first['result'].total_metric_calls} calls / {len(first['result'].candidates)} cands, "
          f"resumed to {second['result'].total_metric_calls} / {len(second['result'].candidates)}; "
          f"uninterrupted {s['total_metric_calls']} / {s['candidates']}; same final candidates as uninterrupted: "
          f"{second['result'].candidates == base['result'].candidates}")

    q4 = one_run(B=B, seed=0, sampling=IndependentSampling(4))
    runs["independent_q4"] = s4 = summary(q4)
    check("C5 q=4 fans out", s4["max_inflight"] > 5 and s4["max_rows_per_engine_step"] > 5,
          f"max in flight {s4['max_inflight']} (q=1: {s['max_inflight']}), max rows per engine step "
          f"{s4['max_rows_per_engine_step']} (q=1: {s['max_rows_per_engine_step']}), {s4['secs']}s vs {s['secs']}s")

    check("C6 transient errors", s["row_errors"] > 0 and s["best_val"] > s["seed_val"],
          f"{s['row_errors']} failed rows scored 0, run continued")

    capped = one_run(B=B, seed=0, task_cap=B // 3)
    runs["task_cap"] = sc = summary(capped)
    check("C7 BudgetExceeded propagates", capped["error"] is not None and sc["task_calls_billed_by_client"] <= B // 3,
          f"{sc.get('error', 'no exception')}; client billed {sc['task_calls_billed_by_client']} <= cap {B // 3}")

    check("C8 dropped slot restored", s["faults"]["dropped"] == 0 or s["evals_slot_restored"] > 0,
          f"reflector dropped the slot {s['faults']['dropped']}x; {s['evals_slot_restored']} evaluations ran with it restored; "
          f"reflector replies without ``` {s['faults']['garbage']}x (GEPA takes the whole reply as the instruction)")

    # --- the q-EI sampler in GEPA's expand seat (HashEmbedder offline; Titan live) ---
    from bpto.bo import HashEmbedder
    from .qei_sampling import QEISampling

    def qei(seed):
        return QEISampling(4, HashEmbedder(), seed=seed)

    smp = qei(0)
    qr = one_run(B=B, seed=0, sampling=smp)
    runs["qei_q4"] = sq = summary(qr)
    modes = [e["mode"] for e in smp.log]
    fits = [e["fit"] for e in smp.log if e["mode"] == "qei"]
    check("C9a q-EI completes, counts match", sq["total_metric_calls"] == sq["rows_requested"]
          and sq["best_val"] > sq["seed_val"],
          f"val {sq['seed_val']} -> {sq['best_val']}, {sq['candidates']} candidates, GEPA {sq['total_metric_calls']} = "
          f"requested {sq['rows_requested']} (billed {sq['rows_billed']})")
    check("C9b always q proposals", all(len(e["parents"]) == 4 for e in smp.log) and "qei" in modes,
          f"{len(smp.log)} iterations: {modes.count('warmup')} warmup, {modes.count('qei')} q-EI; "
          f"{sum(len(set(e['parents'])) for e in smp.log if e['mode'] == 'qei')} distinct parents over the q-EI iterations")
    flat = sum(f.get("flat", False) for f in fits)
    differs = sum(set(f["batch"]["ids"]) != set(f["batch"]["top_k_ids"]) for f in fits if "batch" in f)
    check("C9c surrogate is informative", flat < len(fits) / 2 if fits else False,
          f"flat acquisition in {flat}/{len(fits)} fits; joint pick differs from independent top-q in {differs}/"
          f"{sum('batch' in f for f in fits)}; last fit: n_train {fits[-1]['n_train'] if fits else '-'}, "
          f"inputs {fits[-1]['n_inputs'] if fits else '-'}, pca {fits[-1].get('pca') if fits else '-'}")
    smp2 = qei(0)
    qr2 = one_run(B=B, seed=0, sampling=smp2)
    check("C9d q-EI same seed -> same run", qr2["result"].candidates == qr["result"].candidates,
          f"parents identical: {[e['parents'] for e in smp2.log] == [e['parents'] for e in smp.log]}")

    race = {}
    for arm in ("q1", "independent_q4", "qei_q4"):
        race[arm] = []
        for sd in range(3):
            smp_ = None if arm == "q1" else IndependentSampling(4) if arm == "independent_q4" else qei(sd)
            rr = summary(one_run(B=B, seed=sd, sampling=smp_))
            race[arm].append({"best_val": rr["best_val"], "calls": rr["total_metric_calls"], "secs": rr["secs"],
                              "iterations": rr["reflect_calls"]})
    runs["mock_race"] = race
    check("C10 mock race (plumbing only, not evidence)", None, "; ".join(
        f"{a}: best val {[x['best_val'] for x in v]}, reflections {[x['iterations'] for x in v]}" for a, v in race.items()))
    (out / "qei_log.json").write_text(json.dumps(smp.log, indent=1, default=str))

    best = base["result"].best_candidate["instruction"]
    (out / "best_instruction.txt").write_text(best)
    (out / "pretest.json").write_text(json.dumps({"checks": checks, "runs": runs}, indent=2))
    print(f"\n{sum(c['status'] == 'PASS' for c in checks)} pass, {sum(c['status'] == 'FAIL' for c in checks)} fail, "
          f"{sum(c['status'] == 'NOTE' for c in checks)} notes -> {out}/pretest.json")


if __name__ == "__main__":
    main()
