"""Run the lead-qualification use case through the studio's own code, without touching studio/data.

STUDIO_DATA is pointed at ./studio_data before the studio is imported, so the sqlite db, run trees, caches and
events land here. The spec, validation, pilot, cost projection and run loop are the studio's (app.validation,
app.runs), called the way the /pilot and /runs endpoints call them.

    python run_usecase.py pilot                 # static checks + pilot (a fraction of a cent) + projected cost
    python run_usecase.py run --max-usd 3       # the optimization run; hard-stops at the budget
    python run_usecase.py run --max-usd 1.5 --minibatch 15 --patience 8 --rounds 15   # discovery+ settings
    python run_usecase.py show [run_id]         # summary of a finished run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ["STUDIO_DATA"] = str(HERE / "studio_data")
sys.path.insert(0, str(ROOT / "studio"))
for envf in (ROOT / ".env", ROOT / "studio" / ".env"):   # credentials for Bedrock; never printed
    if envf.exists():
        for line in envf.read_text().splitlines():
            m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))

from app import db  # noqa: E402  (after STUDIO_DATA)
from app.clients import Access, make_client, prices  # noqa: E402
from app.compile import build_task  # noqa: E402
from app.datasets import parse_upload, to_dataset  # noqa: E402
from app.models import ProjectSpec  # noqa: E402
from app.runs import RUNS_DIR, RunManager  # noqa: E402
from app.validation import pilot as run_pilot, project_cost, validate_static  # noqa: E402

# ---- edit mode: fix #2 from the bpto review (editing reflection), patched in for one run; studio/ stays untouched ----
EDIT_REFLECT_PROMPT = (
    "I gave an assistant the following prompt template to perform a task. The template is supposed to: "
    "{description}\nIt must keep these placeholders exactly, in curly braces: {placeholders}\n\n"
    "Current prompt template:\n<prompt>\n{prompt}\n</prompt>\n\n"
    "Here are examples of task inputs, the assistant's response under this prompt, and feedback on each:\n\n"
    "{directive}\n"
    "Read the feedback carefully and decide which existing rule or sentence in the template produced these failures. "
    "Prefer changing that text: narrow it, rewrite it, reorder it or delete it. Add a new rule only if no existing rule "
    "covers the case. When a template applies its rules in order and the first match wins, a later addition cannot "
    "override an earlier rule, so fix the earlier rule itself. Copy every line you are not changing exactly, including "
    "its line breaks. Then write {n} improved prompt template(s). Each must be complete and usable on its own.{seed}"
)


def enable_edit_mode():
    import app.compile as C

    C.REFLECT_PROMPT = EDIT_REFLECT_PROMPT   # the prompt-aware critic (fix #1) is now in the studio itself


def enable_gate_rungs(extend, margin=1.0):
    """bpto's opt-in gate (PR #1): unclear children go on to larger batches before a full evaluation is bought.
    Swaps the studio schedule's minibatch + full steps for `gate_steps`, keeping the step names."""
    from bpto.gepa.loop import gate_steps
    import app.compile as C
    import app.runs as R
    base = C.build_schedule

    def build_schedule(spec, task, **kw):
        schedule, stop = base(spec, task, **kw)
        o = spec.optimizer

        def sched(tree, r):
            steps = schedule(tree, r)
            if len(steps) < 3:          # round 0: root only
                return steps
            return steps[:-2] + gate_steps(r, task.dataset, o.minibatch, o.seed, extend=extend, margin=margin, tag=f"r{r}")
        return sched, stop
    C.build_schedule = R.build_schedule = build_schedule


LABELS = ["accepted", "rejected_fit", "rejected_no_intent", "existing_customer", "partner_route"]
LABEL_COL = "sdr_disposition"
EVAL_MODEL = "us.amazon.nova-micro-v1:0"      # studio defaults: Micro evaluates, Lite reflects
REFLECT_MODEL = "us.amazon.nova-lite-v1:0"


def load():
    rows = parse_upload("impromptune_ready.csv", (HERE / "impromptune_ready.csv").read_bytes())
    for r in rows:
        r["id"] = r["lead_id"]
    template = (HERE / "baseline_prompt.txt").read_text()
    placeholders = sorted(set(re.findall(r"\{([a-z_]+)\}", template)))
    input_map = {p: p for p in placeholders}          # sdr_notes_len is in the file and deliberately unmapped
    return rows, template, input_map


def make_spec(template, max_usd=3.0, rounds=12, minibatch=5, patience=4, reflect_model=REFLECT_MODEL,
              eval_model=EVAL_MODEL, balanced=False):
    return ProjectSpec.model_validate({
        "name": "Lead qualification (synthetic)",
        "modules": [{"id": "qualify", "template": template,
                     "description": "routes an inbound B2B lead to one of five dispositions",
                     "schema_fields": [{"name": "disposition", "type": "string", "description": "", "enum": LABELS}],
                     "max_tokens": 64}],
        "eval_model": eval_model,
        "evaluate": {"label_column": LABEL_COL,
                     "scorers": [{"type": "exact_match", "field": "disposition", "normalize": True, "balanced": balanced}],
                     "objective": {"accuracy_balanced" if balanced else "accuracy": 1.0}},
        "optimizer": {"goal": "accuracy", "engine": "gepa", "rounds": rounds, "no_improvement_rounds": patience,
                      "minibatch": minibatch,
                      "max_usd": max_usd, "reflect_model": reflect_model, "holdout_frac": 0.2, "seed": 0},
    })


def access():
    return Access(house_keys=True, house_models=None, max_concurrency=8, tier="local")


async def cmd_pilot(a):
    rows, template, input_map = load()
    spec = make_spec(template)
    rep = validate_static(spec, rows, input_map)
    for i in rep.issues:
        print(f"[{i.level}] {i.stage} {i.where}: {i.message}")
    if not rep.ok:
        sys.exit("static validation failed")
    from bpto import Budget
    client = make_client(spec.eval_model, budget=Budget(max_calls=a.rows * 3 * spec.max_steps + 20, prices=prices()),
                         access=access(), purpose="pilot")
    task = build_task(spec, to_dataset(rows, input_map, LABEL_COL), client)
    out = await run_pilot(spec, task, rows=a.rows, rep=rep)
    cost = project_cost(spec, len(rows), avg_steps=out.get("avg_steps"), tokens_per_module=out.get("tokens_per_module") or None,
                        avg_in=out.get("avg_input_tokens"), avg_out=out.get("avg_output_tokens"))
    (HERE / "studio_data").mkdir(exist_ok=True)
    (HERE / "studio_data" / "pilot.json").write_text(json.dumps({"pilot": out, "cost": cost}, indent=2, default=str))
    print(json.dumps({"pilot": out, "cost": cost}, indent=2, default=str))


async def cmd_run(a):
    rows, template, input_map = load()
    spec = make_spec(template, max_usd=a.max_usd, rounds=a.rounds, minibatch=a.minibatch, patience=a.patience,
                     reflect_model=a.reflect_model, eval_model=a.eval_model,
                     balanced=a.balanced)
    rep = validate_static(spec, rows, input_map)
    if not rep.ok:
        sys.exit("static validation failed")
    pf = HERE / "studio_data" / "pilot.json"
    pilot = json.loads(pf.read_text())["pilot"] if pf.exists() else None
    if a.edit_mode:
        enable_edit_mode()
    if a.extend:
        enable_gate_rungs(tuple(int(x) for x in a.extend.split(",")))
    con = db.connect()
    rid = db.new_id()
    if a.cache_from:   # same model calls as an earlier run are free and identical (the root evaluation)
        (RUNS_DIR / rid).mkdir(parents=True, exist_ok=True)
        shutil.copy(RUNS_DIR / a.cache_from / "cache.jsonl", RUNS_DIR / rid / "cache.jsonl")
    con.execute("insert into runs (id, project_id, dataset_id, status, dir, started, finished, summary, error, spec)"
                " values (?,?,?,?,?,?,?,?,?,?)",
                (rid, "lead-qual", "impromptune_ready", "running", str(RUNS_DIR / rid), db.now(), None, None, None,
                 spec.model_dump_json()))
    con.commit()
    print(f"run {rid} -> {RUNS_DIR / rid}", flush=True)
    await RunManager()._run(con, rid, spec, rows, input_map, LABEL_COL, False, pilot, access())
    show(rid)


def show(rid=None):
    runs = sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime) if RUNS_DIR.exists() else []
    d = RUNS_DIR / rid if rid else (runs[-1] if runs else None)
    if not d:
        sys.exit("no runs")
    st = json.loads((d / "status.json").read_text())
    print(json.dumps({k: st.get(k) for k in ("state", "error", "round", "spent_usd", "train_rows", "holdout_rows",
                                             "nondeterminism_band", "accepted", "proposed")}, indent=2, default=str))
    s = st.get("summary") or {}
    print(json.dumps({k: s.get(k) for k in ("stopped_because", "rounds", "root_score", "holdout", "spent_usd")},
                     indent=2, default=str))
    if s.get("best"):
        print("\n--- best prompt ---\n" + s["best"]["modules"]["qualify"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot"); p.add_argument("--rows", type=int, default=12)
    r = sub.add_parser("run"); r.add_argument("--max-usd", type=float, required=True); r.add_argument("--rounds", type=int, default=12)
    r.add_argument("--minibatch", type=int, default=5); r.add_argument("--patience", type=int, default=4)
    r.add_argument("--reflect-model", default=REFLECT_MODEL); r.add_argument("--eval-model", default=EVAL_MODEL)
    r.add_argument("--edit-mode", action="store_true", help="editing reflection prompt instead of bpto's additive one")
    r.add_argument("--balanced", action="store_true", help="objective on balanced accuracy (mean per-class recall)")
    r.add_argument("--extend", help="gate rungs for unclear children, e.g. 60,120 (bpto gate_steps)")
    r.add_argument("--cache-from", help="run id whose cache.jsonl seeds this run")
    s = sub.add_parser("show"); s.add_argument("run_id", nargs="?")
    a = ap.parse_args()
    if a.cmd == "pilot":
        asyncio.run(cmd_pilot(a))
    elif a.cmd == "run":
        asyncio.run(cmd_run(a))
    else:
        show(a.run_id)


if __name__ == "__main__":
    main()
