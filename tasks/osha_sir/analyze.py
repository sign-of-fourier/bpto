"""Collect the main runs: per arm and seed, the chosen prompt's validation and holdout scores, wall-clock, calls.

    python -m tasks.osha_sir.analyze [--tag main_] [--score-holdout]

--score-holdout scores every run's chosen prompt plus the reference prompts on the holdout (2 repeats each, fresh
calls per repeat) through `run.py score`; without it the table reads existing holdout scores.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from statistics import mean, pstdev

from .run import PROMPTS, RUNS

ARMS = ["q1", "independent4", "qei4"]
REFERENCE = ["B0", "B1", "B2", "onestep_lite", "onestep_strong"]


def runs(tag: str) -> dict[str, list[dict]]:
    out = {a: [] for a in ARMS}
    for d in sorted((RUNS / "gepa").glob(f"{tag}*_s*")):
        m = re.fullmatch(re.escape(tag) + r"(\w+?)_s(\d+)", d.name)
        if not m or not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        log = (d / "gepa_log.txt").read_text() if (d / "gepa_log.txt").exists() else ""
        s["dir"] = str(d)
        s["gate_passed"] = log.count("Accepted candidate")
        s["gate_ties"] = len(re.findall(r"New subsample score (\S+) is not better than old score \1", log))
        s["gate_rejected"] = log.count("is not better than")
        out.setdefault(m.group(1), []).append(s)
    return out


def calls_to_reach(s: dict, level: float) -> int | None:
    """GEPA metric calls at which the first candidate with validation score >= level was discovered."""
    for v, c in zip(s.get("val_scores", []), s.get("discovery_calls", [])):
        if v >= level:
            return c
    return None


def export_best(tag: str, rs: dict) -> list[str]:
    names = []
    for arm, lst in rs.items():
        for s in lst:
            name = f"{tag}{arm}_s{s['seed']}"
            src = Path(s["dir"]) / "best_prompt.txt"
            if src.exists():
                shutil.copy(src, PROMPTS / f"{name}.txt")
                names.append(name)
    return names


def holdout_acc(name: str) -> tuple[float | None, float | None]:
    p = RUNS / "scores" / "holdout" / "summary.json"
    if not p.exists():
        return None, None
    s = json.loads(p.read_text()).get(name)
    if not s:
        return None, None
    return mean(r["acc"] for r in s["reps"]), mean(r["macro_f1"] for r in s["reps"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main_")
    ap.add_argument("--score-holdout", action="store_true")
    args = ap.parse_args(argv)
    rs = runs(args.tag)
    names = export_best(args.tag, rs)
    if args.score_holdout:
        from .run import main as run_main
        run_main(["score", *REFERENCE, *names, "--split", "holdout", "--repeats", "2", "--max-usd", "1.0"])

    b2_val = mean(r["acc"] for r in json.loads((RUNS / "scores/val/summary.json").read_text())["B2"]["reps"])
    rows, table = [], {}
    for arm in ARMS:
        lst = sorted(rs.get(arm, []), key=lambda s: s["seed"])
        for s in lst:
            ho, hf = holdout_acc(f"{args.tag}{arm}_s{s['seed']}")
            s.update(holdout_acc=ho, holdout_mf1=hf)
            rows.append(f"| {arm} | {s['seed']} | {s.get('best_val')} | {ho if ho is None else round(ho, 3)} | "
                        f"{s['secs']:.0f} | {s.get('total_metric_calls')} | {s['reflect_calls']} | {s['gate_passed']} | "
                        f"{s['gate_ties']} | {s.get('candidates')} | {calls_to_reach(s, s['val_scores'][0] + 0.02) if s.get('val_scores') else None} | "
                        f"${s['spent_usd']:.3f} |")
        ok = [s for s in lst if s.get("best_val") is not None]
        if ok:
            table[arm] = {k: (round(mean(v), 4), round(pstdev(v), 4)) for k, v in {
                "best_val": [s["best_val"] for s in ok], "secs": [s["secs"] for s in ok],
                "holdout_acc": [s["holdout_acc"] for s in ok if s["holdout_acc"] is not None] or [float("nan")],
                "rewrites": [s["reflect_calls"] for s in ok], "gate_passed": [s["gate_passed"] for s in ok],
                "spent_usd": [s["spent_usd"] for s in ok]}.items()}
    print("| arm | seed | best val | holdout acc | secs | GEPA calls | rewrites | gate passed | gate ties | candidates |"
          " calls to seed+.02 | $ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print(json.dumps(table, indent=1))
    print("reference holdout:", {n: holdout_acc(n) for n in REFERENCE}, "B2 val (pilot)", round(b2_val, 4))
    (RUNS / f"{args.tag}analysis.json").write_text(json.dumps({"runs": rs, "table": table}, indent=1, default=str))


if __name__ == "__main__":
    main()
