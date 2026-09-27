"""Collect the main runs: per arm and seed, the chosen program's val and holdout scores, wall-clock, calls, and which
steps the search changed.

    python -m tasks.osha_oiics.analyze [--tag main_] [--score-holdout] [--export experiments/<dir>]

--score-holdout scores every run's chosen program plus the reference programs (C0, A) on the holdout (2 repeats,
fresh calls per repeat) through `run.py score`; without it the table reads existing holdout scores. --export writes evaluations.jsonl (one run per line),
reference_scores.json and table.md into the experiment dir.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev

from .run import PROMPTS, RUNS

ARMS = ["q1", "independent4", "qei4", "qei4o"]
REFERENCE = ["C0", "A"]


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
        s["proposals"] = {k: len(re.findall(rf"Proposed new text for {k}:", log)) for k in ("route", "code")}
        if (d / "candidates.json").exists():
            cands = json.loads((d / "candidates.json").read_text())
            best = cands[s["best_idx"]]
            s["best_changed"] = [k for k in ("route", "code") if best[k] != cands[0][k]]
            s["accepted_by_step"] = {k: sum(c[k] != cands[p[0]][k] for c, p in zip(cands[1:], s["parents"][1:]) if p)
                                     for k in ("route", "code")}
        out.setdefault(m.group(1), []).append(s)
    return out


def export_best(tag: str, rs: dict) -> list[str]:
    names = []
    for arm, lst in rs.items():
        for s in lst:
            src = Path(s["dir"]) / "best_prompt.json"
            if src.exists():
                name = f"{tag}{arm}_s{s['seed']}"
                (PROMPTS / f"{name}.json").write_text(src.read_text())
                names.append(name)
    return names


def holdout(name: str) -> dict | None:
    p = RUNS / "scores" / "holdout" / "summary.json"
    s = json.loads(p.read_text()).get(f"{name}_both") if p.exists() else None
    if not s:
        return None
    return {k: round(mean(r[k] for r in s["reps"]), 4) for k in ("code_acc", "group_acc", "code_acc_given_group")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main_")
    ap.add_argument("--score-holdout", action="store_true")
    ap.add_argument("--export", type=Path)
    args = ap.parse_args(argv)
    rs = runs(args.tag)
    names = export_best(args.tag, rs)
    if args.score_holdout:
        from .run import main as run_main
        run_main(["score", *REFERENCE, *names, "--split", "holdout", "--repeats", "2", "--max-usd", "2.0"])

    rows, table = [], {}
    for arm in ARMS:
        lst = sorted(rs.get(arm, []), key=lambda s: s["seed"])
        for s in lst:
            s["holdout"] = h = holdout(f"{args.tag}{arm}_s{s['seed']}")
            rows.append(f"| {arm} | {s['seed']} | {s.get('seed_val')} | {s.get('best_val')} | "
                        f"{h and h['code_acc']} | {h and h['group_acc']} | {s['secs']:.0f} | {s.get('total_metric_calls')} | "
                        f"{s['reflect_calls']} | {s['gate_passed']} | {s['gate_ties']} | {'+'.join(s.get('best_changed') or []) or 'seed'} | "
                        f"${s['spent_usd']:.3f} |")
        ok = [s for s in lst if s.get("best_val") is not None]
        if ok:
            col = lambda f: [f(s) for s in ok if f(s) is not None]
            table[arm] = {k: (round(mean(v), 4), round(pstdev(v), 4)) if v else None for k, v in {
                "seed_val": col(lambda s: s["seed_val"]), "best_val": col(lambda s: s["best_val"]),
                "val_gain": col(lambda s: s["best_val"] - s["seed_val"]), "secs": col(lambda s: s["secs"]),
                "holdout_code": col(lambda s: s["holdout"] and s["holdout"]["code_acc"]),
                "holdout_group": col(lambda s: s["holdout"] and s["holdout"]["group_acc"]),
                "rewrites": col(lambda s: s["reflect_calls"]), "gate_passed": col(lambda s: s["gate_passed"]),
                "spent_usd": col(lambda s: s["spent_usd"])}.items()}
    header = ("| arm | seed | seed val | best val | holdout code | holdout group | secs | GEPA rows | rewrites | gate passed |"
              " gate ties | best changed | $ |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("| arm | seed | seed val | best val | holdout code | holdout group | secs | GEPA rows | rewrites | gate passed |"
          " gate ties | best changed | $ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print(json.dumps(table, indent=1))
    print("reference holdout:", {n: holdout(n) for n in REFERENCE})
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        keep = lambda s: {k: v for k, v in s.items() if k != "dir"}
        (args.export / "evaluations.jsonl").write_text(
            "".join(json.dumps(keep(s)) + "\n" for arm in ARMS for s in sorted(rs.get(arm, []), key=lambda s: s["seed"])))
        (args.export / "reference_scores.json").write_text(json.dumps({n: holdout(n) for n in REFERENCE}, indent=1))
        (args.export / "table.md").write_text(header + "\n" + "\n".join(rows) + "\n")
    (RUNS / f"{args.tag}analysis.json").write_text(json.dumps({"runs": rs, "table": table}, indent=1, default=str))


if __name__ == "__main__":
    main()
