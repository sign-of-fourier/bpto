"""Expand-seat diagnostics from GEPA's own run_log.json ($0; run from the repo root, needs runs/osha_oiics/).

Per arm and seed: how many distinct parents a q=4 iteration draws, how often the parent is the current best-on-val
program, gate outcome of every proposal (strict improvement on its 15 rows / tie / worse), the full-val gain of
accepted children over their parent, and the time split between reflection and task calls (latencies from the caches).
"""
import json
from collections import Counter
from pathlib import Path
from statistics import mean

G, R = Path("runs/osha_oiics/gepa"), Path("runs/osha_oiics")
ARMS = [a for a in ["q1", "independent4", "qei4", "qei4o"] if (G / f"main_{a}_s4/summary.json").exists()]


def one(arm, seed):
    d = G / f"main_{arm}_s{seed}"
    log = json.loads((d / "gepa_state/run_log.json").read_text())
    summ = json.loads((d / "summary.json").read_text())
    val = summ["val_scores"]
    distinct, on_best, outcomes, gains, n_pool, parent_rank = [], 0, Counter(), [], [], []
    n_prop = 0
    for it in log:
        tasks = it.get("tasks") or [{"parent_idx": it["selected_program_candidate"],
                                     "subsample_scores": it.get("subsample_scores"),
                                     "new_subsample_scores": it.get("new_subsample_scores")}]
        # candidates that exist at this point = those created before this iteration
        pool = [j for j in range(len(val)) if j == 0 or j <= max([0] + [p for x in log[:log.index(it)]
                                                                        for p in x.get("new_program_indices", [])])]
        best = max(pool, key=lambda j: val[j])
        ranked = sorted(pool, key=lambda j: -val[j])
        n_pool.append(len(pool))
        distinct.append(len({t["parent_idx"] for t in tasks}))
        for t in tasks:
            n_prop += 1
            on_best += t["parent_idx"] == best
            parent_rank.append(ranked.index(t["parent_idx"]) / max(len(pool) - 1, 1))
            old, new = t.get("subsample_scores"), t.get("new_subsample_scores")
            if old is None or new is None:
                outcomes["no_eval"] += 1
            else:
                outcomes["better" if sum(new) > sum(old) else "tie" if sum(new) == sum(old) else "worse"] += 1
        for j in it.get("new_program_indices", []):
            p = summ["parents"][j][0]
            gains.append(val[j] - val[p])
    # time split: summed call latency (reflection calls are one per proposal; task calls overlap 16-wide)
    rl = [json.loads(l)["completion"]["latency_s"] for l in open(R / f"reflect_log/main_{arm}_s{seed}.jsonl")]
    q = 1 if arm == "q1" else 4
    return {"iters": len(log), "proposals": n_prop, "pool_mean": round(mean(n_pool), 1),
            "distinct_parents_per_iter": round(mean(distinct), 2), "parent_is_best_val": round(on_best / n_prop, 2),
            "parent_rank_pct": round(mean(parent_rank), 2),
            "gate": {k: round(v / n_prop, 2) for k, v in outcomes.items()},
            "accepted_gain_vs_parent": round(mean(gains), 4) if gains else None,
            "accepted_beat_parent": f"{sum(g > 0 for g in gains)}/{len(gains)}",
            "reflect_s_sum": round(sum(rl)), "reflect_s_mean": round(mean(rl), 1),
            "reflect_wall_est": round(sum(rl) / q), "secs": summ["secs"]}


out = {a: [one(a, s) for s in range(5)] for a in ARMS}
for a, rs in out.items():
    agg = {k: round(mean(r[k] for r in rs), 3) for k in rs[0] if isinstance(rs[0][k], (int, float)) and rs[0][k] is not None}
    gate = {k: round(mean(r["gate"].get(k, 0) for r in rs), 3) for k in ("better", "tie", "worse", "no_eval")}
    gains = [r["accepted_gain_vs_parent"] for r in rs if r["accepted_gain_vs_parent"] is not None]
    print(a, json.dumps({**agg, "gate": gate, "accepted_gain_vs_parent": round(mean(gains), 4),
                         "beat_parent": [r["accepted_beat_parent"] for r in rs]}))
(Path(__file__).parent / "sampler_diagnostics.json").write_text(json.dumps(out, indent=1))
