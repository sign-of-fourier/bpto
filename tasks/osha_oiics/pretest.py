"""Offline check of the two-step program under official GEPA ($0): real rows and codebook, mock models.

    python -m tasks.osha_oiics.pretest

Mock task model: step 1 names the gold group only if the route prompt carries "[rule <group>]", else group 42;
step 2 names the gold code only if the code prompt carries "[rule <code>]", else the first listed code. Mock
reflector: appends a rule for the most frequent gold in GEPA's feedback to the component it was shown. So the search
can climb only by rewriting both components. Checks: climbs; both components rewritten; GEPA rows == adapter rows;
billed calls == client calls; <= 2 calls per row; q=4 (independent, q-EI with a hash embedder, q-EI on outcome vectors) runs.
Decoupled mode (step=route / step=code, q=1): climbs on its own labels; only that component exists and changes;
<= 1 call per row; the code step never sees a route prompt.
"""
from __future__ import annotations

import re
from collections import Counter

from bpto import Budget, MockClient, ModelConfig
from bpto.bo import HashEmbedder

from .data import codebook, read
from .program import Row, run_official
from .run import load as _unused  # noqa: F401  (import check)


def main():
    book = codebook()
    train = [Row(r["id"], r["narrative"], r["group"], r["code"]) for r in read("train")][:60]
    val = [Row(r["id"], r["narrative"], r["group"], r["code"]) for r in read("val")][:40]
    by_narr = {r.narrative: r for r in train + val}
    seed = {"route": "Pick the major group.\n\nNarrative: {narrative}",
            "code": "Group {group}. Codes:\n{codes}\nAnswer with the code.\n\nNarrative: {narrative}"}

    def task(prompt, cfg, schema):
        row = next(r for n, r in by_narr.items() if n in prompt)
        if "Codes:" in prompt or "{codes}" not in prompt and re.search(r"^\d{4}: ", prompt, re.M):
            first = re.search(r"^(\d{4}): ", prompt, re.M).group(1)
            return row.code if f"[rule {row.code}]" in prompt else first
        return f"Answer: {row.group}" if f"[rule {row.group}]" in prompt else "42 Falls on same level"

    def reflect(prompt, cfg, schema):
        current = re.search(r"```\n(.*?)\n```", prompt, re.S).group(1)
        golds = [g for g in re.findall(r"gold (?:was|code is) (\d{2,4})", prompt) if f"[rule {g}]" not in current]
        new = current + (f"\n[rule {Counter(golds).most_common(1)[0][0]}]" if golds else "\n(no change)")
        return f"```\n{new}\n```"

    results = {}
    for arm in ("q1", "independent4", "qei4", "qei4o"):
        sampling = None
        if arm == "independent4":
            from gepa.strategies.proposal_sampling import IndependentSampling
            sampling = IndependentSampling(4)
        elif arm == "qei4":
            from tasks.osha_sir.qei_sampling import QEISampling
            sampling = QEISampling(4, HashEmbedder(), seed=0)
        elif arm == "qei4o":
            from tasks.osha_sir.qei_sampling import QEISampling
            sampling = QEISampling(4, seed=0, features="outcomes")
        tc = MockClient(task, budget=Budget(max_calls=4000), default_config=ModelConfig(model="mock-task"))
        rc = MockClient(reflect, default_config=ModelConfig(model="mock-reflect"))
        res, ad, lm = run_official(seed, train, val, book, task_client=tc, reflect_client=rc, max_metric_calls=1500,
                                   minibatch=15, seed_value=0, sampling_strategy=sampling)
        changed = {k: any(c[k] != seed[k] for c in res.candidates) for k in seed}
        s = ad.stats
        out = {"seed_val": round(res.val_aggregate_scores[0], 3), "best_val": round(max(res.val_aggregate_scores), 3),
               "candidates": len(res.candidates), "components_rewritten": changed,
               "gepa_rows": res.total_metric_calls, "adapter_rows": s.requested, "billed": s.billed,
               "client_calls": tc.usage.calls, "cache_hits": tc.usage.cache_hits, "errors": s.errors,
               "reflections": rc.usage.calls}
        ok = (out["best_val"] > out["seed_val"] and all(changed.values()) and out["gepa_rows"] == out["adapter_rows"]
              and out["billed"] == out["client_calls"] and out["billed"] + out["cache_hits"] <= 2 * out["adapter_rows"])
        print(f"[{'PASS' if ok else 'FAIL'}] {arm}: {out}")
        results[arm] = ok
    for step in ("route", "code"):
        prompts = []

        def spy(prompt, cfg, schema, _p=prompts):
            _p.append(prompt)
            return task(prompt, cfg, schema)
        tc = MockClient(spy, budget=Budget(max_calls=4000), default_config=ModelConfig(model="mock-task"))
        rc = MockClient(reflect, default_config=ModelConfig(model="mock-reflect"))
        res, ad, lm = run_official(seed, train, val, book, task_client=tc, reflect_client=rc, max_metric_calls=1500,
                                   minibatch=15, seed_value=0, step=step)
        s = ad.stats
        route_calls = sum("Pick the major group" in p for p in prompts)
        out = {"seed_val": round(res.val_aggregate_scores[0], 3), "best_val": round(max(res.val_aggregate_scores), 3),
               "components": sorted({k for c in res.candidates for k in c}),
               "rewritten": any(c[step] != seed[step] for c in res.candidates), "gepa_rows": res.total_metric_calls,
               "adapter_rows": s.requested, "billed": s.billed, "cache_hits": tc.usage.cache_hits,
               "route_calls": route_calls, "calls": len(prompts)}
        ok = (out["best_val"] > out["seed_val"] and out["components"] == [step] and out["rewritten"]
              and out["gepa_rows"] == out["adapter_rows"] and out["billed"] + out["cache_hits"] <= out["adapter_rows"]
              and (route_calls == len(prompts) if step == "route" else route_calls == 0))
        print(f"[{'PASS' if ok else 'FAIL'}] decoupled {step}: {out}")
        results[f"decoupled_{step}"] = ok
    assert all(results.values()), results


if __name__ == "__main__":
    main()
