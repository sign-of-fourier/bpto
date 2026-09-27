"""Build the detailed-OIICS splits from OSHA's full Severe Injury Report CSV (the same file as tasks/osha_sir).

    python -m tasks.osha_oiics.data --csv path/to/January2015toNovember2025.csv

Rows are kept only when they fit the two-step form of the problem (fixed before any run):
  - osha_sir's filters: EventDate 2015-2023, non-empty narrative, 9999 dropped, near-duplicate narratives deduplicated.
  - A full four-digit OIICS 2.01 event code. Rows coded only to three digits (~26%) have no detailed answer: dropped.
  - Not "unspecified" (a four-digit code ending in 0: the coder could not tell from the report, a judgement about the
    narrative rather than a category): dropped. "n.e.c." (ending in 9) is a real category and is kept.
  - Codes with at least MIN_ROWS rows in the filtered data; rarer codes' rows dropped.
Step 1 picks the two-digit major group (first two digits of the code; its label is free), step 2 picks the detailed
code among that group's codes. Code titles are the CSV's `EventTitle` (most frequent spelling, whitespace collapsed);
group titles are the OIICS 2.01 major-group headings.

Splits, disjoint, seeded: train 200 (reflection minibatches), val 200 (GEPA's full / Pareto set), holdout 200 (scored
once per finished prompt). Each: at least FLOOR rows per major group, the rest in proportion to the group's share;
within a group, rows are drawn at random, so codes appear at their natural rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from tasks.osha_sir.data import OIICS_TITLES, load

DATA = Path(__file__).parent / "data"
MIN_ROWS = 30
FLOOR = 4
SIZES = {"train": 200, "val": 200, "holdout": 200}
SEED = 20260927

# OIICS 2.01 major-group headings missing from osha_sir's table (it only needed the top 12 + a few)
GROUP_TITLES = {**OIICS_TITLES, "23": "Animal and other non-motorized vehicle transportation incidents",
                "41": "Slip or trip without fall", "44": "Jumps to lower level"}


def _title(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("occupant s ", "occupant's ")


def build(csv: str | Path, out: Path = DATA) -> dict:
    df, stats = load(csv)
    df = df[df["code"].str.len() == 4]
    stats["rows_four_digit"] = len(df)
    df = df[~df["code"].str.endswith("0")]
    stats["rows_not_unspecified"] = len(df)
    vc = df["code"].value_counts()
    df = df[df["code"].isin(vc[vc >= MIN_ROWS].index)].copy()
    stats["rows_codes_min_rows"] = len(df)
    titles = {c: _title(s.value_counts().index[0]) for c, s in df.groupby("code")["EventTitle"]}
    groups = sorted(df["group"].unique())
    missing = [g for g in groups if g not in GROUP_TITLES]
    assert not missing, f"no OIICS title recorded for major group(s) {missing}"
    codes = {"groups": {g: GROUP_TITLES[g] for g in groups},
             "codes": {g: {c: titles[c] for c in sorted(df.loc[df["group"] == g, "code"].unique())} for g in groups}}
    share = df["group"].value_counts(normalize=True).to_dict()

    rng = random.Random(SEED)
    pools = {g: sorted(df.index[df["group"] == g].tolist()) for g in groups}
    for g in groups:
        rng.shuffle(pools[g])
    splits = {}
    for name, n in SIZES.items():
        quota = {g: FLOOR for g in groups}
        rest = n - FLOOR * len(groups)
        raw = {g: rest * share[g] for g in groups}
        for g in groups:
            quota[g] += int(raw[g])
        for g in sorted(groups, key=lambda g: raw[g] - int(raw[g]), reverse=True)[: n - sum(quota.values())]:
            quota[g] += 1
        rows = []
        for g in groups:
            take, pools[g] = pools[g][: quota[g]], pools[g][quota[g]:]
            rows += [{"id": str(df.at[i, "ID"]), "narrative": df.at[i, "narrative"], "group": g,
                      "code": df.at[i, "code"]} for i in take]
        rng.shuffle(rows)
        splits[name] = rows

    out.mkdir(parents=True, exist_ok=True)
    (out / "codes.json").write_text(json.dumps(codes, indent=1, ensure_ascii=False))
    manifest = {"source": Path(csv).name, "seed": SEED, "min_rows_per_code": MIN_ROWS, "floor_per_group": FLOOR,
                "filter": stats, "n_groups": len(groups), "n_codes": len(titles),
                "group_share_filtered": {g: round(share[g], 4) for g in groups}, "splits": {}}
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path.write_text(text)
        cc = Counter(r["code"] for r in rows)
        manifest["splits"][name] = {"rows": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                    "per_group": dict(Counter(r["group"] for r in rows)), "distinct_codes": len(cc),
                                    "majority_code_share": round(cc.most_common(1)[0][1] / len(rows), 3)}
    ids = [r["id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids)), "splits overlap"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def read(split: str, data: Path = DATA) -> list[dict]:
    return [json.loads(l) for l in (data / f"{split}.jsonl").read_text().splitlines()]


def codebook(data: Path = DATA) -> dict:
    """{"groups": {group: title}, "codes": {group: {code: title}}}"""
    return json.loads((data / "codes.json").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    m = build(ap.parse_args().csv)
    print(json.dumps({k: m[k] for k in ("filter", "n_groups", "n_codes")}, indent=1))
    for name, s in m["splits"].items():
        print(name, s["rows"], s["sha256"][:12], "codes", s["distinct_codes"], "majority", s["majority_code_share"])
