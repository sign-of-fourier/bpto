"""Build the OSHA SIR splits from OSHA's full Severe Injury Report CSV.

    python -m tasks.osha_sir.data --csv path/to/January2015toNovember2025.csv

Source: https://www.osha.gov/severe-injury-reports ("Download the full current SIR data set"; the 2025-11 file is
https://www.osha.gov/sites/default/files/January2015toNovember2025.zip). OSHA's CDN refuses some cloud IPs; the direct
file link has worked where the dashboard page did not.

Rules (fixed before any run):
  - EventDate years 2015-2023; `Final Narrative` non-empty; whitespace collapsed.
  - Target: `Event` (OIICS 2.01 event or exposure code, 2-4 digits, stored without leading zeros) cut to its two-digit
    major group = the code's first two characters. Rows coded only at division level (two-digit codes such as 60
    "Contact with objects and equipment, unspecified") and 9999 "Nonclassifiable" have no major group: dropped.
  - Classes: the 12 most frequent major groups by their official OIICS titles, everything else "Other" (13).
  - Duplicates: narratives equal after lower-casing, dropping digits and punctuation are one incident; the first
    (by ID) is kept, so no incident can land in two splits.
  - Splits, disjoint, seeded: train 200 (reflection minibatches), val 200 (GEPA's full / Pareto set), holdout 200
    (scored once per finished prompt, never seen by an optimizer). Each: at least FLOOR rows per class, the rest in
    proportion to the class's share of the filtered data. The holdout's sha256 goes in manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
FLOOR = 8
SIZES = {"train": 200, "val": 200, "holdout": 200}
SEED = 20260926

# OIICS 2.01 manual, section 2.4.2 (Event or Exposure), major-group headings, verbatim
OIICS_TITLES = {
    "11": "Intentional injury by person", "12": "Injury by person—unintentional or intent unknown",
    "13": "Animal and insect related incidents", "24": "Pedestrian vehicular incident",
    "26": "Roadway incidents involving motorized land vehicle",
    "27": "Nonroadway incidents involving motorized land vehicles", "31": "Fires", "32": "Explosions",
    "42": "Falls on same level", "43": "Falls to lower level", "51": "Exposure to electricity",
    "53": "Exposure to temperature extremes", "55": "Exposure to other harmful substances",
    "62": "Struck by object or equipment", "63": "Struck against object or equipment",
    "64": "Caught in or compressed by equipment or objects", "71": "Overexertion involving outside sources",
    "73": "Other exertions or bodily reactions",
}
OTHER = "Other"


def _key(narrative: str) -> str:
    return re.sub(r"[^a-z]+", " ", narrative.lower()).strip()


def load(csv: str | Path):
    import pandas as pd

    df = pd.read_csv(csv, encoding="latin-1", low_memory=False)
    year = pd.to_datetime(df["EventDate"], errors="coerce").dt.year
    df = df[(year >= 2015) & (year <= 2023)].copy()
    df["narrative"] = df["Final Narrative"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["narrative"] != ""]
    df["code"] = df["Event"].astype(int).astype(str)
    stats = {"rows_2015_2023_nonempty": len(df)}
    df = df[(df["code"].str.len() >= 3) & (df["code"] != "9999")]
    stats["rows_with_major_group"] = len(df)
    df = df.sort_values("ID")
    df["dup_key"] = df["narrative"].map(_key)
    df = df.drop_duplicates("dup_key")
    stats["rows_after_dedup"] = len(df)
    df["group"] = df["code"].str[:2]
    return df, stats


def build(csv: str | Path, out: Path = DATA) -> dict:
    df, stats = load(csv)
    counts = df["group"].value_counts()
    top = list(counts.index[:12])
    missing = [g for g in top if g not in OIICS_TITLES]
    assert not missing, f"no OIICS title recorded for major group(s) {missing}"
    labels = [OIICS_TITLES[g] for g in top] + [OTHER]
    df["label"] = df["group"].map(lambda g: OIICS_TITLES[g] if g in top else OTHER)
    share = df["label"].value_counts(normalize=True).to_dict()

    rng = random.Random(SEED)
    pools = {l: sorted(df.index[df["label"] == l].tolist()) for l in labels}
    for l in labels:
        rng.shuffle(pools[l])
    splits = {}
    for name, n in SIZES.items():
        quota = {l: FLOOR for l in labels}
        rest = n - FLOOR * len(labels)
        raw = {l: rest * share[l] for l in labels}
        for l in labels:
            quota[l] += int(raw[l])
        for l in sorted(labels, key=lambda l: raw[l] - int(raw[l]), reverse=True)[: n - sum(quota.values())]:
            quota[l] += 1
        rows = []
        for l in labels:
            take, pools[l] = pools[l][: quota[l]], pools[l][quota[l]:]
            rows += [{"id": str(df.at[i, "ID"]), "narrative": df.at[i, "narrative"], "label": l,
                      "event_code": df.at[i, "code"], "event_title": str(df.at[i, "EventTitle"]).strip()} for i in take]
        rng.shuffle(rows)
        splits[name] = rows

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": Path(csv).name, "seed": SEED, "floor_per_class": FLOOR, "labels": labels,
                "top_groups": {g: OIICS_TITLES[g] for g in top}, "filter": stats,
                "class_share_filtered": {l: round(share[l], 4) for l in labels}, "splits": {}}
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path.write_text(text)
        manifest["splits"][name] = {"rows": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                    "per_class": dict(Counter(r["label"] for r in rows))}
    ids = [r["id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids)), "splits overlap"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def read(split: str, data: Path = DATA) -> list[dict]:
    return [json.loads(l) for l in (data / f"{split}.jsonl").read_text().splitlines()]


def labels(data: Path = DATA) -> list[str]:
    return json.loads((data / "manifest.json").read_text())["labels"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    m = build(ap.parse_args().csv)
    print(json.dumps({k: m[k] for k in ("filter", "top_groups")}, indent=2, ensure_ascii=False))
    for name, s in m["splits"].items():
        print(name, s["rows"], s["sha256"][:12], s["per_class"])
