"""IFBench data: 300 prompts, each with 1-2 verifiable constraints (58 types), no train split released.

Record format (JSONL): {"id": key, "inputs": {"instruction": prompt},
                        "meta": {"instruction_id_list": [...], "kwargs": [{...}, ...]}}
`answer` is None - correctness is decided by the constraint checkers, not a reference output.
"""
from __future__ import annotations

import json
from pathlib import Path

from bpto import Dataset

HF_NAME = "allenai/IFBench_test"


def from_hf(name: str = HF_NAME) -> Dataset:
    """Download (cached by `datasets`) and convert. 300 rows."""
    from datasets import load_dataset
    rows = load_dataset(name)["train"]
    return Dataset.from_records([
        {"id": str(r["key"]), "inputs": {"instruction": r["prompt"]}, "answer": None,
         "meta": {"instruction_id_list": list(r["instruction_id_list"]),
                  "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in r["kwargs"]]}}
        for r in rows
    ])


def save_jsonl(dataset: Dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in dataset:
            f.write(json.dumps({"id": ex.id, "inputs": ex.inputs, "answer": ex.answer, "meta": ex.meta}) + "\n")
    return path


def load(path: str | Path) -> Dataset:
    return Dataset.from_jsonl(path)


def load_or_fetch(path: str | Path = "tasks/ifbench/data/ifbench_test.jsonl") -> Dataset:
    """Use the local JSONL copy if present, otherwise fetch from Hugging Face and save it."""
    path = Path(path)
    if path.exists():
        return load(path)
    ds = from_hf()
    save_jsonl(ds, path)
    return ds


def constraint_types(dataset: Dataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in dataset:
        for cid in ex.meta["instruction_id_list"]:
            counts[cid] = counts.get(cid, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
