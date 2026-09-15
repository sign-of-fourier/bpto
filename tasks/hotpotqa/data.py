"""HotpotQA distractor setting (Yang et al. 2018): a question plus 10 Wikipedia paragraphs (2 gold, 8 distractors),
answer = short span or yes/no. Single-prompt reading-comprehension version of the GEPA-paper task (their
HotpotQA is a 3-module retrieval program; this is the closest thing that fits one `Task`).

Record format (JSONL): {"id", "inputs": {"question", "context"}, "answer": str,
                        "meta": {"type": bridge|comparison, "supporting_titles": [...], "gold": [paragraph, ...]}}
`context` is the 10 paragraphs rendered as "Title: sentences" blocks, in the dataset's order.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from bpto import Dataset

HF_NAME = "hotpotqa/hotpot_qa"
DEFAULT_PATH = Path(__file__).parent / "data" / "hotpotqa_distractor_dev600.jsonl"


def render_context(titles: list[str], sentences: list[list[str]]) -> str:
    return "\n\n".join(f"{t}: {''.join(s)}" for t, s in zip(titles, sentences))


def from_hf(n: int | None = None, seed: int = 0, split: str = "validation") -> Dataset:
    """Download (cached by `datasets`) the distractor dev split and convert; `n` = seeded random subset."""
    from datasets import load_dataset
    rows = load_dataset(HF_NAME, "distractor", split=split)
    idx = list(range(len(rows)))
    if n is not None:
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n])
    out = []
    for i in idx:
        r = rows[i]
        titles, sents = r["context"]["title"], r["context"]["sentences"]
        gold = set(r["supporting_facts"]["title"])
        out.append({"id": r["id"], "inputs": {"question": r["question"], "context": render_context(titles, sents)},
                    "answer": r["answer"],
                    "meta": {"type": r["type"], "supporting_titles": sorted(gold),
                             "gold": [f"{t}: {''.join(s)}" for t, s in zip(titles, sents) if t in gold]}})
    return Dataset.from_records(out)


def save_jsonl(dataset: Dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in dataset:
            f.write(json.dumps({"id": ex.id, "inputs": ex.inputs, "answer": ex.answer, "meta": ex.meta}) + "\n")
    return path


def load(path: str | Path = DEFAULT_PATH) -> Dataset:
    return Dataset.from_jsonl(path)


def load_or_fetch(path: str | Path = DEFAULT_PATH, n: int = 600, seed: int = 0) -> Dataset:
    path = Path(path)
    if path.exists():
        return load(path)
    ds = from_hf(n, seed)
    save_jsonl(ds, path)
    return ds
