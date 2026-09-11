"""Training examples: (inputs, answer) pairs."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field


class Example(BaseModel):
    id: str
    inputs: dict[str, Any]
    answer: Any = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Dataset:
    def __init__(self, examples: list[Example]):
        self.examples = list(examples)

    @classmethod
    def from_records(cls, records: list[dict]) -> "Dataset":
        out = []
        for i, r in enumerate(records):
            r = dict(r)
            r.setdefault("id", str(i))
            out.append(Example(**r))
        return cls(out)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        with open(path) as f:
            return cls.from_records([json.loads(l) for l in f if l.strip()])

    def sample(self, k: int, seed: int | None = None) -> "Dataset":
        """First k of a seeded shuffle, so samples with the same seed are nested (k=8 ⊂ k=32)."""
        if k >= len(self):
            return Dataset(self.examples)
        ex = list(self.examples)
        random.Random(seed).shuffle(ex)
        return Dataset(ex[:k])

    def split(self, frac: float, seed: int | None = None) -> tuple["Dataset", "Dataset"]:
        ex = list(self.examples)
        random.Random(seed).shuffle(ex)
        n = int(len(ex) * frac)
        return Dataset(ex[:n]), Dataset(ex[n:])

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
