"""Completion cache: in-memory, optionally persisted to a JSONL file (write-through)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .base import Completion


class CompletionCache:
    def __init__(self, path: str | Path | None = None):
        self._mem: dict[str, "Completion"] = {}
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            from .base import Completion

            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._mem[rec["key"]] = Completion.model_validate(rec["completion"])

    def get(self, key: str):
        return self._mem.get(key)

    def put(self, key: str, comp: "Completion") -> None:
        self._mem[key] = comp
        if self.path:
            dump = comp.model_dump()
            if isinstance(comp.parsed, BaseModel):
                dump["parsed"] = comp.parsed.model_dump()
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "completion": dump}, default=str) + "\n")

    def __len__(self) -> int:
        return len(self._mem)
