"""Deterministic client for tests. `handler(prompt, config, schema)` returns str or a schema instance."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from pydantic import BaseModel

from .base import Completion, ModelClient, ModelConfig


class MockClient(ModelClient):
    def __init__(self, handler: Callable[..., Any] | None = None, delay: float = 0.0, **kw):
        super().__init__(**kw)
        self.handler = handler or (lambda prompt, cfg, schema: prompt)
        self.delay = delay
        self.calls: list[str] = []

    async def count_tokens(self, text: str, config: ModelConfig | None = None) -> int:
        return len(text.split())

    async def _complete(self, prompt: str, cfg: ModelConfig, schema: type[BaseModel] | None) -> Completion:
        self.calls.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        out = self.handler(prompt, cfg, schema)
        if inspect.isawaitable(out):
            out = await out
        if isinstance(out, BaseModel):
            return Completion(text=out.model_dump_json(), parsed=out,
                              input_tokens=len(prompt.split()), output_tokens=len(out.model_dump_json().split()))
        text = str(out)
        parsed = schema.model_validate_json(text) if schema is not None else None
        return Completion(text=text, parsed=parsed,
                          input_tokens=len(prompt.split()), output_tokens=len(text.split()))
