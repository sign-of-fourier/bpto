"""Model client protocol, config, completions, and the shared concurrency/cache layer."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .cache import CompletionCache


class ModelConfig(BaseModel, frozen=True):
    model: str = "claude-opus-5"
    max_tokens: int = 4096
    # Only sent when not None. Current-generation models (Opus 5, Sonnet 5, ...) reject
    # sampling params; diversity for expansion comes from asking for N variants per call.
    temperature: float | None = None
    effort: str | None = None  # low | medium | high | xhigh | max
    system: str | None = None

    def merged(self, override: "ModelConfig | None") -> "ModelConfig":
        if override is None:
            return self
        return self.model_copy(update=override.model_dump(exclude_unset=True))


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.cache_hits += other.cache_hits


class Completion(BaseModel):
    text: str
    parsed: Any = None  # validated schema instance (or dict when loaded from cache)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    cached: bool = False
    stop_reason: str | None = None

    def parsed_as(self, schema: type[BaseModel] | None):
        if schema is None or self.parsed is None or isinstance(self.parsed, schema):
            return self.parsed
        return schema.model_validate(self.parsed)


class BudgetExceeded(RuntimeError):
    pass


# $ per 1M tokens (input, output). On-demand list prices as of 2026-09; verify before relying on them.
PRICES: dict[str, tuple[float, float]] = {
    "amazon.nova-micro-v1:0": (0.035, 0.14),
    "amazon.nova-lite-v1:0": (0.06, 0.24),
    "amazon.nova-pro-v1:0": (0.80, 3.20),
    "amazon.titan-embed-text-v2:0": (0.02, 0.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def price_for(model: str, prices: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
    table = {**PRICES, **(prices or {})}
    for key in (model, model.split("/")[-1], ".".join(model.split(".")[1:])):  # "us.amazon.nova-micro-v1:0" -> table key
        if key in table:
            return table[key]
    raise KeyError(f"no price known for model {model!r}; pass prices={{...}} to Budget")


class Budget(BaseModel):
    """Hard stop on spend. Share one instance across clients/embedders for a global cap: each call is
    checked before it starts and charged when it returns, so `spent_usd` is the whole run's bill."""
    max_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_usd: float | None = None
    prices: dict[str, tuple[float, float]] = Field(default_factory=dict)
    spent: Usage = Field(default_factory=Usage)
    spent_usd: float = 0.0
    parent: "Budget | None" = None  # a local cap (e.g. max_calls per run) that also charges a shared meter

    def check(self, usage: Usage) -> None:
        if self.parent is not None:
            self.parent.check(self.parent.spent)
        if self.max_calls is not None and usage.calls >= self.max_calls:
            raise BudgetExceeded(f"calls {usage.calls} >= {self.max_calls}")
        if self.max_input_tokens is not None and usage.input_tokens >= self.max_input_tokens:
            raise BudgetExceeded(f"input tokens {usage.input_tokens} >= {self.max_input_tokens}")
        if self.max_output_tokens is not None and usage.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded(f"output tokens {usage.output_tokens} >= {self.max_output_tokens}")
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent_usd:.4f} >= ${self.max_usd:.2f}")

    def charge(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record a completed call. Raises KeyError for an unpriced model when a $ cap is set."""
        self.spent.add(Usage(input_tokens=input_tokens, output_tokens=output_tokens, calls=1))
        if self.parent is not None:
            self.parent.charge(model, input_tokens, output_tokens)
        if self.max_usd is None and not self.prices and model not in PRICES:
            return 0.0
        pin, pout = price_for(model, self.prices)
        cost = (input_tokens * pin + output_tokens * pout) / 1e6
        self.spent_usd += cost
        return cost


class ModelClient(ABC):
    """One global semaphore bounds in-flight requests for the whole tree.

    Subclasses implement `_complete`. Caching, concurrency, budget and usage
    accounting live here so every caller (evaluation, expansion, scorers) shares them.
    """

    def __init__(
        self,
        default_config: ModelConfig | None = None,
        max_concurrency: int = 8,
        cache: CompletionCache | None = None,
        budget: Budget | None = None,
    ):
        self.default_config = default_config or ModelConfig()
        self._sem = asyncio.Semaphore(max_concurrency)
        self.cache = cache if cache is not None else CompletionCache()
        self.budget = budget
        self.usage = Usage()

    async def complete(
        self,
        prompt: str,
        *,
        config: ModelConfig | None = None,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        cfg = self.default_config.merged(config)
        key = self._key(prompt, cfg, schema)
        hit = self.cache.get(key)
        if hit is not None:
            self.usage.cache_hits += 1
            return hit.model_copy(update={"cached": True})
        if self.budget is not None:
            self.budget.check(self.usage)
        self.usage.calls += 1  # reserve before awaiting, so concurrent in-flight calls count against the budget
        async with self._sem:
            t0 = time.perf_counter()
            comp = await self._complete(prompt, cfg, schema)
            comp.latency_s = time.perf_counter() - t0
        comp.model = comp.model or cfg.model
        self.usage.add(Usage(input_tokens=comp.input_tokens, output_tokens=comp.output_tokens))
        if self.budget is not None:
            self.budget.charge(cfg.model, comp.input_tokens, comp.output_tokens)
        self.cache.put(key, comp)
        return comp

    @abstractmethod
    async def _complete(self, prompt: str, config: ModelConfig, schema: type[BaseModel] | None) -> Completion: ...

    async def count_tokens(self, text: str, config: ModelConfig | None = None) -> int:
        """Tokens in `text` as a user message. Default is a chars/4 estimate; providers override."""
        return max(1, round(len(text) / 4))

    @staticmethod
    def _key(prompt: str, cfg: ModelConfig, schema: type[BaseModel] | None) -> str:
        payload = {
            "prompt": prompt,
            "config": cfg.model_dump(),
            "schema": schema.model_json_schema() if schema else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
