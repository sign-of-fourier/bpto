"""Client for TypeSafe's Jev (`/v1/systemone`): typed answers instead of text. Uses httpx directly, no SDK.

Jev takes a `state` (text or JSON) and named questions, and returns one typed answer per question:
- noul   -> {"noul": P(yes)}
- choice -> {"choice": label, "confidence", "probabilities": {label: p}}
- score  -> {"score": probability-weighted level (may fall between levels), "confidence", "probabilities"}

Served by TypeSafe (`https://api.typesafe.ai`) and by OpenRouter (`https://openrouter.ai/api`, OpenRouter key) with the
same protocol. The request goes through `ModelClient.complete` so cache, budget and concurrency are shared: the
(state, questions) pair, in the caller's key order, is the cached "prompt". Pin a model version (e.g. `jev-1.13`)
rather than `jev-latest` when a cache file outlives a run.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from pydantic import BaseModel

from .base import Completion, ModelClient, ModelConfig


def noul(instructions: Any, yes: Any = None, no: Any = None) -> dict:
    """A yes/no question; the answer is P(yes). `yes` / `no` optionally describe the two outcomes."""
    q: dict = {"type": "noul", "instructions": instructions}
    if yes is not None or no is not None:
        q["criteria"] = {"true": yes, "false": no}
    return q


def choice(instructions: Any, criteria: dict[str, Any]) -> dict:
    """Pick one label (up to 255); `criteria` maps each label to its description (or None)."""
    if not 1 <= len(criteria) <= 255:
        raise ValueError(f"choice needs 1-255 labels, got {len(criteria)}")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: Any, levels: list[Any]) -> dict:
    """Place the state on an ordered rubric; `levels[i]` describes score i (from 0)."""
    if not levels:
        raise ValueError("score needs at least one level")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


class JevClient(ModelClient):
    def __init__(self, model: str = "jev-latest", base_url: str = "https://api.typesafe.ai", api_key: str | None = None,
                 *, timeout: float = 30.0, max_retries: int = 4, transport: httpx.AsyncBaseTransport | None = None,
                 **kw):
        cfg = kw.pop("default_config", None) or ModelConfig(model=model)
        super().__init__(default_config=cfg, **kw)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.max_retries = max_retries
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout, transport=transport)

    async def ask(self, state: Any, questions: dict[str, dict], *, config: ModelConfig | None = None) -> dict[str, dict]:
        """One request; returns {question name: answer dict}. Cached on (state, questions, config)."""
        for name, q in questions.items():
            if q.get("type") not in ("noul", "choice", "score"):
                raise ValueError(f"question {name!r}: type must be noul, choice or score")
        # Not sort_keys: Jev reads the state in order and key order changes its answers, so the order is sent as
        # given and is part of the cache key.
        prompt = json.dumps({"state": state, "questions": questions}, default=str)
        comp = await self.complete(prompt, config=config)
        return comp.parsed

    async def _complete(self, prompt: str, cfg: ModelConfig, schema: type[BaseModel] | None) -> Completion:
        if schema is not None:
            raise ValueError("Jev returns typed answers, not schema-shaped text; use JevClient.ask")
        try:
            req = json.loads(prompt)
            body = {"state": req["state"], "questions": req["questions"], "model": cfg.model}
        except (json.JSONDecodeError, KeyError, TypeError):
            raise ValueError("JevClient takes (state, questions), not free text; use JevClient.ask") from None
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._http.post("/v1/systemone", json=body)
                if r.status_code in (408, 409, 429) or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                retryable = not isinstance(e, httpx.HTTPStatusError) or e.response.status_code in (408, 409, 429) \
                    or e.response.status_code >= 500
                if attempt == self.max_retries or not retryable:
                    raise
                await asyncio.sleep(min(2 ** attempt, 30))
        data = r.json()
        answers = data.get("answers") or {}
        missing = set(body["questions"]) - set(answers)
        if missing:
            raise ValueError(f"Jev returned no answer for {sorted(missing)}")
        usage = data.get("usage") or {}
        return Completion(text=r.text, parsed=answers, input_tokens=usage.get("input_tokens") or 0,
                          output_tokens=usage.get("output_tokens") or 0, model=data.get("model", cfg.model))

    async def aclose(self) -> None:
        await self._http.aclose()
