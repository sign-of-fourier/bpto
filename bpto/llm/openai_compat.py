"""Client for any OpenAI-compatible `/v1/chat/completions` endpoint (OpenAI, vLLM, Ollama,
OpenRouter, TGI, ...). Uses httpx directly so there is no provider SDK dependency.

Structured output is requested with `response_format: json_schema`; if the server ignores it,
the text is still validated against the schema client-side.
"""
from __future__ import annotations

import json

import httpx
from pydantic import BaseModel

from .base import Completion, ModelClient, ModelConfig


class OpenAICompatibleClient(ModelClient):
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 *, timeout: float = 120.0, max_retries: int = 4, extra_headers: dict | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, **kw):
        cfg = kw.pop("default_config", None) or ModelConfig(model=model)
        super().__init__(default_config=cfg, **kw)
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.max_retries = max_retries
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout,
                                       transport=transport)

    def _body(self, prompt: str, cfg: ModelConfig, schema: type[BaseModel] | None) -> dict:
        messages = []
        if cfg.system:
            messages.append({"role": "system", "content": cfg.system})
        messages.append({"role": "user", "content": prompt})
        body: dict = {"model": cfg.model, "messages": messages, "max_tokens": cfg.max_tokens}
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.effort:
            body["reasoning_effort"] = cfg.effort
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema()}}
        return body

    async def _complete(self, prompt: str, cfg: ModelConfig, schema: type[BaseModel] | None) -> Completion:
        body = self._body(prompt, cfg, schema)
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._http.post("/chat/completions", json=body)
                if r.status_code in (408, 409, 429) or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                if attempt == self.max_retries:
                    raise
                import asyncio
                await asyncio.sleep(min(2 ** attempt, 30))
        data = r.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        parsed = schema.model_validate(json.loads(text)) if schema is not None else None
        usage = data.get("usage") or {}
        return Completion(text=text, parsed=parsed,
                          input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0),
                          model=data.get("model", cfg.model), stop_reason=choice.get("finish_reason"))

    async def aclose(self) -> None:
        await self._http.aclose()
