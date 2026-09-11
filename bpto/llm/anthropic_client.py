"""Anthropic Messages API client."""
from __future__ import annotations

import anthropic
from pydantic import BaseModel

from .base import Completion, ModelClient, ModelConfig


class AnthropicClient(ModelClient):
    def __init__(self, model: str = "claude-opus-5", *, api_key: str | None = None, max_retries: int = 4, **kw):
        cfg = kw.pop("default_config", None) or ModelConfig(model=model)
        super().__init__(default_config=cfg, **kw)
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=max_retries)

    def _params(self, prompt: str, cfg: ModelConfig) -> dict:
        p: dict = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if cfg.system:
            p["system"] = cfg.system
        if cfg.temperature is not None:
            p["temperature"] = cfg.temperature
        if cfg.effort:
            p["output_config"] = {"effort": cfg.effort}
        return p

    async def count_tokens(self, text: str, config: ModelConfig | None = None) -> int:
        cfg = self.default_config.merged(config)
        params = {"model": cfg.model, "messages": [{"role": "user", "content": text}]}
        if cfg.system:
            params["system"] = cfg.system
        r = await self._client.messages.count_tokens(**params)
        return r.input_tokens

    async def _complete(self, prompt: str, cfg: ModelConfig, schema: type[BaseModel] | None) -> Completion:
        params = self._params(prompt, cfg)
        if schema is not None:
            resp = await self._client.messages.parse(output_format=schema, **params)
            parsed = resp.parsed_output
        else:
            resp = await self._client.messages.create(**params)
            parsed = None
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(
            text=text,
            parsed=parsed,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
            stop_reason=resp.stop_reason,
        )
