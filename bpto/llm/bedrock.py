"""Amazon Bedrock client via the Converse API (boto3). Works for any Bedrock model - Amazon Nova,
Llama, Mistral, ... For Claude on Bedrock you can also use AnthropicClient(bedrock_region=...).

boto3 is synchronous, so calls run in a thread; the shared semaphore still bounds concurrency.
Structured output: no native JSON-schema mode in Converse for most models, so the schema is
appended as an instruction and the reply is validated client-side (code fences tolerated).
"""
from __future__ import annotations

import asyncio
import json
import re

from pydantic import BaseModel

from .base import Completion, ModelClient, ModelConfig

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def schema_instruction(schema: type[BaseModel]) -> str:
    js = schema.model_json_schema()
    fields = ", ".join(f"{k} ({_type_name(v)})" for k, v in js.get("properties", {}).items())
    return (f"\n\nRespond with a single JSON object with exactly these fields: {fields}. "
            "Output only the object - no prose, no code fences, and do not repeat the schema.\n"
            "Schema for reference: " + json.dumps(js, separators=(",", ":")))


def _type_name(prop: dict) -> str:
    t = prop.get("type", "any")
    if t == "array":
        return f"array of {_type_name(prop.get('items', {}))}"
    return t


def _json_objects(text: str):
    """Yield every balanced top-level {...} substring, last first (answers tend to come last)."""
    spans, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            esc = (ch == "\\") and not esc
            in_str = not (ch == '"' and not esc) if not esc or ch != '"' else in_str
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str, esc = True, False
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start:i + 1])
    return reversed(spans)


def parse_json_reply(text: str, schema: type[BaseModel]) -> BaseModel:
    t = _FENCE.sub("", text.strip())
    try:
        return schema.model_validate_json(t)
    except Exception as first:
        for cand in _json_objects(t):
            try:
                return schema.model_validate_json(cand)
            except Exception:
                continue
        raise first


class BedrockClient(ModelClient):
    def __init__(self, model: str = "amazon.nova-micro-v1:0", region: str | None = None, *,
                 profile: str | None = None, max_retries: int = 4, **kw):
        import boto3
        from botocore.config import Config

        cfg = kw.pop("default_config", None) or ModelConfig(model=model)
        super().__init__(default_config=cfg, **kw)
        session = boto3.Session(profile_name=profile, region_name=region)
        self._rt = session.client("bedrock-runtime", config=Config(retries={"max_attempts": max_retries, "mode": "adaptive"}))

    def _params(self, prompt: str, cfg: ModelConfig, schema) -> dict:
        p: dict = {
            "modelId": cfg.model,
            "messages": [{"role": "user", "content": [{"text": prompt + (schema_instruction(schema) if schema else "")}]}],
            "inferenceConfig": {"maxTokens": cfg.max_tokens},
        }
        if cfg.temperature is not None:
            p["inferenceConfig"]["temperature"] = cfg.temperature
        if cfg.system:
            p["system"] = [{"text": cfg.system}]
        return p

    async def _complete(self, prompt: str, cfg: ModelConfig, schema: type[BaseModel] | None) -> Completion:
        resp = await asyncio.to_thread(self._rt.converse, **self._params(prompt, cfg, schema))
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        parsed = parse_json_reply(text, schema) if schema is not None else None
        usage = resp.get("usage", {})
        return Completion(text=text, parsed=parsed, input_tokens=usage.get("inputTokens", 0),
                          output_tokens=usage.get("outputTokens", 0), model=cfg.model, stop_reason=resp.get("stopReason"))
