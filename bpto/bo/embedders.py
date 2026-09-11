"""Embedders: text -> vector. One HTTP implementation covers Voyage and OpenAI-compatible
`/v1/embeddings` (same request/response shape); a hashing embedder is offline for tests/baselines."""
from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
from typing import Sequence

import httpx


class HTTPEmbedder:
    def __init__(self, model: str, base_url: str, api_key: str | None, *, batch_size: int = 64,
                 extra_body: dict | None = None, timeout: float = 60.0, max_retries: int = 4,
                 transport: httpx.AsyncBaseTransport | None = None, path: str = "/embeddings",
                 params: dict | None = None, headers: dict | None = None):
        self.model, self.batch_size, self.extra_body, self.max_retries = model, batch_size, extra_body or {}, max_retries
        self.path, self.params = path, params or {}
        headers = {"Content-Type": "application/json", **(headers or {})}
        if api_key and "api-key" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout, transport=transport)
        self._cache: dict[str, list[float]] = {}

    async def _post(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.model, "input": texts, **self.extra_body}
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._http.post(self.path, json=body, params=self.params)
                if r.status_code in (408, 409, 429) or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in data]
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(min(2 ** attempt, 30))
        raise RuntimeError("unreachable")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            for t, v in zip(batch, await self._post(batch)):
                self._cache[t] = v
        return [self._cache[t] for t in texts]

    async def aclose(self) -> None:
        await self._http.aclose()


class VoyageEmbedder(HTTPEmbedder):
    def __init__(self, model: str = "voyage-3.5", api_key: str | None = None, **kw):
        super().__init__(model, "https://api.voyageai.com/v1", api_key or os.environ.get("VOYAGE_API_KEY"), **kw)


class OpenAIEmbedder(HTTPEmbedder):
    def __init__(self, model: str = "text-embedding-3-small", base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, **kw):
        super().__init__(model, base_url, api_key or os.environ.get("OPENAI_API_KEY"), **kw)


class AzureOpenAIEmbedder(HTTPEmbedder):
    """Azure OpenAI embeddings deployment: {endpoint}/openai/deployments/{deployment}/embeddings?api-version=..."""

    def __init__(self, deployment: str, endpoint: str | None = None, api_key: str | None = None,
                 api_version: str = "2024-10-21", **kw):
        endpoint = (endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]).rstrip("/")
        key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        super().__init__(deployment, f"{endpoint}/openai/deployments/{deployment}", None,
                         params={"api-version": api_version}, headers={"api-key": key} if key else None, **kw)


class HashEmbedder:
    """Offline bag-of-hashed-word-n-grams, L2-normalised. Deterministic; fine for tests and as a baseline."""

    def __init__(self, dim: int = 256, ngrams: tuple[int, ...] = (1, 2)):
        self.dim, self.ngrams = dim, ngrams

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = re.findall(r"\w+|[^\w\s]", text.lower())
        for n in self.ngrams:
            for i in range(len(toks) - n + 1):
                h = int(hashlib.blake2b(" ".join(toks[i:i + n]).encode(), digest_size=8).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 63) else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]
