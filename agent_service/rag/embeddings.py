from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from importlib import import_module
from typing import Any, Protocol, cast

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class MockEmbeddingClient:
    def __init__(self, *, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("EMBEDDING_API_KEY or OPENAI_API_KEY is required.")
        openai_module = import_module("openai")
        openai_client = openai_module.OpenAI
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout_seconds,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = openai_client(**client_kwargs)
        response = client.embeddings.create(model=self.model, input=list(texts))
        vectors: list[list[float]] = []
        for item in response.data:
            vectors.append([float(value) for value in item.embedding])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class GeminiEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings.")
        payload: dict[str, object] = {
            "model": self.model,
            "content": {"parts": [{"text": text}]},
        }
        response = _request_json(
            "POST",
            self._embed_url(),
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        if not isinstance(response, dict):
            raise RuntimeError("Gemini embedding response must be a JSON object.")
        embedding = response.get("embedding")
        if not isinstance(embedding, dict):
            raise RuntimeError("Gemini embedding response is missing embedding.")
        values = embedding.get("values")
        if not isinstance(values, list):
            raise RuntimeError("Gemini embedding response is missing embedding values.")
        return [float(value) for value in values]

    def _embed_url(self) -> str:
        query = urllib.parse.urlencode({"key": self.api_key or ""})
        return f"{self.base_url}/{self.model}:embedContent?{query}"


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    timeout_seconds: float,
) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_http_error_message(exc.code, raw_body)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini embedding request failed: {exc.reason}") from exc
    return json.loads(raw_body) if raw_body else {}


def _http_error_message(status_code: int, raw_body: str) -> str:
    try:
        payload = cast(dict[str, object], json.loads(raw_body))
    except json.JSONDecodeError:
        return f"Gemini embedding request failed with HTTP {status_code}: {raw_body[:200]}"
    error = payload.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "")
        status = str(error.get("status") or error.get("code") or "")
        return (
            f"Gemini embedding request failed with HTTP {status_code}: "
            f"{status} {message}"
        ).strip()
    return f"Gemini embedding request failed with HTTP {status_code}: {raw_body[:200]}"
