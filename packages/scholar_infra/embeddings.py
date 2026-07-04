# 中文功能说明：统一 dense embedding 客户端，支持本地 sentence-transformers 和 OpenAI-compatible 服务。

from __future__ import annotations

import json
from math import sqrt
import ssl
import urllib.error
import urllib.request
from typing import Any, Protocol


class DenseEmbedder(Protocol):
    def encode_one(self, text: str) -> list[float]:
        ...

    def encode_batch(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        ...


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str, *, device: str = "") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        kwargs = {"device": device} if device else {}
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode_one(self, text: str) -> list[float]:
        return self.encode_batch([text], batch_size=1)[0]

    def encode_batch(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return [[float(value) for value in vector] for vector in vectors]


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        verify_ssl: bool = True,
        normalize: bool = True,
    ) -> None:
        if not base_url:
            raise RuntimeError("DENSE_EMBEDDING_BASE_URL or GPUSTACK_BASE_URL is required")
        if not api_key:
            raise RuntimeError("DENSE_EMBEDDING_API_KEY or GPUSTACK_API_KEY is required")
        if not model:
            raise RuntimeError("DENSE_EMBEDDING_MODEL is required")
        self.base_url = _normalize_embedding_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = max(0.1, float(timeout))
        self.verify_ssl = verify_ssl
        self.normalize = normalize

    def encode_one(self, text: str) -> list[float]:
        return self.encode_batch([text], batch_size=1)[0]

    def encode_batch(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        results: list[list[float]] = []
        for start in range(0, len(texts), max(1, int(batch_size))):
            chunk = texts[start : start + max(1, int(batch_size))]
            results.extend(self._request_embeddings(chunk))
        return results

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/embeddings",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            context = None if self.verify_ssl else ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=self.timeout, context=context) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"embedding service returned {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"embedding service request failed: {exc}") from exc
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise RuntimeError("embedding service returned an unexpected data array")
        ordered = sorted(items, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        vectors: list[list[float]] = []
        for item in ordered:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise RuntimeError("embedding service returned an item without embedding")
            vector = [float(value) for value in item["embedding"]]
            vectors.append(_normalize_vector(vector) if self.normalize else vector)
        return vectors


def build_dense_embedder(
    settings: Any,
    *,
    model_name: str | None = None,
    device: str | None = None,
) -> DenseEmbedder:
    backend = str(getattr(settings, "dense_embedding_backend", "sentence_transformers") or "").strip().lower()
    model = model_name or getattr(settings, "dense_embedding_model", "")
    if backend in {"sentence_transformers", "sentence-transformers", "local"}:
        return SentenceTransformersEmbedder(model, device=device or getattr(settings, "dense_embedding_device", ""))
    if backend in {"openai", "openai_compatible", "openai-compatible", "gpustack"}:
        return OpenAICompatibleEmbedder(
            getattr(settings, "dense_embedding_base_url", ""),
            api_key=getattr(settings, "dense_embedding_api_key", ""),
            model=model,
            timeout=float(getattr(settings, "dense_embedding_timeout_sec", 120.0)),
            verify_ssl=bool(getattr(settings, "dense_embedding_verify_ssl", True)),
        )
    raise RuntimeError(f"unsupported dense embedding backend: {backend}")


def _normalize_embedding_base_url(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    lowered = value.lower()
    if lowered.endswith("/embeddings"):
        return value[: -len("/embeddings")]
    if lowered.endswith("/v1") or lowered.endswith("/v1-openai"):
        return value
    return value + "/v1"


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [float(value / norm) for value in vector]
