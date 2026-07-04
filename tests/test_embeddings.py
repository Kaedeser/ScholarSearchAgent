from __future__ import annotations

import json
from types import SimpleNamespace

from packages.scholar_infra.embeddings import OpenAICompatibleEmbedder, build_dense_embedder


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_embedder_calls_embeddings_endpoint(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=0, context=None):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 3.0, 4.0]},
                    {"index": 0, "embedding": [3.0, 4.0, 0.0]},
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedder = OpenAICompatibleEmbedder(
        "https://ai.wust.edu.cn/gpustack/v1",
        api_key="secret",
        model="qwen3-embedding-4b",
        timeout=12,
    )
    vectors = embedder.encode_batch(["first", "second"], batch_size=2)

    assert captured["url"] == "https://ai.wust.edu.cn/gpustack/v1/embeddings"
    assert captured["timeout"] == 12
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"] == {"model": "qwen3-embedding-4b", "input": ["first", "second"]}
    assert vectors == [[0.6, 0.8, 0.0], [0.0, 0.6, 0.8]]


def test_build_dense_embedder_accepts_gpustack_backend(monkeypatch) -> None:
    created = {}

    class FakeEmbedder:
        def __init__(self, base_url, *, api_key, model, timeout, verify_ssl):
            created.update(
                {
                    "base_url": base_url,
                    "api_key": api_key,
                    "model": model,
                    "timeout": timeout,
                    "verify_ssl": verify_ssl,
                }
            )

    monkeypatch.setattr("packages.scholar_infra.embeddings.OpenAICompatibleEmbedder", FakeEmbedder)

    settings = SimpleNamespace(
        dense_embedding_backend="gpustack",
        dense_embedding_base_url="https://ai.wust.edu.cn/gpustack/v1",
        dense_embedding_api_key="k",
        dense_embedding_model="qwen3-embedding-4b",
        dense_embedding_timeout_sec=30.0,
        dense_embedding_verify_ssl=True,
    )

    embedder = build_dense_embedder(settings)

    assert isinstance(embedder, FakeEmbedder)
    assert created["model"] == "qwen3-embedding-4b"
    assert created["base_url"] == "https://ai.wust.edu.cn/gpustack/v1"
