# 中文功能说明：Qdrant 客户端，负责稀疏向量集合创建、写入和检索。

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
import zlib
from typing import Any


POINT_NAMESPACE = uuid.UUID("6ed3b6e7-7d4e-4456-95f8-9e98a9de4ac0")


class QdrantClient:
    def __init__(self, base_url: str, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["api-key"] = api_key

    def request(self, method: str, path: str, body: Any | None = None) -> Any:
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload) if payload else None
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Qdrant {method} {path} failed: {exc.code} {payload}") from exc

    def health(self) -> Any:
        return self.request("GET", "/")

    def collections(self) -> Any:
        return self.request("GET", "/collections")

    def collection(self, name: str) -> Any:
        return self.request("GET", f"/collections/{name}")

    def collection_exists(self, name: str) -> bool:
        try:
            self.collection(name)
            return True
        except RuntimeError as exc:
            if "404" in str(exc):
                return False
            raise

    def delete_collection(self, name: str) -> Any:
        if not self.collection_exists(name):
            return {"result": True, "missing": True}
        return self.request("DELETE", f"/collections/{name}")

    def create_collection(self, name: str, vector_size: int, distance: str = "Cosine") -> Any:
        body = {
            "vectors": {"size": vector_size, "distance": distance},
            "optimizers_config": {"default_segment_number": 2},
        }
        return self.request("PUT", f"/collections/{name}", body)

    def create_sparse_collection(self, name: str, sparse_vector_name: str = "text") -> Any:
        body = {
            "vectors": {},
            "sparse_vectors": {sparse_vector_name: {}},
            "optimizers_config": {"default_segment_number": 2, "indexing_threshold": 0},
        }
        return self.request("PUT", f"/collections/{name}", body)

    def init_sparse_collection(
        self,
        name: str,
        *,
        sparse_vector_name: str = "text",
        reset: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if reset:
            result["delete"] = self.delete_collection(name)
        if self.collection_exists(name):
            result["collection"] = {"exists": True}
        else:
            result["collection"] = self.create_sparse_collection(name, sparse_vector_name)
        return result

    def upsert_points(self, collection: str, points: list[dict[str, Any]], *, wait: bool = True) -> Any:
        wait_value = "true" if wait else "false"
        return self.request("PUT", f"/collections/{collection}/points?wait={wait_value}", {"points": points})

    def search_sparse(
        self,
        collection: str,
        query_text: str,
        *,
        vector_size: int,
        sparse_vector_name: str = "text",
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        vector = lexical_sparse_vector(query_text, vector_size)
        body = {
            "vector": {"name": sparse_vector_name, "vector": vector},
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        result = self.request("POST", f"/collections/{collection}/points/search", body)
        return (result or {}).get("result") or []


def lexical_sparse_vector(text: str, dimensions: int) -> dict[str, list[int] | list[float]]:
    weights: dict[int, float] = {}
    for token in str(text or "").lower().split():
        clean = "".join(ch for ch in token if ch.isalnum() or ch in "-_")
        if not clean:
            continue
        index = zlib.crc32(clean.encode("utf-8")) % dimensions
        weights[index] = weights.get(index, 0.0) + 1.0
    indices = sorted(weights)
    return {"indices": indices, "values": [round(weights[index], 6) for index in indices]}


def qdrant_point_from_chunk(
    row: dict[str, Any],
    *,
    vector_size: int,
    sparse_vector_name: str = "text",
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid5(POINT_NAMESPACE, row["chunk_id"])),
        "vector": {sparse_vector_name: lexical_sparse_vector(str(row.get("text") or ""), vector_size)},
        "payload": {
            "chunk_id": row["chunk_id"],
            "paper_id": row["paper_id"],
            "chunk_index": row.get("chunk_index"),
            "chunk_type": row.get("chunk_type"),
            "section_title": row.get("section_title"),
            "source": row.get("source"),
            "token_estimate": row.get("token_estimate"),
        },
    }
