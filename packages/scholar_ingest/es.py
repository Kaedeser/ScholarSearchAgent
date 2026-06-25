# 中文功能说明：Elasticsearch 客户端，负责索引初始化、批量写入和论文/章节检索。

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl


def _auth_header(username: str, password: str) -> dict[str, str]:
    if not username:
        return {}
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


class ElasticsearchClient:
    def __init__(self, base_url: str, username: str = "", password: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json", **_auth_header(username, password)}

    def request(self, method: str, path: str, body: Any | None = None, *, timeout: int = 30) -> Any:
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload) if payload else None
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Elasticsearch {method} {path} failed: {exc.code} {payload}") from exc

    def health(self) -> Any:
        return self.request("GET", "/_cluster/health")

    def index_exists(self, name: str) -> bool:
        req = urllib.request.Request(self.base_url + f"/{name}", method="HEAD", headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            payload = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Elasticsearch HEAD /{name} failed: {exc.code} {payload}") from exc

    def delete_index(self, name: str) -> Any:
        if not self.index_exists(name):
            return {"acknowledged": True, "missing": True}
        return self.request("DELETE", f"/{name}")

    def create_index(self, name: str, mapping: dict[str, Any]) -> Any:
        if self.index_exists(name):
            return {"acknowledged": True, "exists": True}
        return self.request("PUT", f"/{name}", mapping)

    def init_indices(
        self,
        papers_index: str,
        chunks_index: str,
        *,
        reset: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if reset:
            result["delete_papers"] = self.delete_index(papers_index)
            result["delete_chunks"] = self.delete_index(chunks_index)
        result["papers"] = self.create_index(papers_index, PAPERS_MAPPING)
        result["chunks"] = self.create_index(chunks_index, CHUNKS_MAPPING)
        return result

    def count(self, index_name: str) -> int:
        result = self.request("GET", f"/{index_name}/_count")
        return int((result or {}).get("count") or 0)

    def search_papers(self, index_name: str, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "abstract"],
                    "type": "best_fields",
                }
            },
        }
        result = self.request("POST", f"/{index_name}/_search", body)
        return _hits(result)

    def search_chunks(self, index_name: str, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "section_title^1.5"],
                    "type": "best_fields",
                }
            },
        }
        result = self.request("POST", f"/{index_name}/_search", body)
        return _hits(result)

    def bulk_jsonl(self, index_name: str, path: Path, id_field: str, batch_size: int = 500) -> int:
        count = 0
        batch: list[str] = []
        for row in read_jsonl(path):
            doc_id = row[id_field]
            batch.append(json.dumps({"index": {"_index": index_name, "_id": doc_id}}, ensure_ascii=False))
            batch.append(json.dumps(row, ensure_ascii=False))
            count += 1
            if count % batch_size == 0:
                self._send_bulk(batch)
                batch = []
        if batch:
            self._send_bulk(batch)
        return count

    def _send_bulk(self, lines: list[str]) -> None:
        data = ("\n".join(lines) + "\n").encode("utf-8")
        headers = dict(self.headers)
        headers["Content-Type"] = "application/x-ndjson"
        req = urllib.request.Request(self.base_url + "/_bulk", data=data, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        if result.get("errors"):
            raise RuntimeError("Elasticsearch bulk request returned errors")


def _hits(result: Any) -> list[dict[str, Any]]:
    hits = ((result or {}).get("hits") or {}).get("hits") or []
    rows: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source") or {}
        source["_score"] = hit.get("_score")
        source["_index"] = hit.get("_index")
        source["_id"] = hit.get("_id")
        rows.append(source)
    return rows


PAPERS_MAPPING: dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "paper_id": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "title": {"type": "text", "fields": {"raw": {"type": "keyword", "ignore_above": 512}}},
            "abstract": {"type": "text"},
            "year": {"type": "integer"},
            "published_time": {"type": "date"},
            "venue": {"type": "keyword"},
            "authors": {"type": "keyword"},
            "citation_count": {"type": "integer"},
            "source": {"type": "keyword"},
            "fulltext_key": {"type": "keyword", "ignore_above": 1024},
            "has_fulltext": {"type": "boolean"},
        }
    },
}

CHUNKS_MAPPING: dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_type": {"type": "keyword"},
            "section_title": {"type": "text", "fields": {"raw": {"type": "keyword", "ignore_above": 512}}},
            "text": {"type": "text"},
            "token_estimate": {"type": "integer"},
            "source": {"type": "keyword"},
        }
    },
}
