# 中文功能说明：Elasticsearch 客户端，负责索引初始化、批量写入和论文/章节检索。

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from packages.scholar_core.retrieval.weighted_query import analyze_weighted_query
from packages.scholar_ingest.io_utils import read_jsonl


def _auth_header(username: str, password: str) -> dict[str, str]:
    if not username:
        return {}
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


PHRASE_HINTS = (
    "hubert codes",
    "semantic tokens",
    "chain-of-thought prompting",
    "chain of thought prompting",
    "cot prompting",
    "vision-language",
    "vision-language model",
    "vision-language models",
    "prompt learning",
    "prompt tuning",
    "fine-tuning",
    "clip-adapter",
    "latent graphs",
    "graph structure learning",
    "neural radiance fields",
    "nerf",
    "gaussian noise",
    "rgb space",
    "pixel-wise uncertainty",
    "diffusion models",
    "diffusion model alignment",
    "planning with diffusion",
    "video diffusion",
    "video diffusion models",
    "video diffusion alignment",
    "reward gradients",
    "human feedback",
    "text-to-video diffusion",
    "latent space",
    "semantic arithmetic",
    "equivariance",
    "api-use",
    "toolalpaca",
    "toollm",
    "code evaluation",
    "humaneval",
    "mbpp",
    "code_contests",
    "hotpotqa",
    "hotpotqa dataset",
    "multi-hop question answering",
    "visual-llm",
    "visual llm",
    "mixture of experts",
    "sparse mixture of experts",
    "moe architecture",
    "autoregressive transformer",
    "autoregressive transformers",
    "autoregressive video generation",
    "video generation",
    "video synthesis",
    "commonsense machine translation",
    "commonsense reasoning",
    "hallucination mitigation",
    "vision-language hallucination",
    "image captioning hallucination",
    "video captioning hallucination",
    "factually augmented rlhf",
    "preference fine-tuning",
    "fine-grained reward modeling",
    "correctional human feedback",
    "quantization-aware pretraining",
    "quantized language model pretraining",
    "dpo vision-language models",
    "preference optimization vision-language",
    "identity-preserving video generation",
    "personalized video generation",
    "quality-preserving watermarking",
    "watermark robustness",
    "robot task planning",
    "embodied decision making",
    "planning benchmarks",
    "llm agents planning",
    "computer control",
    "game playing",
    "open-world game agents",
    "action role-playing games",
    "gameplay videos",
    "long chain-of-thought data",
    "synthetic reasoning data",
    "reasoning data generation",
    "theorem proving data",
    "theorem and proof data",
    "large-scale theorem proving data",
    "theorem proving data synthesis",
    "proof data",
    "proof data synthesis",
    "mathematical reasoning data",
    "preference data",
    "response comparison",
    "llm reranking",
    "large language model reranker",
    "document reranking",
    "passage ranking",
    "zero-shot rankers",
    "few-shot information extraction",
    "few-shot information extractor",
    "biomedical information extraction",
    "sequence labeling",
    "multimodal scaling laws",
    "mixed-modal language models",
    "contrastive language-image learning",
    "rlhf generalisation diversity",
    "reward collapse",
    "vanishing gradients",
    "reinforcement finetuning",
    "data pruning for pretraining",
    "data-efficient llms",
    "less training data",
    "fewer data",
    "deduplicating training data",
    "long video captioning",
    "long-form video understanding",
    "hour-long videos",
    "dense video captions",
    "long video comprehension",
    "pointnet",
    "point sets",
    "3d classification",
    "point cloud segmentation",
    "uniaudio",
    "audio foundation model",
    "audio generation",
    "bayesian experimental design",
    "contextual optimisation",
    "squeeze recover relabel",
    "source-free domain adaptation",
    "source domain data estimation",
    "perceptual grouping",
    "contrastive vision-language models",
    "xlnet",
    "cloze-driven pretraining",
    "self-attention networks",
    "oversmoothing graph neural networks",
    "over-smoothing bert",
    "multi-domain semantic segmentation",
    "mseg",
    "multi-dataset pretraining",
    "video aesthetics",
    "synthetic data",
    "synthesis data",
    "quantum monte carlo",
    "financial tasks",
    "llm agents",
    "end-to-end object detection with transformers",
    "q-align",
    "visual scoring",
    "discrete text-defined levels",
    "automated readability assessment",
    "text readability",
    "fast structured decoding",
    "fast structured decoding for sequence models",
    "post hoc explainers",
    "large language models post hoc explainers",
    "unsupervised representation learning with deep convolutional generative adversarial networks",
    "deep feature interpolation",
    "deep feature interpolation for image content changes",
    "active view selection",
    "speech tokens",
    "discrete speech units",
    "mask classification",
    "mask classification based",
    "instance-level segmentation",
    "instance segmentation",
    "panoptic segmentation",
    "inverse propensity score",
    "inverse propensity scoring",
    "self-normalized ips",
    "selection bias",
    "target network",
    "target networks",
    "deep q-learning",
    "in-context learning",
    "scaling law",
    "scaling laws",
    "video-text",
    "image-text",
    "vision-language",
    "reconstruction error",
    "discriminator loss",
    "anomaly score",
    "token-level edit",
    "edit operation prediction",
    "seq2edit",
    "supervised fine-tuned",
    "supervised fine-tuning",
    "reinforcement learning",
    "rlhf",
)

LOCATOR_PATTERNS = (
    r"\bwhich\s+paper\s+(?:first\s+)?(?:proposed|introduced|implemented)\b",
    r"\bwhich\s+work\s+(?:first\s+)?(?:proposed|introduced|implemented)\b",
    r"\bwhat\s+(?:work|paper)\s+(?:proposes|proposed|introduced|implemented)\b",
    r"\bfind\s+(?:the\s+)?(?:paper|work|study)\s+(?:that\s+)?(?:proposed|introduced|implemented)\b",
    r"\bfirst\s+(?:proposed|introduced|implemented)\b",
)

GENERIC_QUERY_TERMS = {
    "about",
    "based",
    "find",
    "first",
    "known",
    "as",
    "assumed",
    "context",
    "focus",
    "focused",
    "for",
    "in",
    "introduced",
    "method",
    "methods",
    "model",
    "models",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "proposed",
    "proposes",
    "research",
    "studies",
    "study",
    "that",
    "the",
    "to",
    "what",
    "who",
    "which",
    "with",
    "work",
    "works",
}


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
            "track_total_hits": False,
            "_source": ["paper_id", "arxiv_id", "title", "abstract", "year", "venue", "citation_count", "source"],
            "query": _paper_recall_query(query),
        }
        rescore = _paper_rescore(query, top_k)
        if rescore:
            body["rescore"] = rescore
        result = self.request("POST", f"/{index_name}/_search", body)
        return _hits(result)

    def search_chunks(self, index_name: str, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        body = {
            "size": top_k,
            "track_total_hits": False,
            "_source": ["chunk_id", "paper_id", "chunk_index", "chunk_type", "section_title", "text", "source"],
            "query": _chunk_recall_query(query),
        }
        rescore = _chunk_rescore(query, top_k)
        if rescore:
            body["rescore"] = rescore
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


def _paper_recall_query(query: str) -> dict[str, Any]:
    return _recall_bool_query(
        query,
        primary_fields=["title^3", "abstract"],
        clean_fields=["title^5", "abstract^1.4"],
        phrase_title_field="title",
        phrase_body_field="abstract",
        title_abstract_boost=False,
    )


def _chunk_recall_query(query: str) -> dict[str, Any]:
    return _recall_bool_query(
        query,
        primary_fields=["text", "section_title^1.5"],
        clean_fields=["text^1.2", "section_title^2.4"],
        phrase_title_field="section_title",
        phrase_body_field="text",
        title_abstract_boost=True,
    )


def _recall_bool_query(
    query: str,
    *,
    primary_fields: list[str],
    clean_fields: list[str],
    phrase_title_field: str,
    phrase_body_field: str,
    title_abstract_boost: bool,
) -> dict[str, Any]:
    clean_query = _clean_query_for_recall(query)
    phrases = _extract_key_phrases(query)
    weighted = analyze_weighted_query(query, phrase_hints=PHRASE_HINTS)
    should: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": query,
                "fields": primary_fields,
                "type": "best_fields",
                "minimum_should_match": "2<55%",
            }
        }
    ]
    if clean_query:
        should.append(
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": clean_fields,
                    "type": "best_fields",
                    "minimum_should_match": "2<60%",
                    "boost": 1.25,
                }
            }
        )
        if 1 < _term_count(clean_query) <= 8:
            should.append(
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": clean_fields,
                        "type": "cross_fields",
                        "operator": "and",
                        "boost": 1.6 if _looks_like_locator_query(query) else 0.9,
                    }
                }
            )
    for phrase in phrases[:8]:
        should.append({"match_phrase": {phrase_title_field: {"query": phrase, "slop": 0, "boost": 3.2}}})
        should.append({"match_phrase": {phrase_body_field: {"query": phrase, "slop": 1, "boost": 1.1}}})
    should.extend(
        _weighted_term_clauses(
            weighted,
            title_field=phrase_title_field,
            body_field=phrase_body_field,
            title_boost=2.6 if phrase_title_field != phrase_body_field else 1.5,
            body_boost=1.0,
        )
    )
    if title_abstract_boost:
        should.append({"term": {"chunk_type": {"value": "title_abstract", "boost": 0.2}}})
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _paper_rescore(query: str, top_k: int) -> dict[str, Any] | None:
    clean_query = _clean_query_for_recall(query)
    phrases = _extract_key_phrases(query)
    weighted = analyze_weighted_query(query, phrase_hints=PHRASE_HINTS)
    should: list[dict[str, Any]] = []
    if clean_query and clean_query.lower() != query.lower():
        should.append(
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": ["title^4.2", "abstract^1.2"],
                    "type": "best_fields",
                    "minimum_should_match": "2<60%",
                    "boost": 0.8,
                }
            }
        )
    if _looks_like_locator_query(query) and 1 < _term_count(clean_query) <= 8:
        should.append(
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": ["title^3", "abstract"],
                    "type": "cross_fields",
                    "operator": "and",
                    "boost": 1.2,
                }
            }
        )
        should.append({"match": {"title": {"query": clean_query, "operator": "and", "boost": 2.0}}})
    should.extend(_phrase_clauses(phrases, title_field="title", body_field="abstract"))
    should.extend(
        _weighted_term_clauses(
            weighted,
            title_field="title",
            body_field="abstract",
            title_boost=3.4,
            body_boost=1.1,
            phrase_limit=5,
        )
    )
    is_locator = _looks_like_locator_query(query)
    if is_locator:
        for phrase in phrases[:3]:
            should.append({"match_phrase": {"title": {"query": phrase, "slop": 0, "boost": 12.0}}})
    return _rescore(should, top_k=top_k, weight=0.2 if is_locator else 0.03)


def _chunk_rescore(query: str, top_k: int) -> dict[str, Any] | None:
    clean_query = _clean_query_for_recall(query)
    phrases = _extract_key_phrases(query)
    weighted = analyze_weighted_query(query, phrase_hints=PHRASE_HINTS)
    should: list[dict[str, Any]] = []
    if clean_query and clean_query.lower() != query.lower():
        should.append(
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": ["text^1.15", "section_title^2.2"],
                    "type": "best_fields",
                    "minimum_should_match": "2<60%",
                    "boost": 0.7,
                }
            }
        )
    if _looks_like_locator_query(query) and 1 < _term_count(clean_query) <= 8:
        should.append({"match": {"text": {"query": clean_query, "operator": "and", "boost": 1.2}}})
    should.extend(_phrase_clauses(phrases, title_field="section_title", body_field="text"))
    should.extend(
        _weighted_term_clauses(
            weighted,
            title_field="section_title",
            body_field="text",
            title_boost=2.4,
            body_boost=0.9,
            phrase_limit=5,
        )
    )
    should.append({"term": {"chunk_type": {"value": "title_abstract", "boost": 0.25}}})
    return _rescore(should, top_k=top_k, weight=0.02)


def _rescore(should: list[dict[str, Any]], *, top_k: int, weight: float) -> dict[str, Any] | None:
    if not should:
        return None
    return {
        "window_size": max(top_k * 3, top_k),
        "query": {
            "rescore_query": {"bool": {"should": should, "minimum_should_match": 1}},
            "query_weight": 1.0,
            "rescore_query_weight": weight,
        },
    }


def _phrase_clauses(phrases: list[str], *, title_field: str, body_field: str) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for phrase in phrases[:6]:
        clauses.append({"match_phrase": {title_field: {"query": phrase, "slop": 0, "boost": 1.4}}})
        clauses.append({"match_phrase": {body_field: {"query": phrase, "slop": 1, "boost": 0.6}}})
    return clauses


def _weighted_term_clauses(
    weighted,
    *,
    title_field: str,
    body_field: str,
    title_boost: float,
    body_boost: float,
    term_limit: int = 10,
    phrase_limit: int = 6,
) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for term in weighted.terms[:term_limit]:
        boost = round(term.weight, 4)
        clauses.append({"match": {title_field: {"query": term.term, "boost": round(title_boost * boost, 4)}}})
        clauses.append({"match": {body_field: {"query": term.term, "boost": round(body_boost * boost, 4)}}})
    for phrase in weighted.phrases[:phrase_limit]:
        boost = round(phrase.weight, 4)
        clauses.append({"match_phrase": {title_field: {"query": phrase.term, "slop": 0, "boost": round(title_boost * boost, 4)}}})
        clauses.append({"match_phrase": {body_field: {"query": phrase.term, "slop": 1, "boost": round(body_boost * boost * 0.7, 4)}}})
    return clauses


def _extract_key_phrases(query: str) -> list[str]:
    lowered = query.lower()
    phrases: list[str] = []
    for match in re.findall(r'"([^"]{3,120})"', query):
        phrases.append(_normalize_phrase(match))
    for phrase in PHRASE_HINTS:
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", lowered):
            phrases.append(phrase)
    for match in re.findall(r"\b[a-z][a-z0-9]+(?:-[a-z0-9]+)+(?:\s+[a-z][a-z0-9]+){0,2}\b", lowered):
        phrases.append(_normalize_phrase(match))
    return _unique_phrases(phrases)


def _clean_query_for_recall(query: str) -> str:
    lowered = query.lower()
    for pattern in LOCATOR_PATTERNS:
        lowered = re.sub(pattern, " ", lowered)
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", lowered)
    kept = [token for token in tokens if token not in GENERIC_QUERY_TERMS]
    return " ".join(kept[:14])


def _looks_like_locator_query(query: str) -> bool:
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in LOCATOR_PATTERNS)


def _term_count(query: str) -> int:
    return len(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", query.lower()))


def _normalize_phrase(value: str) -> str:
    return " ".join(value.lower().split())


def _unique_phrases(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = _normalize_phrase(value)
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


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
