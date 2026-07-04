# 中文功能说明：远端模型服务客户端，封装 Query Intent、Selector Reranker 和 Crawler Strategy HTTP 调用。

from __future__ import annotations

import json
from pathlib import Path
import ssl
import urllib.error
import urllib.request
from typing import Any

from packages.scholar_core.model_services.ports import (
    CrawlerStrategyPrediction,
    ModelServiceError,
    QueryIntentPrediction,
    QueryRewritePrediction,
)
from packages.scholar_infra.config import ModelServiceSettings
from packages.scholar_core.models import Candidate


def _post_json(base_url: str, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    return _post_json_with_headers(base_url, path, payload, timeout=timeout, headers={})


def _post_json_with_headers(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    headers: dict[str, str],
    verify_ssl: bool = True,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(
        _join_url(base_url, path),
        data=body,
        method="POST",
        headers=request_headers,
    )
    try:
        context = None if verify_ssl else ssl._create_unverified_context()
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ModelServiceError(f"{base_url}{path} returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ModelServiceError(f"{base_url}{path} request failed: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelServiceError(f"{base_url}{path} returned non-object JSON")
    return data


class QueryRewriteServiceClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        model: str,
        timeout: float,
        max_rewrites: int,
        cache_path: str,
        max_tokens: int = 1024,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = _normalize_openai_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_rewrites = max(0, max_rewrites)
        self.max_tokens = max(128, max_tokens)
        self.verify_ssl = verify_ssl
        self.cache_path = Path(cache_path).expanduser()
        self._cache: dict[str, dict[str, Any]] | None = None

    def rewrite(self, text: str, *, context: dict[str, Any] | None = None) -> QueryRewritePrediction:
        clean = " ".join(text.split())
        if not clean or self.max_rewrites <= 0:
            return QueryRewritePrediction([], [], [], {}, cache_hit=False)
        cache_key = _rewrite_cache_key(clean, self.model, context)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return _query_rewrite_prediction(cached, cache_hit=True)
        if not self.api_key:
            raise ModelServiceError("query rewrite service API key is not configured")
        if not self.model:
            raise ModelServiceError("query rewrite service model is not configured")
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You improve academic paper retrieval queries. Return strict JSON only. "
                        "Do not include paper IDs, arXiv IDs, or citations. Do not use known gold labels. "
                        "Prefer concise English retrieval phrases, method names, task names, datasets, and aliases."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": clean,
                            "parsed_context": context or {},
                            "output_schema": {
                                "rewrites": "3-5 diverse search queries",
                                "concepts": "important task/method/dataset concepts",
                                "possible_answer_terms": "possible title-like or method-like terms without paper IDs",
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        data = _post_json_with_headers(
            self.base_url,
            "/chat/completions",
            payload,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            verify_ssl=self.verify_ssl,
        )
        item = _parse_chat_json(data)
        normalized = {
            "rewrites": _clean_rewrite_items(item.get("rewrites"), limit=self.max_rewrites),
            "concepts": _clean_rewrite_items(item.get("concepts"), limit=12),
            "possible_answer_terms": _clean_rewrite_items(item.get("possible_answer_terms"), limit=10),
            "raw": item,
        }
        self._cache_set(cache_key, normalized)
        return _query_rewrite_prediction(normalized, cache_hit=False)

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        cache = self._load_cache()
        value = cache.get(key)
        return value if isinstance(value, dict) else None

    def _cache_set(self, key: str, value: dict[str, Any]) -> None:
        cache = self._load_cache()
        cache[key] = value
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        except OSError:
            return

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            raw = {}
        self._cache = raw if isinstance(raw, dict) else {}
        return self._cache


class QueryIntentServiceClient:
    def __init__(self, base_url: str, *, timeout: float, mode: str = "auto") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.mode = mode

    def predict_one(self, text: str) -> QueryIntentPrediction:
        payload = {"mode": self.mode, "texts": [text]}
        data = _post_json(self.base_url, "/predict", payload, timeout=self.timeout)
        results = data.get("results") or []
        if not results:
            raise ModelServiceError("query intent service returned no results")
        item = results[0]
        if not isinstance(item, dict):
            raise ModelServiceError("query intent result is not an object")
        gate = item.get("gate") or {}
        intent = item.get("intent") or None
        return QueryIntentPrediction(
            gate_label=str(gate.get("label") or "paper_search"),
            gate_score=_optional_float(gate.get("score")),
            intent_label=str(intent.get("label")) if isinstance(intent, dict) and intent.get("label") else None,
            intent_score=_optional_float(intent.get("score")) if isinstance(intent, dict) else None,
            raw=item,
        )


class SelectorRerankerServiceClient:
    def __init__(self, base_url: str, *, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        top_k: int,
    ) -> tuple[list[Candidate], dict[str, Any]]:
        documents = [
            {
                "id": candidate.canonical_id or candidate.paper_id,
                "paper_id": candidate.paper_id,
                "title": candidate.title,
                "abstract": candidate.abstract,
            }
            for candidate in candidates
        ]
        data = _post_json(
            self.base_url,
            "/rerank",
            {"query": query, "top_k": min(top_k, len(documents)), "documents": documents},
            timeout=self.timeout,
        )
        results = data.get("results") or []
        by_key = {
            candidate.canonical_id or candidate.paper_id: candidate
            for candidate in candidates
        }
        reranked: list[Candidate] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or item.get("paper_id") or "")
            candidate = by_key.get(key)
            if candidate is None:
                continue
            score = _optional_float(item.get("score"))
            if score is None:
                continue
            heuristic_score = candidate.final_score
            alias_bonus = _optional_float(candidate.metadata.get("soft_alias_bonus")) or 0.0
            strong_alias_bonus = _optional_float(candidate.metadata.get("strong_alias_bonus")) or 0.0
            heuristic_weight = min(0.75, 0.25 + 0.35 * alias_bonus + 0.35 * strong_alias_bonus)
            selector_weight = 1.0 - heuristic_weight
            candidate.raw_scores["selector_reranker"] = score
            candidate.metadata["selector_reranker_relevant"] = bool(item.get("relevant"))
            candidate.final_score = selector_weight * score + heuristic_weight * heuristic_score
            candidate.relevance = "highly_relevant" if item.get("relevant") else candidate.relevance
            reranked.append(candidate)
        seen = {id(candidate) for candidate in reranked}
        remainder = [candidate for candidate in candidates if id(candidate) not in seen]
        reranked.extend(sorted(remainder, key=lambda candidate: candidate.final_score, reverse=True))
        reranked = _promote_strong_aliases(reranked)
        return reranked[:top_k], {"count": data.get("count"), "threshold": data.get("threshold")}


class CrawlerStrategyServiceClient:
    def __init__(self, base_url: str, *, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(self, query: str, candidate: Candidate, *, sections: list[str]) -> CrawlerStrategyPrediction:
        data = _post_json(
            self.base_url,
            "/predict",
            {
                "query": query,
                "title": candidate.title,
                "abstract": candidate.abstract,
                "sections": sections,
            },
            timeout=self.timeout,
        )
        raw_sections = data.get("sections") or []
        return CrawlerStrategyPrediction(
            prediction=str(data.get("prediction") or ""),
            parse_success=bool(data.get("parse_success")),
            sections=[str(item) for item in raw_sections if str(item).strip()],
            raw=data,
        )


class ModelServices:
    def __init__(
        self,
        *,
        query_intent: QueryIntentServiceClient | None,
        query_rewriter: QueryRewriteServiceClient | None,
        selector_reranker: SelectorRerankerServiceClient | None,
        crawler_strategy: CrawlerStrategyServiceClient | None,
        selector_candidate_limit: int,
        selector_pool_limit: int,
        crawler_top_n: int,
    ) -> None:
        self.query_intent = query_intent
        self.query_rewriter = query_rewriter
        self.selector_reranker = selector_reranker
        self.crawler_strategy = crawler_strategy
        self.selector_candidate_limit = selector_candidate_limit
        self.selector_pool_limit = selector_pool_limit
        self.crawler_top_n = crawler_top_n

    @classmethod
    def from_settings(cls, settings: ModelServiceSettings) -> "ModelServices":
        if not settings.enabled:
            return cls(
                query_intent=None,
                query_rewriter=None,
                selector_reranker=None,
                crawler_strategy=None,
                selector_candidate_limit=settings.selector_reranker_candidate_limit,
                selector_pool_limit=settings.selector_reranker_pool_limit,
                crawler_top_n=0,
            )
        return cls(
            query_intent=QueryIntentServiceClient(
                settings.query_intent_url,
                timeout=settings.timeout_sec,
                mode=settings.query_intent_mode,
            )
            if settings.query_intent_enabled
            else None,
            query_rewriter=QueryRewriteServiceClient(
                settings.query_rewrite_base_url,
                api_key=settings.query_rewrite_api_key,
                model=settings.query_rewrite_model,
                timeout=settings.query_rewrite_timeout_sec,
                max_rewrites=settings.query_rewrite_max_rewrites,
                max_tokens=settings.query_rewrite_max_tokens,
                cache_path=settings.query_rewrite_cache_path,
                verify_ssl=settings.query_rewrite_verify_ssl,
            )
            if settings.query_rewrite_enabled
            else None,
            selector_reranker=SelectorRerankerServiceClient(
                settings.selector_reranker_url,
                timeout=settings.timeout_sec,
            )
            if settings.selector_reranker_enabled
            else None,
            crawler_strategy=CrawlerStrategyServiceClient(
                settings.crawler_strategy_url,
                timeout=settings.timeout_sec,
            )
            if settings.crawler_strategy_enabled and settings.crawler_strategy_top_n > 0
            else None,
            selector_candidate_limit=settings.selector_reranker_candidate_limit,
            selector_pool_limit=settings.selector_reranker_pool_limit,
            crawler_top_n=settings.crawler_strategy_top_n,
        )

    def enabled_names(self) -> list[str]:
        names: list[str] = []
        if self.query_intent:
            names.append("query_intent")
        if self.query_rewriter:
            names.append("query_rewrite")
        if self.selector_reranker:
            names.append("selector_reranker")
        if self.crawler_strategy:
            names.append("crawler_strategy")
        return names


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _promote_strong_aliases(candidates: list[Candidate], *, protected_head: int = 30) -> list[Candidate]:
    promoted = [
        candidate
        for candidate in candidates
        if max(
            _optional_float(candidate.metadata.get("soft_alias_bonus")) or 0.0,
            _optional_float(candidate.metadata.get("strong_alias_bonus")) or 0.0,
        )
        >= 0.9
    ]
    if not promoted:
        return candidates
    promoted_ids = {id(candidate) for candidate in promoted}
    base = [candidate for candidate in candidates if id(candidate) not in promoted_ids]
    promoted.sort(key=lambda candidate: candidate.final_score, reverse=True)
    return base[:protected_head] + promoted + base[protected_head:]


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _normalize_openai_base_url(base_url: str) -> str:
    value = (base_url or "").strip().rstrip("/")
    if not value:
        return "http://127.0.0.1:80/v1-openai"
    lowered = value.lower()
    if lowered.endswith("/chat/completions"):
        return value[: -len("/chat/completions")]
    if lowered.endswith("/v1") or lowered.endswith("/v1-openai"):
        return value
    return value + "/v1-openai"


def _rewrite_cache_key(text: str, model: str, context: dict[str, Any] | None) -> str:
    import hashlib

    body = json.dumps({"text": text, "model": model, "context": context or {}}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _parse_chat_json(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ModelServiceError("query rewrite service returned no chat choices")
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ModelServiceError("query rewrite service returned empty content")
    try:
        parsed = json.loads(_extract_json_object(_strip_think_block(_strip_code_fence(content))))
    except json.JSONDecodeError as exc:
        raise ModelServiceError(f"query rewrite service returned non-JSON content: {content[:160]}") from exc
    if not isinstance(parsed, dict):
        raise ModelServiceError("query rewrite service JSON must be an object")
    return parsed


def _strip_code_fence(text: str) -> str:
    clean = text.strip()
    if not clean.startswith("```"):
        return clean
    lines = clean.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_think_block(text: str) -> str:
    clean = text.strip()
    marker = "</think>"
    lowered = clean.lower()
    if marker in lowered:
        end = lowered.rfind(marker) + len(marker)
        return clean[end:].strip()
    if lowered.startswith("<think>"):
        start = clean.find("{")
        return clean[start:].strip() if start >= 0 else clean
    return clean


def _extract_json_object(text: str) -> str:
    clean = text.strip()
    if clean.startswith("{"):
        return clean
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        return clean[start : end + 1]
    return clean


def _clean_rewrite_items(value: Any, *, limit: int) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        clean = " ".join(str(item).replace("\n", " ").split())
        if not clean:
            continue
        lowered = clean.lower()
        if lowered in seen or _looks_like_forbidden_rewrite(clean):
            continue
        seen.add(lowered)
        items.append(clean[:180])
        if len(items) >= limit:
            break
    return items


def _looks_like_forbidden_rewrite(value: str) -> bool:
    lowered = value.lower()
    return bool(
        "arxiv:" in lowered
        or "doi:" in lowered
        or any(char.isdigit() for char in lowered[:12] if char not in {"-", " "})
    )


def _query_rewrite_prediction(value: dict[str, Any], *, cache_hit: bool) -> QueryRewritePrediction:
    return QueryRewritePrediction(
        rewrites=_clean_rewrite_items(value.get("rewrites"), limit=8),
        concepts=_clean_rewrite_items(value.get("concepts"), limit=12),
        possible_answer_terms=_clean_rewrite_items(value.get("possible_answer_terms"), limit=10),
        raw=value.get("raw") if isinstance(value.get("raw"), dict) else value,
        cache_hit=cache_hit,
    )
