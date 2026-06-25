# 中文功能说明：远端模型服务客户端，封装 Query Intent、Selector Reranker 和 Crawler Strategy HTTP 调用。

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from packages.scholar_core.model_services.ports import (
    CrawlerStrategyPrediction,
    ModelServiceError,
    QueryIntentPrediction,
)
from packages.scholar_infra.config import ModelServiceSettings
from packages.scholar_core.models import Candidate


def _post_json(base_url: str, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
            candidate.raw_scores["selector_reranker"] = score
            candidate.metadata["selector_reranker_relevant"] = bool(item.get("relevant"))
            candidate.final_score = 0.75 * score + 0.25 * heuristic_score
            candidate.relevance = "highly_relevant" if item.get("relevant") else candidate.relevance
            reranked.append(candidate)
        seen = {id(candidate) for candidate in reranked}
        remainder = [candidate for candidate in candidates if id(candidate) not in seen]
        reranked.extend(sorted(remainder, key=lambda candidate: candidate.final_score, reverse=True))
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
        selector_reranker: SelectorRerankerServiceClient | None,
        crawler_strategy: CrawlerStrategyServiceClient | None,
        selector_candidate_limit: int,
        crawler_top_n: int,
    ) -> None:
        self.query_intent = query_intent
        self.selector_reranker = selector_reranker
        self.crawler_strategy = crawler_strategy
        self.selector_candidate_limit = selector_candidate_limit
        self.crawler_top_n = crawler_top_n

    @classmethod
    def from_settings(cls, settings: ModelServiceSettings) -> "ModelServices":
        if not settings.enabled:
            return cls(
                query_intent=None,
                selector_reranker=None,
                crawler_strategy=None,
                selector_candidate_limit=settings.selector_reranker_candidate_limit,
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
            crawler_top_n=settings.crawler_strategy_top_n,
        )

    def enabled_names(self) -> list[str]:
        names: list[str] = []
        if self.query_intent:
            names.append("query_intent")
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
