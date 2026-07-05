# 中文功能说明：模型服务端口定义，隔离核心检索流水线和远端 HTTP 模型客户端。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from packages.scholar_core.models import Candidate


class ModelServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryIntentPrediction:
    gate_label: str
    gate_score: float | None
    intent_label: str | None
    intent_score: float | None
    raw: dict[str, Any]

    @property
    def is_paper_search(self) -> bool:
        return self.gate_label != "non_paper_search"


@dataclass(frozen=True)
class CrawlerStrategyPrediction:
    prediction: str
    parse_success: bool
    sections: list[str]
    raw: dict[str, Any]


@dataclass(frozen=True)
class QueryRewritePrediction:
    rewrites: list[str]
    concepts: list[str]
    possible_answer_terms: list[str]
    raw: dict[str, Any]
    cache_hit: bool = False


class QueryIntentPort(Protocol):
    def predict_one(self, text: str) -> QueryIntentPrediction:
        ...


class QueryRewritePort(Protocol):
    def rewrite(self, text: str, *, context: dict[str, Any] | None = None) -> QueryRewritePrediction:
        ...


class SelectorRerankerPort(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        top_k: int,
    ) -> tuple[list[Candidate], dict[str, Any]]:
        ...


class CrawlerStrategyPort(Protocol):
    def predict(self, query: str, candidate: Candidate, *, sections: list[str]) -> CrawlerStrategyPrediction:
        ...


class ModelServicesPort(Protocol):
    query_intent: QueryIntentPort | None
    query_rewriter: QueryRewritePort | None
    selector_reranker: SelectorRerankerPort | None
    crawler_strategy: CrawlerStrategyPort | None
    selector_candidate_limit: int
    selector_pool_limit: int
    selector_protected_head: int
    crawler_top_n: int

    def enabled_names(self) -> list[str]:
        ...


class DisabledModelServices:
    query_intent: QueryIntentPort | None = None
    query_rewriter: QueryRewritePort | None = None
    selector_reranker: SelectorRerankerPort | None = None
    crawler_strategy: CrawlerStrategyPort | None = None
    selector_candidate_limit = 0
    selector_pool_limit = 0
    selector_protected_head = 0
    crawler_top_n = 0

    def enabled_names(self) -> list[str]:
        return []
