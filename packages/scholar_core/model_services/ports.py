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


class QueryIntentPort(Protocol):
    def predict_one(self, text: str) -> QueryIntentPrediction:
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
    selector_reranker: SelectorRerankerPort | None
    crawler_strategy: CrawlerStrategyPort | None
    selector_candidate_limit: int
    crawler_top_n: int

    def enabled_names(self) -> list[str]:
        ...


class DisabledModelServices:
    query_intent: QueryIntentPort | None = None
    selector_reranker: SelectorRerankerPort | None = None
    crawler_strategy: CrawlerStrategyPort | None = None
    selector_candidate_limit = 0
    crawler_top_n = 0

    def enabled_names(self) -> list[str]:
        return []
