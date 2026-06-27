# 中文功能说明：搜索策略规划器，根据解析后的查询意图生成多路召回动作和预算。

from __future__ import annotations

from packages.scholar_core.models import QueryIntent, SearchAction, SearchPlan


class SearchPlanner:
    """Small rule planner for offline demo retrieval."""

    def __init__(
        self,
        *,
        per_query_top_k: int = 60,
        max_api_calls: int = 0,
        max_llm_calls: int = 0,
        max_search_rounds: int = 2,
    ) -> None:
        self.per_query_top_k = per_query_top_k
        self.max_api_calls = max_api_calls
        self.max_llm_calls = max_llm_calls
        self.max_search_rounds = max_search_rounds

    def plan(self, intent: QueryIntent, *, round_number: int = 1) -> SearchPlan:
        actions: list[SearchAction] = []
        for index, sub_query in enumerate(intent.sub_queries):
            top_k = self._sub_query_top_k(index)
            title_weight = 1.25 if index == 0 else 1.12
            chunk_weight = 1.05 if _looks_like_constraint_query(sub_query) else 1.0
            sparse_weight = 0.95 if index <= 2 else 0.82
            actions.append(SearchAction("local_title_bm25", sub_query, top_k, title_weight))
            actions.append(SearchAction("local_chunk_bm25", sub_query, top_k, chunk_weight))
            actions.append(SearchAction("local_tfidf", sub_query, max(20, int(top_k * 0.8)), sparse_weight))
        return SearchPlan(
            round=round_number,
            search_actions=actions,
            expand_citations_for=[],
            budget={
                "max_search_rounds": self.max_search_rounds,
                "max_api_calls": self.max_api_calls,
                "max_llm_calls": self.max_llm_calls,
                "max_candidates_for_selector": 400,
                "max_candidates_for_llm_judge": 0,
            },
        )

    def _sub_query_top_k(self, index: int) -> int:
        if index <= 3:
            return self.per_query_top_k
        return max(30, int(self.per_query_top_k * 0.8))


def _looks_like_constraint_query(query: str) -> bool:
    return any(
        marker in query.lower()
        for marker in (
            "analysis",
            "better than",
            "degradation",
            "first proposed",
            "inverse propensity",
            "mask classification",
            "semantic tokens",
            "target network",
        )
    )
