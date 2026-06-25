# 中文功能说明：搜索策略规划器，根据解析后的查询意图生成多路召回动作和预算。

from __future__ import annotations

from packages.scholar_core.models import QueryIntent, SearchAction, SearchPlan


class SearchPlanner:
    """Small rule planner for offline demo retrieval."""

    def __init__(
        self,
        *,
        per_query_top_k: int = 30,
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
        for sub_query in intent.sub_queries:
            actions.append(SearchAction("local_title_bm25", sub_query, self.per_query_top_k, 1.15))
            actions.append(SearchAction("local_chunk_bm25", sub_query, self.per_query_top_k, 1.0))
            actions.append(SearchAction("local_tfidf", sub_query, self.per_query_top_k, 0.85))
        return SearchPlan(
            round=round_number,
            search_actions=actions,
            expand_citations_for=[],
            budget={
                "max_search_rounds": self.max_search_rounds,
                "max_api_calls": self.max_api_calls,
                "max_llm_calls": self.max_llm_calls,
                "max_candidates_for_selector": 300,
                "max_candidates_for_llm_judge": 0,
            },
        )
