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
        profile = query_profile_kind(intent)
        budget = profile_retrieval_budget(profile, self.per_query_top_k)
        for index, sub_query in enumerate(intent.sub_queries):
            top_k = self._sub_query_top_k(index, budget)
            title_weight = 1.25 if index == 0 else 1.12
            dense_weight = 1.04
            chunk_weight = 1.05 if _looks_like_constraint_query(sub_query) else 1.0
            sparse_weight = 1.08 if index <= 2 or _looks_like_constraint_query(sub_query) else 0.92
            sparse_top_k = top_k if index <= 3 else max(30, int(top_k * 0.9))
            if profile in {"auto_locator", "foundational_or_origin", "method_or_dataset"}:
                title_weight += 0.08
                dense_weight += 0.16
            if profile in {"real_multi_answer", "survey_or_list"}:
                chunk_weight += 0.08
                sparse_weight += 0.04
            if profile == "dataset_or_benchmark":
                title_weight += 0.05
                dense_weight += 0.12
                sparse_weight += 0.06
            if index < budget["title_queries"]:
                actions.append(SearchAction("local_title_bm25", sub_query, top_k, title_weight))
            if index < budget["chunk_queries"]:
                actions.append(SearchAction("local_chunk_bm25", sub_query, top_k, chunk_weight))
            if index < budget["sparse_queries"]:
                actions.append(SearchAction("local_tfidf", sub_query, sparse_top_k, sparse_weight))
            if index < budget["dense_queries"]:
                actions.append(SearchAction("qdrant_dense_paper", sub_query, budget["dense_top_k"], dense_weight))
            if index < budget["sparse_paper_queries"]:
                actions.append(
                    SearchAction("qdrant_sparse_paper", sub_query, budget["sparse_paper_top_k"], dense_weight * 0.72)
                )
        concept_query = _neo4j_concept_query(intent)
        if concept_query:
            actions.append(SearchAction("neo4j_concept", concept_query, budget["concept_top_k"], 0.72))
        alias_query = _neo4j_alias_query(intent) or concept_query
        if alias_query and budget["alias_enabled"]:
            alias_weight = 0.92 if profile in {"auto_locator", "method_or_dataset", "dataset_or_benchmark"} else 0.86
            actions.append(SearchAction("neo4j_alias", alias_query, budget["alias_top_k"], alias_weight))
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
                "query_profile": profile,
                "retrieval_budget": budget,
            },
        )

    def _sub_query_top_k(self, index: int, budget: dict[str, int | bool]) -> int:
        if index <= 3:
            return int(budget["lexical_top_k"])
        return max(30, int(int(budget["lexical_top_k"]) * 0.8))


def _looks_like_constraint_query(query: str) -> bool:
    return any(
        marker in query.lower()
        for marker in (
            "analysis",
            "better than",
            "degradation",
            "first proposed",
            "fact checking",
            "factual consistency",
            "few-shot learning",
            "machine-generated text detection",
            "inverse propensity",
            "learning to rank",
            "search ranking",
            "mathematical reasoning",
            "theorem proving",
            "mask classification",
            "multilingual evaluation",
            "cross-lingual",
            "scientific literature review",
            "research synthesis",
            "data-efficient pretraining",
            "data pruning",
            "data selection",
            "data deduplication",
            "less training data",
            "object goal navigation",
            "feature matching",
            "dense correspondence",
            "image animation",
            "controllable video generation",
            "ray-based rendering",
            "novel view synthesis",
            "robustness certification",
            "certified robustness",
            "controlled text generation",
            "state space models",
            "computer control",
            "gameplay",
            "image restoration",
            "image deblurring",
            "dual-pixel",
            "defocus deblurring",
            "direct preference optimization",
            "game playing",
            "computer games",
            "pretraining data",
            "regret bounds",
            "markov decision process",
            "dense labels",
            "semantic tokens",
            "target network",
        )
    )


def _neo4j_concept_query(intent: QueryIntent) -> str:
    terms: list[str] = []
    terms.extend(intent.must_have_constraints[:6])
    terms.extend(intent.research_field[:5])
    terms.extend(intent.soft_constraints[:10])
    for query in intent.sub_queries[:2]:
        terms.extend(query.split()[:12])
    useful = [
        term
        for term in _unique(terms)
        if len(term) >= 4 and term.lower() not in {"analysis", "approach", "comparison", "study"}
    ]
    return " | ".join(useful[:20])


def _neo4j_alias_query(intent: QueryIntent) -> str:
    terms: list[str] = []
    terms.extend(intent.must_have_constraints[:8])
    terms.extend(intent.soft_constraints[:10])
    for query in intent.sub_queries[:3]:
        terms.append(query)
    useful = [
        term
        for term in _unique(terms)
        if len(term) >= 4 and term.lower() not in {"analysis", "approach", "comparison", "study"}
    ]
    return " | ".join(useful[:18])


def query_profile_kind(intent: QueryIntent) -> str:
    text = " ".join(
        [
            intent.main_intent,
            *intent.research_field,
            *intent.must_have_constraints,
            *intent.soft_constraints,
            *intent.sub_queries[:2],
        ]
    ).lower()
    if any(
        marker in text
        for marker in (
            "known as",
            "called",
            "which paper",
            "which work",
            "what work",
            "first proposed",
            "introduced",
        )
    ):
        return "auto_locator"
    if any(marker in text for marker in ("gave rise", "origin", "foundational")):
        return "foundational_or_origin"
    if any(marker in text for marker in ("dataset", "datasets", "benchmark", "benchmarks", "corpus")):
        return "dataset_or_benchmark"
    if any(marker in text for marker in ("survey", "summaries", "literature review", "list all")):
        return "survey_or_list"
    if any(marker in text for marker in ("papers about", "all papers", "provide related papers", "related papers")):
        return "real_multi_answer"
    if any(marker in text for marker in ("method", "algorithm")):
        return "method_or_dataset"
    if any(marker in text for marker in ("better than", "claim", "negative impact")):
        return "comparison_or_claim"
    return "real_multi_answer"


def profile_retrieval_budget(profile: str, per_query_top_k: int) -> dict[str, int | bool]:
    base = {
        "title_queries": 4,
        "chunk_queries": 4,
        "sparse_queries": 4,
        "dense_queries": 3,
        "sparse_paper_queries": 3,
        "lexical_top_k": per_query_top_k,
        "dense_top_k": 70,
        "sparse_paper_top_k": 100,
        "concept_top_k": min(45, per_query_top_k),
        "alias_top_k": min(50, per_query_top_k),
        "alias_enabled": profile in {"auto_locator", "method_or_dataset", "dataset_or_benchmark", "foundational_or_origin"},
        "graph_expansion_neighbors": 30,
        "diversity_protected": 4,
        "second_round_dense_queries": 3,
        "second_round_dense_top_k": 60,
    }
    overrides: dict[str, dict[str, int | bool]] = {
        "auto_locator": {
            "title_queries": 4,
            "chunk_queries": 3,
            "sparse_queries": 4,
            "dense_queries": 3,
            "sparse_paper_queries": 4,
            "dense_top_k": 100,
            "sparse_paper_top_k": 140,
            "alias_top_k": min(70, max(50, per_query_top_k)),
            "alias_enabled": True,
            "diversity_protected": 8,
            "second_round_dense_queries": 2,
            "second_round_dense_top_k": 70,
        },
        "real_multi_answer": {
            "title_queries": 4,
            "chunk_queries": 6,
            "sparse_queries": 6,
            "dense_queries": 4,
            "sparse_paper_queries": 5,
            "dense_top_k": 80,
            "sparse_paper_top_k": 140,
            "alias_enabled": False,
            "graph_expansion_neighbors": 45,
            "diversity_protected": 3,
            "second_round_dense_queries": 4,
            "second_round_dense_top_k": 80,
        },
        "survey_or_list": {
            "title_queries": 4,
            "chunk_queries": 6,
            "sparse_queries": 6,
            "dense_queries": 4,
            "sparse_paper_queries": 5,
            "dense_top_k": 100,
            "sparse_paper_top_k": 160,
            "alias_enabled": False,
            "graph_expansion_neighbors": 50,
            "diversity_protected": 3,
            "second_round_dense_queries": 4,
            "second_round_dense_top_k": 80,
        },
        "dataset_or_benchmark": {
            "dense_queries": 4,
            "sparse_paper_queries": 4,
            "dense_top_k": 100,
            "sparse_paper_top_k": 150,
            "alias_top_k": min(80, max(50, per_query_top_k)),
            "alias_enabled": True,
            "concept_top_k": min(70, max(45, per_query_top_k)),
            "second_round_dense_top_k": 80,
        },
        "foundational_or_origin": {
            "title_queries": 4,
            "chunk_queries": 3,
            "dense_queries": 3,
            "sparse_paper_queries": 4,
            "dense_top_k": 100,
            "sparse_paper_top_k": 160,
            "alias_top_k": min(70, max(50, per_query_top_k)),
            "alias_enabled": True,
            "graph_expansion_neighbors": 50,
            "second_round_dense_queries": 2,
            "second_round_dense_top_k": 70,
        },
        "comparison_or_claim": {
            "dense_queries": 2,
            "dense_top_k": 60,
            "alias_enabled": False,
            "second_round_dense_queries": 2,
            "second_round_dense_top_k": 60,
        },
    }
    base.update(overrides.get(profile, {}))
    return base


def _dense_paper_top_k(profile: str) -> int:
    budgets = {
        "auto_locator": 100,
        "real_multi_answer": 80,
        "survey_or_list": 100,
        "method_or_dataset": 80,
        "dataset_or_benchmark": 100,
        "foundational_or_origin": 100,
        "comparison_or_claim": 60,
    }
    return budgets.get(profile, 60)


def _dense_query_limit(profile: str) -> int:
    limits = {
        "auto_locator": 3,
        "real_multi_answer": 4,
        "survey_or_list": 4,
        "method_or_dataset": 3,
        "dataset_or_benchmark": 4,
        "foundational_or_origin": 3,
        "comparison_or_claim": 2,
    }
    return limits.get(profile, 2)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = " ".join(str(value).split())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result
