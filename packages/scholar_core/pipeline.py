# 中文功能说明：检索主编排流水线，串联查询理解、召回、排序、覆盖分析、模型服务和结果成本记录。

from __future__ import annotations

from dataclasses import asdict, replace
import re
import time
from typing import Any

from packages.scholar_core.normalization.normalizer import CandidateNormalizer
from packages.scholar_core.citation.planner import CitationExpansionPlanner
from packages.scholar_core.coverage.analyzer import CoverageAnalyzer
from packages.scholar_core.model_services.ports import (
    DisabledModelServices,
    ModelServiceError,
    ModelServicesPort,
    QueryIntentPrediction,
    QueryRewritePrediction,
)
from packages.scholar_core.query_understanding.parser import QueryParser
from packages.scholar_core.ranking.preselector import CandidatePreselector
from packages.scholar_core.ranking.ranker import CandidateRanker
from packages.scholar_core.retrieval.ports import CorpusBackend
from packages.scholar_core.models import Candidate, CoverageReport, QueryIntent, SearchAction, SearchPlan, SearchResponse
from packages.scholar_core.planning.planner import SearchPlanner, profile_retrieval_budget, query_profile_kind
from packages.scholar_core.text import tokenize


SECTION_RE = re.compile(r"\bSection:\s*([^\n]+)", re.IGNORECASE)
DEFAULT_SELECTOR_POOL_LIMIT = 500
DEFAULT_SELECTOR_CANDIDATE_LIMIT = 120
DEFAULT_SELECTOR_PROTECTED_HEAD = 0


class SearchPipeline:
    def __init__(
        self,
        corpus: CorpusBackend,
        *,
        per_query_top_k: int = 30,
        model_services: ModelServicesPort | None = None,
        backend_error: str | None = None,
        academic_search_enabled: bool = False,
        academic_search_provider: str = "semantic_scholar",
        academic_search_query_limit: int = 2,
        academic_search_top_k: int = 20,
    ) -> None:
        self.corpus = corpus
        self.backend_error = backend_error
        self.model_services = model_services or DisabledModelServices()
        self.parser = QueryParser()
        self.planner = SearchPlanner(
            per_query_top_k=per_query_top_k,
            academic_search_enabled=academic_search_enabled,
            academic_search_provider=academic_search_provider,
            academic_search_query_limit=academic_search_query_limit,
            academic_search_top_k=academic_search_top_k,
        )
        self.normalizer = CandidateNormalizer()
        self.ranker = CandidateRanker()
        self.preselector = CandidatePreselector()
        self.coverage = CoverageAnalyzer()
        self.citation_expansion = CitationExpansionPlanner()

    def search(self, query: str, *, top_k: int = 20) -> SearchResponse:
        started = time.perf_counter()
        model_events: dict[str, Any] = {"enabled": self.model_services.enabled_names(), "errors": []}
        parsed = self.parser.parse(query)
        parsed = self._apply_query_intent(query, parsed, model_events)
        parsed = self._apply_query_rewrite(query, parsed, model_events)
        query_profile = query_profile_kind(parsed)
        if not parsed.sub_queries:
            elapsed = time.perf_counter() - started
            return SearchResponse(
                query=query,
                parsed_query=parsed,
                plan=SearchPlan(
                    round=1,
                    search_actions=[],
                    expand_citations_for=[],
                    budget={
                        "max_search_rounds": self.planner.max_search_rounds,
                        "max_api_calls": self.planner.max_api_calls,
                        "max_llm_calls": self.planner.max_llm_calls,
                        "max_candidates_for_selector": 0,
                        "max_candidates_for_llm_judge": 0,
                    },
                ),
                papers=[],
                coverage=CoverageReport(
                    coverage={},
                    matched_constraints=[],
                    missing_constraints=[],
                    next_queries=[],
                    should_continue=False,
                    reason="query intent model classified the request as non-paper-search",
                ),
                cost={
                    "latency_sec": round(elapsed, 4),
                    "rounds": 0,
                    "actions_executed": 0,
                    "raw_candidates": 0,
                    "unique_candidates": 0,
                    "api_calls": 0,
                    "llm_calls": 0,
                    "citation_expansion_seeds": [],
                    "query_type": query_profile,
                    "rewrite_used": False,
                    "dense_used": False,
                    "alias_used": False,
                    "model_services": model_events,
                    **self.corpus.stats(),
                },
            )
        first_plan = self.planner.plan(parsed, round_number=1)

        all_candidates: list[Candidate] = []
        actions_executed = 0
        first_round_candidates = self._run_actions(first_plan.search_actions)
        all_candidates.extend(first_round_candidates)
        actions_executed += len(first_plan.search_actions)

        ranked = self._rank(all_candidates, parsed, query=query, top_k=top_k, model_events=model_events)
        coverage = self.coverage.analyze(parsed, ranked)
        rounds = 1

        graph_candidates = self._maybe_expand_with_graph(
            query,
            parsed,
            ranked,
            coverage,
            model_events,
        )
        if graph_candidates:
            all_candidates.extend(graph_candidates)
            actions_executed += 1
            ranked = self._rank(all_candidates, parsed, query=query, top_k=top_k, model_events=model_events)
            coverage = self.coverage.analyze(parsed, ranked)
            rounds = max(rounds, 2)

        if coverage.should_continue and first_plan.budget["max_search_rounds"] > 1:
            retrieval_budget = first_plan.budget.get("retrieval_budget") or profile_retrieval_budget(
                query_profile,
                self.planner.per_query_top_k,
            )
            second_actions = [
                SearchAction("local_title_bm25", next_query, self.planner.per_query_top_k, 1.05)
                for next_query in coverage.next_queries
            ]
            second_actions.extend(
                SearchAction("local_chunk_bm25", next_query, self.planner.per_query_top_k, 1.0)
                for next_query in coverage.next_queries
            )
            second_actions.extend(
                SearchAction("local_tfidf", next_query, self.planner.per_query_top_k, 1.0)
                for next_query in coverage.next_queries
            )
            second_actions.extend(
                SearchAction(
                    "qdrant_dense_paper",
                    next_query,
                    int(retrieval_budget.get("second_round_dense_top_k") or _second_round_dense_top_k(query_profile)),
                    1.08,
                )
                for next_query in coverage.next_queries[: int(retrieval_budget.get("second_round_dense_queries") or 3)]
            )
            all_candidates.extend(self._run_actions(second_actions))
            actions_executed += len(second_actions)
            ranked = self._rank(all_candidates, parsed, query=query, top_k=top_k, model_events=model_events)
            coverage = self.coverage.analyze(parsed, ranked)
            rounds = 2

        self._apply_crawler_strategy(query, ranked, model_events)
        elapsed = time.perf_counter() - started
        citation_seeds = self.citation_expansion.select_seeds(parsed, ranked)
        final_plan = replace(first_plan, expand_citations_for=[seed.paper_id for seed in citation_seeds])
        diagnostic_pool = _diagnostic_pool_snapshot(self.normalizer.merge(all_candidates), limit=max(500, top_k))
        cost: dict[str, Any] = {
            "latency_sec": round(elapsed, 4),
            "rounds": rounds,
            "actions_executed": actions_executed,
            "raw_candidates": len(all_candidates),
            "unique_candidates": len(self.normalizer.merge(all_candidates)),
            "diagnostic_pool_candidates": diagnostic_pool,
            "api_calls": 0,
            "llm_calls": 0,
            "citation_expansion_seeds": [asdict(seed) for seed in citation_seeds],
            "query_type": query_profile,
            "rewrite_used": bool((model_events.get("query_rewrite") or {}).get("rewrites")),
            "dense_used": _source_used(all_candidates, "qdrant_dense_paper"),
            "sparse_paper_used": _source_used(all_candidates, "qdrant_sparse_paper"),
            "alias_used": _source_used(all_candidates, "neo4j_alias"),
            "model_services": model_events,
            **self.corpus.stats(),
        }
        if self.backend_error:
            cost["backend_fallback_reason"] = self.backend_error
        return SearchResponse(
            query=query,
            parsed_query=parsed,
            plan=final_plan,
            papers=ranked,
            coverage=coverage,
            cost=cost,
        )

    def _maybe_expand_with_graph(
        self,
        query: str,
        parsed: QueryIntent,
        ranked: list[Candidate],
        coverage: CoverageReport,
        model_events: dict[str, Any],
    ) -> list[Candidate]:
        expand_fn = getattr(self.corpus, "expand_graph_candidates", None)
        if not callable(expand_fn):
            return []
        if not _should_expand_with_graph(query, parsed, coverage, ranked):
            return []
        seed_candidates = _graph_seed_candidates(ranked, coverage)
        if not seed_candidates:
            return []
        graph_candidates = expand_fn(
            seed_candidates,
            max_neighbors=_graph_neighbor_budget(query_profile_kind(parsed)),
            min_concept_confidence=_graph_min_confidence(query_profile_kind(parsed)),
        )
        if not graph_candidates:
            return []
        model_events["graph_expansion"] = {
            "enabled": True,
            "seed_ids": [candidate.canonical_id or candidate.paper_id for candidate in seed_candidates],
            "expanded_candidates": len(graph_candidates),
            "neighbor_ids": [candidate.paper_id for candidate in graph_candidates[:10]],
        }
        return graph_candidates

    def _run_actions(self, actions: list[SearchAction]) -> list[Candidate]:
        candidates: list[Candidate] = []
        for action in actions:
            action_candidates = self.corpus.run_action(action)
            _annotate_source_ranks(action_candidates, action)
            candidates.extend(action_candidates)
        return candidates

    def _rank(
        self,
        candidates: list[Candidate],
        parsed: QueryIntent,
        *,
        query: str,
        top_k: int,
        model_events: dict[str, Any],
    ) -> list[Candidate]:
        unique_candidates = self.normalizer.merge(candidates)
        pre_rank_k = top_k
        if self.model_services.selector_reranker:
            selector_pool_limit = _selector_pool_limit(self.model_services)
            selector_candidate_limit = _selector_candidate_limit(self.model_services)
            pre_rank_k = min(
                len(unique_candidates),
                max(top_k, selector_pool_limit, selector_candidate_limit),
            )
        pre_rank_k = min(len(unique_candidates), max(pre_rank_k, min(len(unique_candidates), 120)))
        ranked = self.ranker.rank(unique_candidates, parsed, top_k=pre_rank_k)
        if self.model_services.selector_reranker and ranked:
            try:
                selector_pool_limit = _selector_pool_limit(self.model_services)
                selector_candidate_limit = _selector_candidate_limit(self.model_services)
                preselection = self.preselector.select(
                    ranked,
                    parsed,
                    top_k=selector_candidate_limit,
                    pool_limit=selector_pool_limit,
                )
                model_events.setdefault("selector_preselector", []).append(preselection.metadata)
                reranked, metadata = self.model_services.selector_reranker.rerank(
                    query,
                    preselection.candidates,
                    top_k=len(preselection.candidates),
                )
                model_events.setdefault("selector_reranker", []).append(
                    {
                        "candidates": len(preselection.candidates),
                        "candidate_pool": min(len(ranked), selector_pool_limit),
                        **metadata,
                    }
                )
                ranked = _merge_reranked_head(
                    reranked,
                    ranked,
                    protected_head=_selector_protected_head(self.model_services, top_k),
                )
            except ModelServiceError as exc:
                model_events["errors"].append({"service": "selector_reranker", "error": str(exc)})
        ranked = _diversify_ranked(ranked, parsed, top_k=top_k)
        ranked = _source_rank_backfill(ranked, parsed, top_k=top_k)
        return ranked[:top_k]

    def _apply_query_intent(
        self,
        query: str,
        parsed: QueryIntent,
        model_events: dict[str, Any],
    ) -> QueryIntent:
        if not self.model_services.query_intent:
            return parsed
        try:
            prediction = self.model_services.query_intent.predict_one(query)
        except ModelServiceError as exc:
            model_events["errors"].append({"service": "query_intent", "error": str(exc)})
            return parsed
        model_events["query_intent"] = {
            "gate_label": prediction.gate_label,
            "gate_score": prediction.gate_score,
            "intent_label": prediction.intent_label,
            "intent_score": prediction.intent_score,
        }
        adjusted = _intent_adjusted_query(query, parsed, prediction)
        if not prediction.is_paper_search and adjusted.sub_queries:
            model_events["query_intent"]["override"] = "paper_like_rule_fallback"
        return adjusted

    def _apply_query_rewrite(
        self,
        query: str,
        parsed: QueryIntent,
        model_events: dict[str, Any],
    ) -> QueryIntent:
        query_rewriter = getattr(self.model_services, "query_rewriter", None)
        if not query_rewriter or not parsed.sub_queries:
            return parsed
        if not _should_use_query_rewrite(query, parsed):
            model_events["query_rewrite"] = {"skipped": "low_risk_query"}
            return parsed
        try:
            prediction = query_rewriter.rewrite(query, context=_rewrite_context(parsed))
        except ModelServiceError as exc:
            model_events["errors"].append({"service": "query_rewrite", "error": str(exc)})
            return parsed
        additions = _rewrite_sub_queries(prediction)
        if not additions:
            model_events["query_rewrite"] = {
                "cache_hit": prediction.cache_hit,
                "rewrites": [],
                "concepts": prediction.concepts[:8],
                "possible_answer_terms": prediction.possible_answer_terms[:8],
            }
            return parsed
        rewritten = replace(
            parsed,
            sub_queries=_unique([*additions, *parsed.sub_queries])[:12],
            research_field=_unique([*parsed.research_field, *prediction.concepts[:5]])[:8],
            soft_constraints=_unique(
                [
                    *prediction.possible_answer_terms[:6],
                    *prediction.concepts[:6],
                    *parsed.soft_constraints,
                ]
            )[:20],
        )
        model_events["query_rewrite"] = {
            "cache_hit": prediction.cache_hit,
            "rewrites": additions[:8],
            "concepts": prediction.concepts[:8],
            "possible_answer_terms": prediction.possible_answer_terms[:8],
        }
        return rewritten

    def _apply_crawler_strategy(
        self,
        query: str,
        ranked: list[Candidate],
        model_events: dict[str, Any],
    ) -> None:
        if not self.model_services.crawler_strategy or not ranked:
            return
        inspected = 0
        for candidate in ranked[: self.model_services.crawler_top_n]:
            sections = _candidate_sections(candidate)
            try:
                prediction = self.model_services.crawler_strategy.predict(query, candidate, sections=sections)
            except ModelServiceError as exc:
                model_events["errors"].append(
                    {"service": "crawler_strategy", "paper_id": candidate.paper_id, "error": str(exc)}
                )
                continue
            candidate.metadata["crawler_strategy"] = {
                "prediction": prediction.prediction,
                "parse_success": prediction.parse_success,
                "sections": prediction.sections,
                "input_sections": sections,
            }
            inspected += 1
        if inspected:
            model_events["crawler_strategy"] = {"papers_inspected": inspected}


def _intent_adjusted_query(query: str, parsed: QueryIntent, prediction: QueryIntentPrediction) -> QueryIntent:
    if not prediction.is_paper_search:
        if _looks_like_scholarly_search(query, parsed):
            return replace(
                parsed,
                main_intent=f"paper search fallback: {parsed.main_intent}",
            )
        return replace(
            parsed,
            main_intent="non-paper-search request",
            research_field=[],
            must_have_constraints=[],
            soft_constraints=[],
            sub_queries=[],
        )
    if not prediction.intent_label:
        return parsed
    intent_hint = prediction.intent_label.replace("_", " ")
    return replace(
        parsed,
        main_intent=f"{intent_hint}: {parsed.main_intent}",
        research_field=_unique([intent_hint, *parsed.research_field]),
        soft_constraints=_unique([intent_hint, *parsed.soft_constraints]),
    )


def _rewrite_context(parsed: QueryIntent) -> dict[str, Any]:
    return {
        "main_intent": parsed.main_intent,
        "research_field": parsed.research_field[:6],
        "must_have_constraints": parsed.must_have_constraints[:8],
        "soft_constraints": parsed.soft_constraints[:10],
        "time_range": parsed.time_range,
        "venues": parsed.venues,
        "sub_queries": parsed.sub_queries[:6],
    }


def _rewrite_sub_queries(prediction: QueryRewritePrediction) -> list[str]:
    queries: list[str] = []
    queries.extend(prediction.rewrites)
    if prediction.concepts:
        queries.append(" ".join(prediction.concepts[:8]))
    if prediction.possible_answer_terms:
        queries.append(" ".join(prediction.possible_answer_terms[:8]))
    return _unique([item for item in queries if len(tokenize(item)) >= 2])


def _should_use_query_rewrite(query: str, parsed: QueryIntent) -> bool:
    lowered = query.lower()
    intent_text = " ".join(
        [
            parsed.main_intent,
            *parsed.research_field,
            *parsed.must_have_constraints,
            *parsed.soft_constraints,
            *parsed.sub_queries[:3],
        ]
    ).lower()
    markers = (
        "which paper",
        "which work",
        "what work",
        "first proposed",
        "introduced",
        "known as",
        "called",
        "gave rise",
        "origin",
        "foundational",
        "llm-generated",
        "machine-generated text",
        "factuality",
        "hallucination",
        "prompt-based",
        "zero-shot",
        "better than",
        "negative impact",
        "mathematical",
        "theorem proving",
        "object navigation",
        "ray-based",
        "novel view synthesis",
    )
    if any(marker in lowered or marker in intent_text for marker in markers):
        return True
    if len(parsed.must_have_constraints) >= 4 and len(tokenize(query)) >= 10:
        return True
    return False


def _looks_like_scholarly_search(query: str, parsed: QueryIntent) -> bool:
    lowered = query.lower()
    if re.search(r"\b(write|implement|debug|fix|run|execute)\b.+\b(code|python|function|script|program)\b", lowered):
        return False
    if re.search(r"\b(email|poem|story|translate|summarize this|rewrite this)\b", lowered):
        return False
    explicit_search = re.search(
        r"\b(paper|papers|study|studies|work|works|research|publication|survey|benchmark|dataset|datasets)\b",
        lowered,
    )
    scholarly_terms = re.search(
        r"\b(llm|llms|model|models|neural|network|networks|dataset|benchmark|agent|agents|"
        r"multimodal|video|quantum|monte carlo|sft|synthetic|synthesis|financial|"
        r"humaneval|mbpp|code_contests|reinforcement|diffusion|language)\b",
        lowered,
    )
    return bool(parsed.sub_queries and parsed.must_have_constraints and (explicit_search or scholarly_terms))


def _diversify_ranked(candidates: list[Candidate], intent: QueryIntent, *, top_k: int) -> list[Candidate]:
    if len(candidates) <= 12 or top_k <= 10:
        return candidates
    profile = query_profile_kind(intent)
    budget = profile_retrieval_budget(profile, top_k)
    protected = min(len(candidates), int(budget.get("diversity_protected") or (8 if _locator_intent(intent) else 4)))
    selected = candidates[:protected]
    pool = candidates[protected:]
    selected_token_sets = [_title_token_set(candidate) for candidate in selected]
    covered_constraints = {constraint for candidate in selected for constraint in candidate.matched_constraints}
    while pool and len(selected) < top_k:
        best_index = max(
            range(len(pool)),
            key=lambda index: _diversity_score(
                pool[index],
                selected_token_sets,
                covered_constraints,
                profile=profile,
            ),
        )
        candidate = pool.pop(best_index)
        selected.append(candidate)
        selected_token_sets.append(_title_token_set(candidate))
        covered_constraints.update(candidate.matched_constraints)
    return selected + pool


def _merge_reranked_head(
    reranked: list[Candidate],
    original_pool: list[Candidate],
    *,
    protected_head: int = 0,
) -> list[Candidate]:
    merged: list[Candidate] = []
    selected_ids: set[str] = set()

    def add(candidate: Candidate) -> bool:
        paper_id = candidate.canonical_id or candidate.paper_id
        if not paper_id or paper_id in selected_ids:
            return False
        selected_ids.add(paper_id)
        merged.append(candidate)
        return True

    for candidate in original_pool[: max(0, protected_head)]:
        add(candidate)
    for candidate in reranked:
        add(candidate)
    for candidate in original_pool:
        add(candidate)
    return merged


def _source_rank_backfill(candidates: list[Candidate], intent: QueryIntent, *, top_k: int) -> list[Candidate]:
    if len(candidates) <= 12 or top_k <= 10:
        return candidates
    target = min(len(candidates), top_k, 50)
    profile = query_profile_kind(intent)
    anchor_count = _source_backfill_anchor_count(profile, target)
    original_rank = {candidate.canonical_id or candidate.paper_id: index for index, candidate in enumerate(candidates, start=1)}
    front_pool_size = _source_backfill_pool_size(profile, target, len(candidates))
    selected: list[Candidate] = []
    selected_ids: set[str] = set()

    def add(candidate: Candidate, reason: str | None = None) -> bool:
        paper_id = candidate.canonical_id or candidate.paper_id
        if not paper_id or paper_id in selected_ids:
            return False
        selected_ids.add(paper_id)
        if reason:
            candidate.metadata["source_rank_backfill"] = reason
        selected.append(candidate)
        return True

    for candidate in candidates[:anchor_count]:
        add(candidate)

    scored_pool: list[tuple[float, float, int, str, Candidate]] = []
    for index, candidate in enumerate(candidates[anchor_count:front_pool_size], start=anchor_count + 1):
        bonus, reason = _source_rank_backfill_bonus(candidate, profile)
        adjusted_score = candidate.final_score + bonus
        scored_pool.append((adjusted_score, candidate.final_score, -index, reason, candidate))
    scored_pool.sort(reverse=True)
    for adjusted_score, _base_score, _negative_index, reason, candidate in scored_pool:
        if len(selected) >= target:
            break
        paper_id = candidate.canonical_id or candidate.paper_id
        if add(candidate):
            original_position = original_rank.get(paper_id, 0)
            if reason and original_position > target and adjusted_score > candidate.final_score:
                candidate.metadata["source_rank_backfill"] = reason

    for candidate in candidates:
        if len(selected) >= target:
            break
        add(candidate)

    if len(selected) < target:
        for candidate in candidates:
            add(candidate)
    return selected + [candidate for candidate in candidates if (candidate.canonical_id or candidate.paper_id) not in selected_ids]


def _source_rank_backfill_bonus(candidate: Candidate, profile: str) -> tuple[float, str]:
    ranks = candidate.metadata.get("source_ranks") or {}
    if not isinstance(ranks, dict) or not ranks:
        return 0.0, ""
    support_sources = [
        source
        for source in ("local_title_bm25", "local_chunk_bm25", "qdrant_sparse_paper", "local_tfidf", "neo4j_concept")
        if source in ranks and max(1, _safe_int(ranks.get(source))) <= 100
    ]
    support = len(support_sources)
    lexical_support = any(source in support_sources for source in ("local_title_bm25", "local_chunk_bm25", "local_tfidf"))
    semantic_bonus = max(
        _safe_float(candidate.metadata.get("soft_alias_bonus")),
        _safe_float(candidate.metadata.get("strong_alias_bonus")),
        _safe_float(candidate.metadata.get("graph_alias_bonus")),
    )
    semantic_support = semantic_bonus >= 0.45 or bool(candidate.matched_constraints)
    bonus = 0.0
    reasons: list[str] = []

    title_rank = max(1, _safe_int(ranks.get("local_title_bm25"))) if "local_title_bm25" in ranks else None
    chunk_rank = max(1, _safe_int(ranks.get("local_chunk_bm25"))) if "local_chunk_bm25" in ranks else None
    sparse_rank = max(1, _safe_int(ranks.get("qdrant_sparse_paper"))) if "qdrant_sparse_paper" in ranks else None
    tfidf_rank = max(1, _safe_int(ranks.get("local_tfidf"))) if "local_tfidf" in ranks else None
    graph_rank = max(1, _safe_int(ranks.get("neo4j_concept"))) if "neo4j_concept" in ranks else None

    if title_rank is not None and title_rank <= 100:
        value = 0.2 / (1.0 + title_rank / 28.0)
        if profile in {"real_multi_answer", "survey_or_list"} and title_rank <= 35 and semantic_support:
            value += 0.08 / (1.0 + title_rank / 35.0)
        bonus += value
        reasons.append(f"local_title_bm25:{title_rank}")
    if chunk_rank is not None and chunk_rank <= 100:
        if support >= 2 or candidate.final_score >= 0.9 or chunk_rank <= 12:
            value = 0.14 / (1.0 + chunk_rank / 26.0)
            bonus += value
            reasons.append(f"local_chunk_bm25:{chunk_rank}")
    if sparse_rank is not None and sparse_rank <= 90:
        if lexical_support:
            value = 0.16 / (1.0 + sparse_rank / 34.0)
            if profile in {"real_multi_answer", "survey_or_list"} and semantic_support and sparse_rank <= 70:
                value += 0.06 / (1.0 + sparse_rank / 40.0)
            bonus += value
            reasons.append(f"qdrant_sparse_paper:{sparse_rank}")
        elif sparse_rank <= 12:
            value = 0.04 / (1.0 + sparse_rank / 20.0)
            if semantic_support and profile in {"real_multi_answer", "survey_or_list", "dataset_or_benchmark"}:
                value += 0.1 / (1.0 + sparse_rank / 12.0)
            bonus += value
            reasons.append(f"qdrant_sparse_paper:{sparse_rank}")
    if tfidf_rank is not None and tfidf_rank <= 60 and support >= 2:
        value = 0.06 / (1.0 + tfidf_rank / 24.0)
        bonus += value
        reasons.append(f"local_tfidf:{tfidf_rank}")
    if graph_rank is not None and graph_rank <= 45 and support >= 2:
        value = 0.05 / (1.0 + graph_rank / 20.0)
        bonus += value
        reasons.append(f"neo4j_concept:{graph_rank}")
    if support >= 3:
        bonus += 0.08
        reasons.append("support>=3")
    elif support >= 2:
        bonus += 0.04
        reasons.append("support>=2")
    if semantic_support and support >= 2 and profile in {"real_multi_answer", "survey_or_list"}:
        bonus += 0.06
        reasons.append("semantic_support")

    if support <= 1:
        single_cap = 0.12
        if title_rank is not None and title_rank <= 15:
            single_cap = 0.16
        if semantic_support and title_rank is not None and title_rank <= 40:
            single_cap = 0.24
        if semantic_support and sparse_rank is not None and sparse_rank <= 8:
            single_cap = max(single_cap, 0.22)
        bonus = min(bonus, single_cap)
    if candidate.relevance == "weakly_relevant":
        bonus *= 0.55
    if profile in {"real_multi_answer", "survey_or_list"} and support <= 1 and not semantic_support:
        bonus *= 0.7
    cap = 0.5 if profile in {"real_multi_answer", "survey_or_list"} and semantic_support else 0.36
    return min(cap, bonus), ",".join(reasons[:5])


def _source_backfill_anchor_count(profile: str, target: int) -> int:
    if profile in {"real_multi_answer", "survey_or_list"}:
        return min(6, max(3, target // 8))
    if profile in {"auto_locator", "foundational_or_origin"}:
        return min(8, max(4, target // 7))
    return min(7, max(4, target // 7))


def _source_backfill_pool_size(profile: str, target: int, total: int) -> int:
    base = 140
    if profile in {"real_multi_answer", "survey_or_list"}:
        base = 500
    elif profile in {"dataset_or_benchmark", "method_or_dataset", "comparison_or_claim"}:
        base = 220
    elif profile in {"auto_locator", "foundational_or_origin"}:
        base = 180
    return min(total, max(target, base))


def _diversity_score(
    candidate: Candidate,
    selected_token_sets: list[set[str]],
    covered_constraints: set[str],
    *,
    profile: str,
) -> float:
    title_tokens = _title_token_set(candidate)
    redundancy = max((_jaccard(title_tokens, tokens) for tokens in selected_token_sets), default=0.0)
    alias_bonus = max(
        _safe_float(candidate.metadata.get("soft_alias_bonus")),
        _safe_float(candidate.metadata.get("strong_alias_bonus")),
    )
    if candidate.relevance == "highly_relevant":
        redundancy *= 0.55
    if alias_bonus >= 0.8:
        redundancy *= 0.25
    new_constraints = set(candidate.matched_constraints) - covered_constraints
    candidate.metadata["diversity_redundancy"] = round(redundancy, 6)
    candidate.metadata["diversity_new_constraints"] = sorted(new_constraints)
    redundancy_penalty = 0.02 if profile in {"real_multi_answer", "survey_or_list"} else 0.1
    return candidate.final_score + 0.025 * len(new_constraints) - redundancy_penalty * redundancy


def _title_token_set(candidate: Candidate) -> set[str]:
    return set(tokenize(candidate.title))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _selector_pool_limit(model_services: ModelServicesPort) -> int:
    raw_limit = (
        _safe_int(getattr(model_services, "selector_pool_limit", DEFAULT_SELECTOR_POOL_LIMIT))
        or DEFAULT_SELECTOR_POOL_LIMIT
    )
    return max(1, raw_limit)


def _selector_candidate_limit(model_services: ModelServicesPort) -> int:
    raw_limit = (
        _safe_int(getattr(model_services, "selector_candidate_limit", DEFAULT_SELECTOR_CANDIDATE_LIMIT))
        or DEFAULT_SELECTOR_CANDIDATE_LIMIT
    )
    return max(1, raw_limit)


def _selector_protected_head(model_services: ModelServicesPort, top_k: int) -> int:
    raw_limit = _safe_int(getattr(model_services, "selector_protected_head", DEFAULT_SELECTOR_PROTECTED_HEAD))
    return min(max(0, raw_limit), max(0, top_k))


def _locator_intent(intent: QueryIntent) -> bool:
    text = " ".join([intent.main_intent, *intent.soft_constraints, *intent.must_have_constraints]).lower()
    return any(marker in text for marker in ("first proposed", "introduced", "known as", "called"))


def _should_expand_with_graph(
    query: str,
    intent: QueryIntent,
    coverage: CoverageReport,
    ranked: list[Candidate],
) -> bool:
    if not ranked:
        return False
    if _looks_like_locator_query(query) or _locator_intent(intent):
        return False
    if len(tokenize(query)) < 5:
        return False
    if len(intent.sub_queries) < 2 or len(intent.research_field) == 0:
        return False
    if not coverage.should_continue or not coverage.missing_constraints:
        return False
    if not _looks_like_scholarly_search(query, intent):
        return False
    top_slice = ranked[:6]
    highly_relevant = sum(1 for candidate in top_slice if candidate.relevance == "highly_relevant")
    if highly_relevant >= 3:
        return False
    if any(_safe_float(candidate.metadata.get("strong_alias_bonus")) >= 0.8 for candidate in top_slice[:3]):
        return False
    return True


def _looks_like_locator_query(query: str) -> bool:
    lowered = query.lower()
    return bool(
        re.search(
            r"\b(which|what|find)\b.*\b(paper|work|study|method)\b.*\b(first\s+)?(proposed|introduced|implemented)\b",
            lowered,
        )
        or re.search(r"\b(known as|called)\b", lowered)
    )


def _graph_seed_candidates(ranked: list[Candidate], coverage: CoverageReport) -> list[Candidate]:
    seeds: list[Candidate] = []
    for candidate in ranked:
        if candidate.relevance == "weakly_relevant":
            continue
        if candidate.relevance == "highly_relevant" or candidate.matched_constraints or len(candidate.sources) >= 2:
            seeds.append(candidate)
        if len(seeds) >= 3:
            break
    if seeds:
        return seeds[:3]
    if coverage.should_continue:
        fallback = [candidate for candidate in ranked[:2] if candidate.raw_scores]
        return fallback[:1]
    return []


def _candidate_sections(candidate: Candidate) -> list[str]:
    sections: list[str] = []
    metadata_sections = candidate.metadata.get("section_titles") or []
    if isinstance(metadata_sections, list):
        sections.extend(str(item) for item in metadata_sections)
    section_title = candidate.metadata.get("section_title")
    if section_title:
        sections.append(str(section_title))
    for snippet in candidate.snippets:
        sections.extend(match.strip() for match in SECTION_RE.findall(snippet) if match.strip())
    unique = _unique([section for section in sections if section])
    return unique[:12] or ["Title and Abstract"]


def _annotate_source_ranks(candidates: list[Candidate], action: SearchAction) -> None:
    source_key = action.source
    source_weight = _rrf_source_weight(action)
    for rank, candidate in enumerate(candidates, start=1):
        rrf_score = source_weight / (60.0 + rank)
        candidate.raw_scores[f"rrf:{source_key}"] = rrf_score
        ranks = candidate.metadata.setdefault("source_ranks", {})
        if isinstance(ranks, dict):
            ranks[source_key] = min(int(ranks.get(source_key, rank) or rank), rank)
        weights = candidate.metadata.setdefault("source_weights", {})
        if isinstance(weights, dict):
            weights[source_key] = action.weight


def _rrf_source_weight(action: SearchAction) -> float:
    base = {
        "local_title_bm25": 1.25,
        "local_chunk_bm25": 1.0,
        "local_tfidf": 0.94,
        "qdrant_dense_paper": 1.18,
        "qdrant_sparse_paper": 1.08,
        "neo4j_alias": 1.1,
        "neo4j_concept": 0.72,
        "semantic_scholar": 0.86,
    }.get(action.source, 1.0)
    return base * max(0.1, action.weight)


def _source_used(candidates: list[Candidate], source: str) -> bool:
    return any(source in candidate.sources for candidate in candidates)


def _second_round_dense_top_k(profile: str) -> int:
    if profile in {"real_multi_answer", "survey_or_list", "dataset_or_benchmark"}:
        return 80
    if profile in {"auto_locator", "foundational_or_origin"}:
        return 70
    return 60


def _graph_neighbor_budget(profile: str) -> int:
    return int(profile_retrieval_budget(profile, 60).get("graph_expansion_neighbors") or 30)


def _graph_min_confidence(profile: str) -> float:
    if profile in {"foundational_or_origin", "survey_or_list", "real_multi_answer"}:
        return 0.6
    if profile == "dataset_or_benchmark":
        return 0.62
    return 0.65


def _diagnostic_pool_snapshot(candidates: list[Candidate], *, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=_diagnostic_pool_score, reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ordered[:limit], start=1):
        rows.append(
            {
                "rank": rank,
                "paper_id": candidate.canonical_id or candidate.paper_id,
                "sources": sorted(candidate.sources),
                "source_ranks": candidate.metadata.get("source_ranks") or {},
                "rrf_score": round(_rrf_sum(candidate), 8),
                "dense_used": "qdrant_dense_paper" in candidate.sources,
                "sparse_paper_used": "qdrant_sparse_paper" in candidate.sources,
                "alias_used": "neo4j_alias" in candidate.sources,
                "alias_support": candidate.metadata.get("alias_support"),
                "alias_relations": candidate.metadata.get("alias_relations") or [],
                "alias_matched_terms": candidate.metadata.get("alias_matched_terms") or [],
                "graph_alias_bonus": candidate.metadata.get("graph_alias_bonus"),
                "dense_query": candidate.metadata.get("dense_query"),
            }
        )
    return rows


def _diagnostic_pool_score(candidate: Candidate) -> tuple[float, float, str]:
    return (_rrf_sum(candidate), max(candidate.raw_scores.values(), default=0.0), candidate.paper_id)


def _rrf_sum(candidate: Candidate) -> float:
    return sum(float(value) for key, value in candidate.raw_scores.items() if key.startswith("rrf:"))


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
