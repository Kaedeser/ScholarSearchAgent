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
)
from packages.scholar_core.query_understanding.parser import QueryParser
from packages.scholar_core.ranking.ranker import CandidateRanker
from packages.scholar_core.retrieval.ports import CorpusBackend
from packages.scholar_core.models import Candidate, CoverageReport, QueryIntent, SearchAction, SearchPlan, SearchResponse
from packages.scholar_core.planning.planner import SearchPlanner


SECTION_RE = re.compile(r"\bSection:\s*([^\n]+)", re.IGNORECASE)


class SearchPipeline:
    def __init__(
        self,
        corpus: CorpusBackend,
        *,
        per_query_top_k: int = 30,
        model_services: ModelServicesPort | None = None,
        backend_error: str | None = None,
    ) -> None:
        self.corpus = corpus
        self.backend_error = backend_error
        self.model_services = model_services or DisabledModelServices()
        self.parser = QueryParser()
        self.planner = SearchPlanner(per_query_top_k=per_query_top_k)
        self.normalizer = CandidateNormalizer()
        self.ranker = CandidateRanker()
        self.coverage = CoverageAnalyzer()
        self.citation_expansion = CitationExpansionPlanner()

    def search(self, query: str, *, top_k: int = 20) -> SearchResponse:
        started = time.perf_counter()
        model_events: dict[str, Any] = {"enabled": self.model_services.enabled_names(), "errors": []}
        parsed = self.parser.parse(query)
        parsed = self._apply_query_intent(query, parsed, model_events)
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

        if coverage.should_continue and first_plan.budget["max_search_rounds"] > 1:
            second_actions = [
                SearchAction("local_title_bm25", next_query, self.planner.per_query_top_k, 1.05)
                for next_query in coverage.next_queries
            ]
            second_actions.extend(
                SearchAction("local_chunk_bm25", next_query, self.planner.per_query_top_k, 1.0)
                for next_query in coverage.next_queries
            )
            second_actions.extend(
                SearchAction("local_tfidf", next_query, max(20, int(self.planner.per_query_top_k * 0.8)), 0.9)
                for next_query in coverage.next_queries
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
        cost: dict[str, Any] = {
            "latency_sec": round(elapsed, 4),
            "rounds": rounds,
            "actions_executed": actions_executed,
            "raw_candidates": len(all_candidates),
            "unique_candidates": len(self.normalizer.merge(all_candidates)),
            "api_calls": 0,
            "llm_calls": 0,
            "citation_expansion_seeds": [asdict(seed) for seed in citation_seeds],
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

    def _run_actions(self, actions: list[SearchAction]) -> list[Candidate]:
        candidates: list[Candidate] = []
        for action in actions:
            candidates.extend(self.corpus.run_action(action))
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
            pre_rank_k = min(
                len(unique_candidates),
                max(top_k, self.model_services.selector_candidate_limit),
            )
        ranked = self.ranker.rank(unique_candidates, parsed, top_k=pre_rank_k)
        if self.model_services.selector_reranker and ranked:
            try:
                reranked, metadata = self.model_services.selector_reranker.rerank(query, ranked, top_k=len(ranked))
                model_events.setdefault("selector_reranker", []).append({"candidates": len(ranked), **metadata})
                ranked = reranked
            except ModelServiceError as exc:
                model_events["errors"].append({"service": "selector_reranker", "error": str(exc)})
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
        return _intent_adjusted_query(parsed, prediction)

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


def _intent_adjusted_query(parsed: QueryIntent, prediction: QueryIntentPrediction) -> QueryIntent:
    if not prediction.is_paper_search:
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
