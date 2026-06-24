from __future__ import annotations

from dataclasses import asdict, replace
import time
from pathlib import Path
from typing import Any

from candidate_normalization.normalizer import CandidateNormalizer
from citation_network_expansion.citation import CitationExpansionPlanner
from coverage_iteration.coverage import CoverageAnalyzer
from multi_source_retrieval.retrieval import DatabaseCorpus, LocalCorpus
from query_understanding_decomposition.query import QueryParser
from relevance_ranking.ranking import CandidateRanker
from scholar_common.models import Candidate, SearchAction, SearchPlan, SearchResponse
from search_strategy_planning.planner import SearchPlanner


class SearchPipeline:
    def __init__(
        self,
        processed_dir: Path,
        *,
        paper_limit: int | None = None,
        chunk_limit: int | None = None,
        max_chunks_per_paper: int = 4,
        per_query_top_k: int = 30,
        backend: str = "auto",
    ) -> None:
        self.processed_dir = processed_dir
        self.backend = backend
        self.backend_error: str | None = None
        self.parser = QueryParser()
        self.planner = SearchPlanner(per_query_top_k=per_query_top_k)
        self.corpus = self._build_corpus(
            processed_dir,
            backend=backend,
            paper_limit=paper_limit,
            chunk_limit=chunk_limit,
            max_chunks_per_paper=max_chunks_per_paper,
        )
        self.normalizer = CandidateNormalizer()
        self.ranker = CandidateRanker()
        self.coverage = CoverageAnalyzer()
        self.citation_expansion = CitationExpansionPlanner()

    def _build_corpus(
        self,
        processed_dir: Path,
        *,
        backend: str,
        paper_limit: int | None,
        chunk_limit: int | None,
        max_chunks_per_paper: int,
    ):
        if backend not in {"auto", "jsonl", "database"}:
            raise ValueError(f"Unsupported backend: {backend}")
        if backend in {"auto", "database"}:
            try:
                return DatabaseCorpus()
            except Exception as exc:
                if backend == "database":
                    raise
                self.backend_error = str(exc)
        return LocalCorpus(
            processed_dir,
            paper_limit=paper_limit,
            chunk_limit=chunk_limit,
            max_chunks_per_paper=max_chunks_per_paper,
        )

    def search(self, query: str, *, top_k: int = 20) -> SearchResponse:
        started = time.perf_counter()
        parsed = self.parser.parse(query)
        first_plan = self.planner.plan(parsed, round_number=1)

        all_candidates: list[Candidate] = []
        actions_executed = 0
        first_round_candidates = self._run_actions(first_plan.search_actions)
        all_candidates.extend(first_round_candidates)
        actions_executed += len(first_plan.search_actions)

        ranked = self._rank(all_candidates, parsed, top_k=top_k)
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
            all_candidates.extend(self._run_actions(second_actions))
            actions_executed += len(second_actions)
            ranked = self._rank(all_candidates, parsed, top_k=top_k)
            coverage = self.coverage.analyze(parsed, ranked)
            rounds = 2

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

    def _rank(self, candidates: list[Candidate], parsed, *, top_k: int) -> list[Candidate]:
        unique_candidates = self.normalizer.merge(candidates)
        return self.ranker.rank(unique_candidates, parsed, top_k=top_k)
