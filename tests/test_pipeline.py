# 中文功能说明：主检索流水线测试，覆盖查询解析、候选归一、评测指标和模型服务接入。

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.scholar_api.api.routes.health import health_response
from apps.backend.scholar_api.api.schemas.search import parse_search_query
from packages.scholar_core.models import Candidate
from packages.scholar_core.model_services.ports import CrawlerStrategyPrediction, QueryIntentPrediction
from packages.scholar_core.normalization.normalizer import CandidateNormalizer
from packages.scholar_core.pipeline import SearchPipeline
from packages.scholar_core.planning.planner import SearchPlanner
from packages.scholar_core.query_understanding.parser import QueryParser
from packages.scholar_eval.evaluation import score_prediction
from packages.scholar_infra.retrieval_backends.retrieval import LocalCorpus


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def make_processed_dir(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    write_jsonl(
        processed / "papers.jsonl",
        [
            {
                "paper_id": "arxiv:1111.00001",
                "title": "Universal Image Text Representation Learning for Image Retrieval",
                "abstract": "A model for cross-modal image retrieval and visual search.",
                "year": 2020,
                "source": "pasa",
            },
            {
                "paper_id": "arxiv:2222.00002",
                "title": "A Survey of Database Query Optimizers",
                "abstract": "This survey discusses relational query processing.",
                "year": 2018,
                "source": "pasa",
            },
        ],
    )
    write_jsonl(
        processed / "paper_chunks.jsonl",
        [
            {
                "chunk_id": "arxiv:1111.00001#chunk:0",
                "paper_id": "arxiv:1111.00001",
                "text": "Title: Universal Image Text Representation Learning for Image Retrieval\nAbstract: cross-modal retrieval.",
                "section_title": None,
            },
            {
                "chunk_id": "arxiv:1111.00001#chunk:1",
                "paper_id": "arxiv:1111.00001",
                "text": "Title: Universal Image Text Representation Learning for Image Retrieval\nSection: Method\nReferenced papers: visual search.",
                "section_title": "Method",
            },
            {
                "chunk_id": "arxiv:2222.00002#chunk:0",
                "paper_id": "arxiv:2222.00002",
                "text": "Title: A Survey of Database Query Optimizers\nAbstract: relational query plans.",
                "section_title": None,
            },
        ],
    )
    write_jsonl(
        processed / "queries.jsonl",
        [
            {
                "qid": "q1",
                "query_text": "What works are related to the field of image retrieval?",
                "split_name": "dev",
                "dataset_name": "mini",
            }
        ],
    )
    write_jsonl(
        processed / "eval_sets.jsonl",
        [
            {
                "qid": "q1",
                "gold_paper_ids": ["arxiv:1111.00001"],
                "split_name": "dev",
                "dataset_name": "mini",
            }
        ],
    )
    return processed


def build_pipeline(processed: Path, *, model_services=None) -> SearchPipeline:
    corpus = LocalCorpus(processed, max_chunks_per_paper=4)
    return SearchPipeline(corpus, per_query_top_k=5, model_services=model_services)


def test_query_parser_generates_constraints_and_subqueries():
    parsed = QueryParser().parse("Find image retrieval papers after 2020")
    assert "image retrieval" in parsed.must_have_constraints
    assert parsed.time_range == (2020, None)
    assert parsed.sub_queries


def test_query_parser_expands_failure_case_terms():
    parsed = QueryParser().parse("What work proposes to model speech based on HuBERT codes or semantic tokens?")

    assert "hubert codes" in parsed.must_have_constraints
    assert "semantic tokens" in parsed.must_have_constraints
    assert any("discrete speech units" in query or "speech tokens" in query for query in parsed.sub_queries)


def test_search_planner_uses_bounded_multi_source_actions():
    parsed = QueryParser().parse("Which work introduced mask classification-based methods for instance-level segmentation?")
    plan = SearchPlanner(per_query_top_k=60).plan(parsed)

    sources = {action.source for action in plan.search_actions}

    assert {"local_title_bm25", "local_chunk_bm25", "local_tfidf"} <= sources
    assert any(action.top_k < 60 for action in plan.search_actions)
    assert plan.budget["max_candidates_for_selector"] == 400


def test_candidate_normalizer_merges_arxiv_aliases():
    candidates = [
        Candidate("arxiv:2301.00001", "Paper A", sources={"local_title_bm25"}, raw_scores={"a": 1.0}),
        Candidate("arXiv:2301.00001", "Paper A", sources={"local_tfidf"}, raw_scores={"b": 2.0}),
    ]
    merged = CandidateNormalizer().merge(candidates)
    assert len(merged) == 1
    assert merged[0].sources == {"local_title_bm25", "local_tfidf"}
    assert merged[0].raw_scores["b"] == 2.0


def test_score_prediction_computes_basic_metrics():
    metrics = score_prediction(["a", "b", "c"], ["b", "d"], k=3)
    assert metrics.hits == 1
    assert round(metrics.recall_at_k, 3) == 0.5
    assert round(metrics.mrr, 3) == 0.5


def test_ranker_matches_constraints_by_token_not_substring():
    pipeline_query = QueryParser().parse("Find papers about semantic tokens")
    candidate = Candidate(
        "arxiv:3333.00003",
        "Conditional Generative Fields",
        abstract="A conditional model for rendering.",
        sources={"local_title_bm25"},
        raw_scores={"local_title_bm25": 1.0},
    )
    from packages.scholar_core.ranking.ranker import CandidateRanker

    ranked = CandidateRanker().rank([candidate], pipeline_query, top_k=1)

    assert "semantic" in ranked[0].missing_constraints
    assert "tokens" in ranked[0].missing_constraints


def test_ranker_prefers_exact_constraint_coverage():
    parsed = QueryParser().parse("Find mask classification methods for instance-level segmentation")
    candidates = [
        Candidate(
            "arxiv:partial",
            "Per-Pixel Classification is Not All You Need for Semantic Segmentation",
            abstract="A segmentation method with classification labels.",
            sources={"local_title_bm25"},
            raw_scores={"local_title_bm25": 4.0},
        ),
        Candidate(
            "arxiv:exact",
            "Mask Classification for Instance-Level Segmentation",
            abstract="A mask classification method for instance-level segmentation.",
            sources={"local_title_bm25"},
            raw_scores={"local_title_bm25": 2.0},
        ),
    ]
    from packages.scholar_core.ranking.ranker import CandidateRanker

    ranked = CandidateRanker().rank(candidates, parsed, top_k=2)

    assert ranked[0].paper_id == "arxiv:exact"


def test_pipeline_returns_ranked_result(tmp_path):
    processed = make_processed_dir(tmp_path)
    pipeline = build_pipeline(processed)
    response = pipeline.search("image retrieval representation learning", top_k=2)
    assert response.papers
    assert response.papers[0].paper_id == "arxiv:1111.00001"
    assert response.plan.expand_citations_for == ["arxiv:1111.00001"]
    assert response.cost["citation_expansion_seeds"][0]["paper_id"] == "arxiv:1111.00001"
    assert response.coverage.coverage


class RecordingCorpus:
    backend_name = "recording"

    def __init__(self) -> None:
        self.actions = []

    def run_action(self, action):
        self.actions.append(action)
        return [
            Candidate(
                "arxiv:unrelated",
                "Database Query Optimizers",
                abstract="Relational query planning and database execution.",
                sources={action.source},
                raw_scores={action.source: 1.0},
            )
        ]

    def stats(self) -> dict:
        return {"backend": self.backend_name}


def test_pipeline_second_round_includes_sparse_retrieval():
    corpus = RecordingCorpus()
    pipeline = SearchPipeline(corpus, per_query_top_k=5)

    response = pipeline.search("semantic tokens for speech generation", top_k=1)

    first_round_count = len(response.plan.search_actions)
    second_round_sources = {action.source for action in corpus.actions[first_round_count:]}
    assert response.cost["rounds"] == 2
    assert "local_tfidf" in second_round_sources


class FakeQueryIntentService:
    def predict_one(self, text: str) -> QueryIntentPrediction:
        return QueryIntentPrediction(
            gate_label="paper_search",
            gate_score=0.99,
            intent_label="method_search",
            intent_score=0.88,
            raw={},
        )


class FakeSelectorRerankerService:
    def rerank(self, query: str, candidates: list[Candidate], *, top_k: int):
        for candidate in candidates:
            score = 0.99 if candidate.paper_id == "arxiv:1111.00001" else 0.05
            candidate.raw_scores["selector_reranker"] = score
            candidate.final_score = score
        candidates.sort(key=lambda item: item.final_score, reverse=True)
        return candidates[:top_k], {"count": len(candidates), "threshold": 0.5}


class FakeCrawlerStrategyService:
    def predict(self, query: str, candidate: Candidate, *, sections: list[str]) -> CrawlerStrategyPrediction:
        return CrawlerStrategyPrediction(
            prediction="[Expand]Method[StopExpand]",
            parse_success=True,
            sections=["Method"],
            raw={},
        )


class FakeModelServices:
    def __init__(
        self,
        *,
        query_intent=None,
        selector_reranker=None,
        crawler_strategy=None,
        selector_candidate_limit: int = 10,
        crawler_top_n: int = 1,
    ) -> None:
        self.query_intent = query_intent
        self.selector_reranker = selector_reranker
        self.crawler_strategy = crawler_strategy
        self.selector_candidate_limit = selector_candidate_limit
        self.crawler_top_n = crawler_top_n

    def enabled_names(self) -> list[str]:
        names: list[str] = []
        if self.query_intent:
            names.append("query_intent")
        if self.selector_reranker:
            names.append("selector_reranker")
        if self.crawler_strategy:
            names.append("crawler_strategy")
        return names


def test_pipeline_uses_configured_model_services(tmp_path):
    processed = make_processed_dir(tmp_path)
    services = FakeModelServices(
        query_intent=FakeQueryIntentService(),
        selector_reranker=FakeSelectorRerankerService(),
        crawler_strategy=FakeCrawlerStrategyService(),
        selector_candidate_limit=10,
        crawler_top_n=1,
    )
    pipeline = build_pipeline(processed, model_services=services)

    response = pipeline.search("image retrieval representation learning", top_k=2)

    assert response.parsed_query.main_intent.startswith("method search:")
    assert response.papers[0].paper_id == "arxiv:1111.00001"
    assert response.papers[0].raw_scores["selector_reranker"] == 0.99
    assert response.papers[0].metadata["crawler_strategy"]["sections"] == ["Method"]
    assert response.cost["model_services"]["query_intent"]["intent_label"] == "method_search"
    assert response.cost["model_services"]["selector_reranker"]
    assert response.cost["model_services"]["crawler_strategy"]["papers_inspected"] == 1


def test_pipeline_stops_non_paper_queries_from_query_intent(tmp_path):
    class NonPaperQueryIntentService:
        def predict_one(self, text: str) -> QueryIntentPrediction:
            return QueryIntentPrediction("non_paper_search", 0.98, None, None, {})

    processed = make_processed_dir(tmp_path)
    services = FakeModelServices(
        query_intent=NonPaperQueryIntentService(),
        selector_reranker=None,
        crawler_strategy=None,
        selector_candidate_limit=10,
        crawler_top_n=0,
    )
    pipeline = build_pipeline(processed, model_services=services)

    response = pipeline.search("Write a Python function to sort a list.", top_k=2)

    assert response.papers == []
    assert response.cost["actions_executed"] == 0
    assert response.coverage.reason == "query intent model classified the request as non-paper-search"


def test_health_response_contract():
    response = health_response()
    assert response["status"] == "ok"
    assert response["service"] == "scholar-search-api"
    assert "/api/search" in response["endpoints"]


def test_search_query_schema_parses_defaults_and_top_k():
    query, top_k = parse_search_query("q=image%20retrieval&top_k=3")
    assert query == "image retrieval"
    assert top_k == 3
