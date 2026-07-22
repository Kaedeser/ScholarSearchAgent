# 中文功能说明：外部学术搜索 API 接入测试，覆盖 Semantic Scholar 客户端和召回候选转换。

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from packages.scholar_core.models import SearchAction
from packages.scholar_core.planning.planner import SearchPlanner
from packages.scholar_infra import academic_search
from packages.scholar_infra.academic_search import SemanticScholarClient
from packages.scholar_infra.retrieval_backends.retrieval import DatabaseCorpus, LocalCorpus


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_semantic_scholar_client_maps_search_results(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "data": [
                    {
                        "paperId": "abc123",
                        "title": "A Neural Information Retrieval Paper",
                        "abstract": "Dense retrieval and query rewriting.",
                        "year": 2024,
                        "venue": "ACL",
                        "citationCount": 7,
                        "externalIds": {"ArXiv": "2401.00001", "DOI": "10.123/test"},
                        "url": "https://www.semanticscholar.org/paper/abc123",
                    }
                ]
            }
        )

    monkeypatch.setattr(academic_search, "urlopen", fake_urlopen)

    client = SemanticScholarClient(base_url="https://api.test/graph/v1", api_key="secret", timeout=3.0)
    results = client.search("dense retrieval", limit=5)

    assert captured["url"].startswith("https://api.test/graph/v1/paper/search?")
    assert "query=dense+retrieval" in captured["url"]
    assert captured["headers"]["X-api-key"] == "secret"
    assert captured["timeout"] == 3.0
    assert results[0].paper_id == "s2:abc123"
    assert results[0].title == "A Neural Information Retrieval Paper"
    assert results[0].external_ids == {"ArXiv": "2401.00001", "DOI": "10.123/test"}


def test_search_planner_adds_academic_api_actions_when_enabled():
    from packages.scholar_core.query_understanding.parser import QueryParser

    parsed = QueryParser().parse("Find papers about neural information retrieval and query rewriting.")
    plan = SearchPlanner(
        per_query_top_k=30,
        academic_search_enabled=True,
        academic_search_query_limit=1,
        academic_search_top_k=12,
    ).plan(parsed)

    actions = [action for action in plan.search_actions if action.source == "semantic_scholar"]

    assert len(actions) == 1
    assert actions[0].top_k == 12
    assert plan.budget["academic_search"]["enabled"] is True
    assert plan.budget["max_api_calls"] == 1


def test_database_corpus_converts_semantic_scholar_results_to_candidates():
    corpus = DatabaseCorpus.__new__(DatabaseCorpus)
    corpus._semantic_scholar = type(
        "Client",
        (),
        {
            "search": lambda self, query, limit: [
                academic_search.AcademicSearchResult(
                    paper_id="s2:abc123",
                    title="External Retrieval Paper",
                    abstract="A paper about academic search APIs.",
                    year=2025,
                    venue="TestConf",
                    citation_count=11,
                    url="https://www.semanticscholar.org/paper/abc123",
                    external_ids={"ArXiv": "2501.00001", "DOI": "10.123/test"},
                    raw={},
                )
            ]
        },
    )()

    candidates = DatabaseCorpus._search_semantic_scholar(
        corpus,
        SearchAction("semantic_scholar", "academic search APIs", 10, 0.8),
    )

    assert candidates[0].paper_id == "s2:abc123"
    assert {"semantic_scholar", "academic_api"} <= candidates[0].sources
    assert candidates[0].raw_scores["semantic_scholar"] == 0.8
    assert "arxiv:2501.00001" in candidates[0].aliases
    assert "doi:10.123/test" in candidates[0].aliases
    assert candidates[0].metadata["academic_api_provider"] == "semantic_scholar"


def test_local_corpus_routes_semantic_scholar_actions():
    corpus = LocalCorpus.__new__(LocalCorpus)
    corpus._academic_search_error = None
    corpus._semantic_scholar = type(
        "Client",
        (),
        {
            "search": lambda self, query, limit: [
                academic_search.AcademicSearchResult(
                    paper_id="s2:local123",
                    title="External Search with a Local Corpus",
                    abstract="Semantic Scholar results can supplement local JSONL retrieval.",
                    year=2026,
                    venue="TestConf",
                    citation_count=3,
                    url="https://www.semanticscholar.org/paper/local123",
                    external_ids={"DOI": "10.123/local"},
                    raw={},
                )
            ]
        },
    )()

    candidates = corpus.run_action(
        SearchAction("semantic_scholar", "local corpus external search", 5, 0.9),
    )

    assert candidates[0].paper_id == "s2:local123"
    assert candidates[0].sources == {"semantic_scholar", "academic_api"}
    assert candidates[0].raw_scores["semantic_scholar"] == 0.9
    assert "doi:10.123/local" in candidates[0].aliases
