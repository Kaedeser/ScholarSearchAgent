# 中文功能说明：外部学术搜索 API 接入测试，覆盖 Semantic Scholar 客户端和召回候选转换。

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError

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
                        "corpusId": 987654,
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
    results = client.search("dense retrieval", limit=5, filters={"year": "2020-", "venue": "ACL"})

    assert captured["url"].startswith("https://api.test/graph/v1/paper/search?")
    assert "query=dense+retrieval" in captured["url"]
    assert "year=2020-" in captured["url"]
    assert "venue=ACL" in captured["url"]
    assert captured["headers"]["X-api-key"] == "secret"
    assert captured["timeout"] == 3.0
    assert results[0].paper_id == "s2:abc123"
    assert results[0].corpus_id == "987654"
    assert results[0].title == "A Neural Information Retrieval Paper"
    assert results[0].external_ids == {"ArXiv": "2401.00001", "DOI": "10.123/test"}


def test_semantic_scholar_client_maps_snippet_search_results(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return FakeResponse(
            {
                "data": [
                    {
                        "paper": {
                            "corpusId": 123456,
                            "title": "Snippet Search Paper",
                            "authors": [["Ada Lovelace"]],
                        },
                        "score": 0.93,
                        "snippet": {
                            "text": "A decisive passage from the paper body.",
                            "snippetKind": "body",
                            "section": "Methods",
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(academic_search, "urlopen", fake_urlopen)
    client = SemanticScholarClient(base_url="https://api.test/graph/v1")

    results = client.search_snippets("paper body", limit=4, filters={"year": "2021-2025"})

    assert "/snippet/search?" in captured["url"]
    assert "year=2021-2025" in captured["url"]
    assert results[0].paper_id == "s2-corpus:123456"
    assert results[0].score == 0.93
    assert results[0].section == "Methods"
    assert results[0].authors == ("Ada Lovelace",)


def test_search_planner_pushes_time_and_venue_filters_to_semantic_scholar():
    from packages.scholar_core.query_understanding.parser import QueryParser

    parsed = QueryParser().parse("Find ACL papers about neural retrieval after 2020")
    plan = SearchPlanner(
        academic_search_enabled=True,
        academic_search_query_limit=1,
        academic_search_snippet_enabled=True,
    ).plan(parsed)

    actions = [action for action in plan.search_actions if action.source.startswith("semantic_scholar")]

    assert len(actions) == 2
    assert all(action.filters == {"year": "2020-", "venue": "ACL"} for action in actions)


def test_semantic_scholar_client_tracks_retry_and_cache_counts(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise URLError("temporary S2 outage")
        return FakeResponse(
            {
                "data": [
                    {
                        "paperId": "retry123",
                        "title": "Snippet Aware Retrieval",
                        "abstract": "A Semantic Scholar result with matched snippets.",
                        "externalIds": {"CorpusId": 123456, "ArXiv": "2601.00001"},
                    }
                ]
            }
        )

    monkeypatch.setattr(academic_search, "urlopen", fake_urlopen)
    client = SemanticScholarClient(
        base_url="https://api.test/graph/v1",
        timeout=1.0,
        max_retries=1,
        cache_path=str(tmp_path / "semantic-scholar-cache.json"),
    )

    first = client.search("snippet retrieval", limit=3)
    second = client.search("snippet retrieval", limit=3)

    assert [item.paper_id for item in first] == ["s2:retry123"]
    assert second[0].title == first[0].title
    assert len(calls) == 2
    assert client.stats()["requests"] == 1
    assert client.stats()["retries"] == 1
    assert client.stats()["cache_hits"] == 1
    assert client.stats()["cache_entries"] == 1


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
            "search": lambda self, query, limit, filters=None: [
                academic_search.AcademicSearchResult(
                    paper_id="s2:abc123",
                    title="External Retrieval Paper",
                    abstract="A paper about academic search APIs.",
                    year=2025,
                    venue="TestConf",
                    citation_count=11,
                    url="https://www.semanticscholar.org/paper/abc123",
                    external_ids={"ArXiv": "2501.00001", "DOI": "10.123/test"},
                    corpus_id="7654321",
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
    assert "s2-corpus:7654321" in candidates[0].aliases
    assert candidates[0].metadata["academic_api_provider"] == "semantic_scholar"


def test_semantic_scholar_external_ids_merge_with_local_candidates_across_sources():
    from packages.scholar_core.models import Candidate
    from packages.scholar_core.normalization.normalizer import CandidateNormalizer

    local_candidate = Candidate(
        "arxiv:2601.00001",
        "Cross Source Candidate Fusion",
        abstract="A local paper.",
        aliases={"arxiv:2601.00001", "doi:10.555/fusion"},
        sources={"local_title_bm25"},
        raw_scores={"local_title_bm25": 9.0},
    )
    s2_candidate = Candidate(
        "s2:abc123",
        "Cross Source Candidate Fusion",
        abstract="A Semantic Scholar paper.",
        aliases={"s2:abc123", "arxiv:2601.00001", "doi:10.555/fusion", "s2-corpus:987654"},
        sources={"semantic_scholar", "semantic_scholar_snippet", "academic_api"},
        raw_scores={"semantic_scholar": 1.0, "semantic_scholar_snippet": 0.8},
        snippets=["Semantic Scholar snippet"],
        metadata={"external_ids": {"ArXiv": "2601.00001", "DOI": "10.555/fusion", "CorpusId": 987654}},
    )

    merged = CandidateNormalizer().merge([local_candidate, s2_candidate])

    assert len(merged) == 1
    assert {"local_title_bm25", "semantic_scholar", "semantic_scholar_snippet", "academic_api"} <= merged[0].sources
    assert {"arxiv:2601.00001", "doi:10.555/fusion", "s2:abc123", "s2-corpus:987654"} <= merged[0].aliases
    assert merged[0].raw_scores["semantic_scholar_snippet"] == 0.8


def test_local_corpus_routes_semantic_scholar_actions():
    corpus = LocalCorpus.__new__(LocalCorpus)
    corpus._academic_search_error = None
    corpus._semantic_scholar = type(
        "Client",
        (),
        {
            "search": lambda self, query, limit, filters=None: [
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
