# Model service client and configuration tests.

from __future__ import annotations

from packages.scholar_core.models import Candidate
from packages.scholar_infra.config import ScholarSearchSettings
from packages.scholar_infra.model_services import client as model_client
from packages.scholar_infra.model_services.client import QueryRewriteServiceClient, SelectorRerankerServiceClient


def test_model_services_default_to_trained_services_enabled(tmp_path, monkeypatch):
    config_path = tmp_path / "database.env"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SCHOLAR_SEARCH_CONFIG", str(config_path))
    for name in (
        "MODEL_SERVICES_ENABLED",
        "QUERY_INTENT_ENABLED",
        "QUERY_REWRITE_ENABLED",
        "SELECTOR_RERANKER_ENABLED",
        "CRAWLER_STRATEGY_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = ScholarSearchSettings.from_env().model_services

    assert settings.enabled is True
    assert settings.query_intent_enabled is True
    assert settings.selector_reranker_enabled is True
    assert settings.crawler_strategy_enabled is True
    assert settings.query_rewrite_enabled is False


def test_model_service_settings_reads_selector_pool_limit(tmp_path, monkeypatch):
    config_path = tmp_path / "database.env"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SCHOLAR_SEARCH_CONFIG", str(config_path))
    monkeypatch.setenv("MODEL_SERVICES_ENABLED", "true")
    monkeypatch.setenv("SELECTOR_RERANKER_CANDIDATE_LIMIT", "120")
    monkeypatch.setenv("SELECTOR_RERANKER_POOL_LIMIT", "500")
    monkeypatch.setenv("SELECTOR_RERANKER_PROTECTED_HEAD", "0")

    settings = ScholarSearchSettings.from_env().model_services

    assert settings.selector_reranker_candidate_limit == 120
    assert settings.selector_reranker_pool_limit == 500
    assert settings.selector_reranker_protected_head == 0


def test_scholar_settings_reads_academic_search_options(tmp_path, monkeypatch):
    config_path = tmp_path / "database.env"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SCHOLAR_SEARCH_CONFIG", str(config_path))
    monkeypatch.setenv("ACADEMIC_SEARCH_ENABLED", "true")
    monkeypatch.setenv("ACADEMIC_SEARCH_PROVIDER", "semantic_scholar")
    monkeypatch.setenv("ACADEMIC_SEARCH_BASE_URL", "https://api.test/graph/v1")
    monkeypatch.setenv("ACADEMIC_SEARCH_API_KEY", "secret")
    monkeypatch.setenv("ACADEMIC_SEARCH_QUERY_LIMIT", "3")
    monkeypatch.setenv("ACADEMIC_SEARCH_TOP_K", "25")

    settings = ScholarSearchSettings.from_env()

    assert settings.academic_search_enabled is True
    assert settings.academic_search_provider == "semantic_scholar"
    assert settings.academic_search_base_url == "https://api.test/graph/v1"
    assert settings.academic_search_api_key == "secret"
    assert settings.academic_search_query_limit == 3
    assert settings.academic_search_top_k == 25


def test_scholar_settings_reads_academic_search_retry_cache_and_snippet_options(tmp_path, monkeypatch):
    config_path = tmp_path / "database.env"
    cache_path = tmp_path / "semantic-scholar-cache.json"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SCHOLAR_SEARCH_CONFIG", str(config_path))
    monkeypatch.setenv("ACADEMIC_SEARCH_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("ACADEMIC_SEARCH_MAX_RETRIES", "2")
    monkeypatch.setenv("ACADEMIC_SEARCH_SNIPPET_TOP_K", "6")

    settings = ScholarSearchSettings.from_env()

    assert settings.academic_search_cache_path == str(cache_path)
    assert settings.academic_search_max_retries == 2
    assert settings.academic_search_snippet_top_k == 6


def test_selector_reranker_sorts_after_score_fusion(monkeypatch):
    def fake_post_json(base_url, path, payload, *, timeout):
        return {
            "count": 36,
            "threshold": 0.5,
            "results": [
                *[
                    {"id": f"b{index}", "paper_id": f"b{index}", "score": 0.5, "relevant": True}
                    for index in range(35)
                ],
                {"id": "a", "paper_id": "a", "score": 0.0, "relevant": False},
            ],
        }

    monkeypatch.setattr(model_client, "_post_json", fake_post_json)
    service = SelectorRerankerServiceClient("http://selector.test", timeout=1.0)
    alias_candidate = Candidate(
        "a",
        "TranAD",
        raw_scores={"local_title_bm25": 10.0},
        metadata={"soft_alias_bonus": 1.0},
    )
    alias_candidate.final_score = 1.0
    model_candidates = [
        Candidate(f"b{index}", "Generic anomaly detection", raw_scores={"local_title_bm25": 1.0})
        for index in range(35)
    ]
    for candidate in model_candidates:
        candidate.final_score = 0.0

    reranked, _ = service.rerank("query", [*model_candidates, alias_candidate], top_k=36)

    assert reranked[30].paper_id == "a"
    assert reranked.index(alias_candidate) < 35


def test_selector_reranker_request_documents_include_candidate_snippets(monkeypatch):
    captured = {}

    def fake_post_json(base_url, path, payload, *, timeout):
        captured["payload"] = payload
        return {
            "count": 1,
            "threshold": 0.5,
            "results": [{"id": "s2:snippet", "paper_id": "s2:snippet", "score": 0.9, "relevant": True}],
        }

    monkeypatch.setattr(model_client, "_post_json", fake_post_json)
    service = SelectorRerankerServiceClient("http://selector.test", timeout=1.0)
    candidate = Candidate(
        "s2:snippet",
        "Snippet Aware Paper",
        abstract="Abstract alone omits the decisive evidence.",
        snippets=["Decisive Semantic Scholar snippet evidence for reranking."],
    )

    service.rerank("snippet evidence", [candidate], top_k=1)

    serialized_document = str(captured["payload"]["documents"][0])
    assert "Decisive Semantic Scholar snippet evidence for reranking." in serialized_document


def test_query_rewrite_client_parses_and_caches_openai_response(tmp_path, monkeypatch):
    calls = []

    def fake_post_json(base_url, path, payload, *, timeout, headers, verify_ssl=True):
        calls.append((base_url, path, payload, headers))
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"rewrites":["zero-shot generated text detection"],'
                            '"concepts":["machine-generated text detection"],'
                            '"possible_answer_terms":["DetectGPT"]}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(model_client, "_post_json_with_headers", fake_post_json)
    service = QueryRewriteServiceClient(
        "http://gpustack.test",
        api_key="secret",
        model="qwen-test",
        timeout=1.0,
        max_rewrites=3,
        cache_path=str(tmp_path / "rewrite-cache.json"),
    )

    first = service.rewrite("Can LLMs detect LLM-generated text?")
    second = service.rewrite("Can LLMs detect LLM-generated text?")

    assert first.rewrites == ["zero-shot generated text detection"]
    assert first.concepts == ["machine-generated text detection"]
    assert second.cache_hit is True
    assert len(calls) == 1
    assert calls[0][0] == "http://gpustack.test/v1-openai"
    assert calls[0][1] == "/chat/completions"
    assert calls[0][3]["Authorization"] == "Bearer secret"
