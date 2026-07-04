# Model service client and configuration tests.

from __future__ import annotations

from packages.scholar_core.models import Candidate
from packages.scholar_infra.config import ScholarSearchSettings
from packages.scholar_infra.model_services import client as model_client
from packages.scholar_infra.model_services.client import QueryRewriteServiceClient, SelectorRerankerServiceClient


def test_model_service_settings_reads_selector_pool_limit(tmp_path, monkeypatch):
    config_path = tmp_path / "database.env"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SCHOLAR_SEARCH_CONFIG", str(config_path))
    monkeypatch.setenv("MODEL_SERVICES_ENABLED", "true")
    monkeypatch.setenv("SELECTOR_RERANKER_CANDIDATE_LIMIT", "50")
    monkeypatch.setenv("SELECTOR_RERANKER_POOL_LIMIT", "500")

    settings = ScholarSearchSettings.from_env().model_services

    assert settings.selector_reranker_candidate_limit == 50
    assert settings.selector_reranker_pool_limit == 500


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
