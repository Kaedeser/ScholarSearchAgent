# 中文功能说明：Elasticsearch 查询结构测试，锁定字段权重、短语通道和 chunk 加权策略。

from __future__ import annotations

from typing import Any

from packages.scholar_infra.persistence.elasticsearch import ElasticsearchClient


class RecordingElasticsearchClient(ElasticsearchClient):
    def __init__(self) -> None:
        super().__init__("http://example.test")
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(self, method: str, path: str, body: Any | None = None, *, timeout: int = 30) -> Any:
        self.calls.append((method, path, body))
        return {"hits": {"hits": []}}


def test_paper_search_uses_phrase_boosts_and_clean_locator_query():
    client = RecordingElasticsearchClient()

    client.search_papers(
        "papers",
        "Which paper proposed to model speech based on HuBERT codes or semantic tokens?",
        top_k=10,
    )

    _, _, body = client.calls[-1]
    assert body is not None
    recall_should = body["query"]["bool"]["should"]
    rescore_should = body["rescore"]["query"]["rescore_query"]["bool"]["should"]

    assert body["track_total_hits"] is False
    assert body["query"]["bool"]["minimum_should_match"] == 1
    assert any(_multi_match_has_field(clause, "title^3") for clause in recall_should)
    assert any(_multi_match_has_field(clause, "title^5") for clause in recall_should)
    assert any(_multi_match_query(clause) == "speech hubert codes semantic tokens" for clause in recall_should)
    assert any(_match_phrase_query(clause, "title") == "hubert codes" for clause in recall_should)
    assert any(_match_phrase_query(clause, "title") == "semantic tokens" for clause in recall_should)
    assert any(_match_query(clause, "title") == "hubert" for clause in recall_should)
    assert any((_match_boost(clause, "title") or 0) > 3.0 for clause in recall_should)
    assert any(_multi_match_has_field(clause, "title^4.2") for clause in rescore_should)
    assert any(_match_phrase_boost(clause, "title") == 12.0 for clause in rescore_should)
    assert body["rescore"]["query"]["rescore_query_weight"] == 0.2


def test_chunk_search_uses_section_phrases_and_title_abstract_weight():
    client = RecordingElasticsearchClient()

    client.search_chunks(
        "chunks",
        "mask classification-based methods for instance-level segmentation",
        top_k=10,
    )

    _, _, body = client.calls[-1]
    assert body is not None
    recall_should = body["query"]["bool"]["should"]
    rescore_should = body["rescore"]["query"]["rescore_query"]["bool"]["should"]

    assert body["query"]["bool"]["minimum_should_match"] == 1
    assert any(_multi_match_has_field(clause, "text") for clause in recall_should)
    assert any(_multi_match_has_field(clause, "section_title^2.4") for clause in recall_should)
    assert any(_match_phrase_query(clause, "section_title") == "mask classification" for clause in recall_should)
    assert any(_match_phrase_query(clause, "section_title") == "instance-level segmentation" for clause in recall_should)
    assert any(_match_query(clause, "section_title") == "segmentation" for clause in recall_should)
    assert {"term": {"chunk_type": {"value": "title_abstract", "boost": 0.2}}} in recall_should
    assert any(_multi_match_has_field(clause, "section_title^2.2") for clause in rescore_should)
    assert {"term": {"chunk_type": {"value": "title_abstract", "boost": 0.25}}} in rescore_should
    assert body["rescore"]["query"]["rescore_query_weight"] == 0.02


def _multi_match_has_field(clause: dict[str, Any], field: str) -> bool:
    multi_match = clause.get("multi_match")
    return bool(multi_match and field in multi_match.get("fields", []))


def _multi_match_query(clause: dict[str, Any]) -> str | None:
    multi_match = clause.get("multi_match")
    if not multi_match:
        return None
    return str(multi_match.get("query") or "")


def _match_phrase_query(clause: dict[str, Any], field: str) -> str | None:
    match_phrase = clause.get("match_phrase")
    if not match_phrase or field not in match_phrase:
        return None
    return str(match_phrase[field].get("query") or "")


def _match_phrase_boost(clause: dict[str, Any], field: str) -> float | None:
    match_phrase = clause.get("match_phrase")
    if not match_phrase or field not in match_phrase:
        return None
    return float(match_phrase[field].get("boost") or 0.0)


def _match_query(clause: dict[str, Any], field: str) -> str | None:
    match = clause.get("match")
    if not match or field not in match:
        return None
    return str(match[field].get("query") or "")


def _match_boost(clause: dict[str, Any], field: str) -> float | None:
    match = clause.get("match")
    if not match or field not in match:
        return None
    return float(match[field].get("boost") or 0.0)
