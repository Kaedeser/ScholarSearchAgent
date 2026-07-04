# 中文功能说明：Qdrant 稀疏召回测试，覆盖查询向量清洗和 chunk 正文证据回填。

from __future__ import annotations

from typing import Any

from packages.scholar_core.models import SearchAction
from packages.scholar_core.retrieval.weighted_query import (
    analyze_weighted_query,
    dense_retrieval_query_text,
    sparse_document_feature_map,
    weighted_sparse_query_feature_map,
)
from packages.scholar_infra.persistence.qdrant import QdrantClient, lexical_sparse_query_vector, lexical_sparse_vector
from packages.scholar_infra.retrieval_backends.retrieval import BM25Index, DatabaseCorpus
from packages.scholar_ingest.qdrant import qdrant_point_from_sparse_paper


def test_qdrant_query_vector_filters_and_weights_question_terms():
    dimensions = 256

    clean = lexical_sparse_query_vector("Which paper proposed anomaly score?", dimensions)
    unfiltered = lexical_sparse_vector("Which paper proposed anomaly score?", dimensions)

    assert clean != unfiltered
    assert clean["indices"]
    assert max(clean["values"]) > 1.0


def test_weighted_query_boosts_scholarly_terms_and_phrases():
    analyzed = analyze_weighted_query("Which paper proposed anomaly score via reconstruction error?")
    term_weights = {item.term: item.weight for item in analyzed.terms}
    phrases = {item.term for item in analyzed.phrases}

    assert "paper" not in term_weights
    assert term_weights["anomaly"] > 1.0
    assert term_weights["score"] >= 1.0
    assert "anomaly score" in phrases or "reconstruction error" in phrases


def test_weighted_query_expands_scholarly_aliases():
    analyzed = analyze_weighted_query("Which paper introduced CoT prompting for LLM reasoning?")
    terms = {item.term for item in analyzed.terms}
    phrases = {item.term for item in analyzed.phrases}

    assert "cot" in terms
    assert "llm" in terms or "llms" in terms
    assert "chain of thought prompting" in phrases


def test_weighted_query_expands_real_query_bridges():
    diffusion = analyze_weighted_query(
        "Show me papers utilizing reinforcement learning to optimize diffusion models for video generation."
    )
    games = analyze_weighted_query("Find large vision-language model agents that play PC games.")
    ranking = analyze_weighted_query("Give me papers about how to rank search results by the use of LLM.")

    diffusion_phrases = {item.term for item in diffusion.phrases}
    games_phrases = {item.term for item in games.phrases}
    ranking_phrases = {item.term for item in ranking.phrases}

    assert "video diffusion alignment" in diffusion_phrases
    assert "reward gradients" in diffusion_phrases
    assert "computer control" in games_phrases
    assert "document reranking" in ranking_phrases


def test_sparse_feature_maps_align_phrase_features_between_query_and_document():
    query_features = weighted_sparse_query_feature_map("Which paper proposed anomaly score?")
    doc_features = sparse_document_feature_map(
        "TranAD computes an anomaly score from reconstruction error and discriminator loss."
    )

    assert "phrase:anomaly score" in query_features
    assert "phrase:anomaly score" in doc_features
    assert query_features["phrase:anomaly score"] > query_features["score"]


def test_bm25_query_alias_expansion_matches_full_form_document():
    index = BM25Index(
        {
            "full": "Chain of Thought Prompting Elicits Reasoning in Large Language Models",
            "other": "Prompt tuning for visual classification benchmarks",
        }
    )

    hits = index.search("CoT prompting", top_k=2)

    assert hits
    assert hits[0].paper_id == "full"


def test_dense_retrieval_query_text_includes_alias_expansions():
    dense_query = dense_retrieval_query_text("CoT prompting for VLMs")

    assert "cot" in dense_query
    assert "chain of thought prompting" in dense_query
    assert "vision language models" in dense_query


def test_qdrant_candidate_uses_chunk_text_as_evidence():
    corpus = DatabaseCorpus.__new__(DatabaseCorpus)
    corpus._paper = lambda paper_id: {  # type: ignore[method-assign]
        "title": "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data",
        "abstract": "Transformer anomaly detection for multivariate time series.",
        "year": 2022,
        "venue": None,
        "citation_count": 100,
    }
    hit: dict[str, Any] = {
        "id": "point-1",
        "score": 42.0,
        "payload": {
            "paper_id": "arxiv:2201.07284",
            "chunk_id": "arxiv:2201.07284#chunk:1",
            "chunk_type": "section_references",
            "section_title": "1. Introduction",
        },
    }
    chunks = {
        "arxiv:2201.07284#chunk:1": {
            "chunk_id": "arxiv:2201.07284#chunk:1",
            "chunk_type": "section_references",
            "section_title": "1. Introduction",
            "text": "TranAD combines reconstruction error and adversarial discriminator loss for anomaly score.",
        }
    }

    candidate = DatabaseCorpus._candidate_from_qdrant(
        corpus,
        hit,
        SearchAction("local_tfidf", "reconstruction error discriminator loss", 10, 1.0),
        chunks_by_id=chunks,
    )

    assert candidate.paper_id == "arxiv:2201.07284"
    assert {"local_tfidf", "qdrant"} <= candidate.sources
    assert "reconstruction error" in candidate.snippets[0].lower()
    assert candidate.metadata["chunk_type"] == "section_references"
    assert candidate.metadata["qdrant_score"] == 42.0


def test_qdrant_dense_collection_can_use_named_vector(monkeypatch):
    captured = {}
    client = QdrantClient("http://qdrant.test")

    def fake_request(method, path, body=None):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = body
        return {"result": True}

    monkeypatch.setattr(client, "request", fake_request)

    client.create_collection("dense", 768, vector_name="paper")

    assert captured["method"] == "PUT"
    assert captured["path"] == "/collections/dense"
    assert captured["body"]["vectors"] == {"paper": {"size": 768, "distance": "Cosine"}}


def test_dense_paper_candidate_marks_dense_usage():
    corpus = DatabaseCorpus.__new__(DatabaseCorpus)
    corpus.settings = type("Settings", (), {"qdrant_dense_paper_collection": "saiti3_papers_dense_v1"})()
    corpus._paper = lambda paper_id: {  # type: ignore[method-assign]
        "title": "Dense Retrieval for Scientific Papers",
        "abstract": "Semantic paper-level retrieval.",
        "year": 2024,
        "venue": "TestConf",
        "citation_count": 5,
    }
    hit: dict[str, Any] = {
        "id": "point-1",
        "score": 0.87,
        "payload": {"paper_id": "arxiv:2401.00001", "title": "Dense Retrieval for Scientific Papers"},
    }

    candidate = DatabaseCorpus._candidate_from_dense_paper(
        corpus,
        hit,
        SearchAction("qdrant_dense_paper", "semantic paper retrieval", 10, 1.0),
        dense_query="semantic paper retrieval dense title abstract",
    )

    assert "qdrant_dense_paper" in candidate.sources
    assert candidate.metadata["dense_used"] is True
    assert candidate.metadata["qdrant_collection"] == "saiti3_papers_dense_v1"
    assert candidate.metadata["dense_query"] == "semantic paper retrieval dense title abstract"


def test_sparse_paper_point_uses_title_and_abstract_payload():
    point = qdrant_point_from_sparse_paper(
        {
            "paper_id": "arxiv:2401.00001",
            "title": "Chain of Thought Prompting",
            "abstract": "Reasoning with intermediate steps.",
            "year": 2024,
            "venue": "TestConf",
            "source": "pasa",
        },
        vector_size=256,
        sparse_vector_name="text",
    )

    assert point["payload"]["paper_id"] == "arxiv:2401.00001"
    assert point["payload"]["text_type"] == "title_abs_sparse"
    assert point["vector"]["text"]["indices"]


def test_sparse_paper_candidate_marks_paper_level_usage():
    corpus = DatabaseCorpus.__new__(DatabaseCorpus)
    corpus.settings = type("Settings", (), {"qdrant_sparse_paper_collection": "saiti3_papers_sparse_v1"})()
    corpus._paper = lambda paper_id: {  # type: ignore[method-assign]
        "title": "Sparse Paper Retrieval",
        "abstract": "Paper-level sparse retrieval.",
        "year": 2024,
        "venue": "TestConf",
        "citation_count": 3,
    }
    hit: dict[str, Any] = {
        "id": "point-1",
        "score": 8.0,
        "payload": {"paper_id": "arxiv:2401.00002", "title": "Sparse Paper Retrieval"},
    }

    candidate = DatabaseCorpus._candidate_from_sparse_paper(
        corpus,
        hit,
        SearchAction("qdrant_sparse_paper", "paper retrieval", 10, 1.0),
    )

    assert "qdrant_sparse_paper" in candidate.sources
    assert candidate.metadata["sparse_paper_used"] is True
    assert candidate.metadata["qdrant_collection"] == "saiti3_papers_sparse_v1"


def test_dense_paper_text_keeps_title_prominent():
    from packages.scholar_ingest.cli import _dense_paper_text

    text = _dense_paper_text(
        {
            "title": "Chain of Thought Prompting Elicits Reasoning",
            "abstract": "A" * 3000,
            "year": 2022,
            "venue": "NeurIPS",
        }
    )

    assert text.startswith("title: Chain of Thought")
    assert "metadata: NeurIPS 2022" in text
    assert len(text) < 2350
