# Backend container routing tests.

from __future__ import annotations

from pathlib import Path

import pytest

from apps.backend.scholar_api.bootstrap import container


def _configure_semantic_scholar(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "database.env"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SCHOLAR_SEARCH_CONFIG", str(config_path))
    monkeypatch.setenv("ACADEMIC_SEARCH_ENABLED", "true")
    monkeypatch.setenv("ACADEMIC_SEARCH_PROVIDER", "semantic_scholar")
    monkeypatch.setenv("ACADEMIC_SEARCH_API_KEY", "test-key")


def test_build_corpus_routes_semantic_scholar_only_without_local_files(monkeypatch, tmp_path):
    _configure_semantic_scholar(monkeypatch, tmp_path)
    monkeypatch.setattr(
        container,
        "DatabaseCorpus",
        lambda: pytest.fail("semantic_scholar backend must not open database connections"),
    )
    monkeypatch.setattr(
        container,
        "LocalCorpus",
        lambda *args, **kwargs: pytest.fail("semantic_scholar backend must not require local JSONL files"),
    )

    corpus, backend_error = container._build_corpus(
        tmp_path / "missing-processed-dir",
        backend="semantic_scholar",
        paper_limit=None,
        chunk_limit=None,
        max_chunks_per_paper=4,
    )

    assert backend_error is None
    assert corpus.backend_name == "semantic_scholar"
    assert corpus.stats()["academic_search_provider"] == "semantic_scholar"


def test_auto_backend_prefers_semantic_scholar_after_database_failure_without_local_files(monkeypatch, tmp_path):
    _configure_semantic_scholar(monkeypatch, tmp_path)

    def raise_database_error():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(container, "DatabaseCorpus", raise_database_error)
    monkeypatch.setattr(
        container,
        "LocalCorpus",
        lambda *args, **kwargs: pytest.fail("auto fallback should prefer Semantic Scholar before local JSONL"),
    )

    corpus, backend_error = container._build_corpus(
        tmp_path / "missing-processed-dir",
        backend="auto",
        paper_limit=None,
        chunk_limit=None,
        max_chunks_per_paper=4,
    )

    assert "database unavailable" in backend_error
    assert corpus.backend_name == "semantic_scholar"
    assert corpus.stats()["academic_search_provider"] == "semantic_scholar"
