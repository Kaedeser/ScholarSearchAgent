# 中文功能说明：后端依赖装配模块，集中创建 SearchPipeline，避免 API 层直接散落构造细节。

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from packages.scholar_core.pipeline import SearchPipeline
from packages.scholar_infra.config import ScholarSearchSettings
from packages.scholar_infra.model_services.client import ModelServices
from packages.scholar_infra.retrieval_backends.retrieval import DatabaseCorpus, LocalCorpus, SemanticScholarCorpus


def build_search_pipeline(
    processed_dir: Path,
    *,
    paper_limit: int | None,
    chunk_limit: int | None,
    max_chunks_per_paper: int,
    per_query_top_k: int,
    backend: str,
    model_services_enabled: bool | None,
) -> SearchPipeline:
    settings = ScholarSearchSettings.from_env()
    model_settings = settings.model_services
    if model_services_enabled is not None:
        model_settings = replace(model_settings, enabled=model_services_enabled)
    model_services = ModelServices.from_settings(model_settings)
    corpus, backend_error = _build_corpus(
        processed_dir,
        backend=backend,
        paper_limit=paper_limit,
        chunk_limit=chunk_limit,
        max_chunks_per_paper=max_chunks_per_paper,
        settings=settings,
    )
    return SearchPipeline(
        corpus,
        per_query_top_k=per_query_top_k,
        model_services=model_services,
        backend_error=backend_error,
        academic_search_enabled=settings.academic_search_enabled,
        academic_search_provider=settings.academic_search_provider,
        academic_search_query_limit=settings.academic_search_query_limit,
        academic_search_top_k=settings.academic_search_top_k,
        academic_search_snippet_enabled=settings.academic_search_snippet_enabled,
        academic_search_snippet_top_k=settings.academic_search_snippet_top_k,
    )


def _build_corpus(
    processed_dir: Path,
    *,
    backend: str,
    paper_limit: int | None,
    chunk_limit: int | None,
    max_chunks_per_paper: int,
    settings: ScholarSearchSettings | None = None,
):
    if backend not in {"auto", "jsonl", "database", "semantic_scholar"}:
        raise ValueError(f"Unsupported backend: {backend}")
    settings = settings or ScholarSearchSettings.from_env()
    if backend == "semantic_scholar":
        return SemanticScholarCorpus(settings), None
    if backend in {"auto", "database"}:
        try:
            return DatabaseCorpus(), None
        except Exception as exc:
            if backend == "database":
                raise
            backend_error = str(exc)
        if settings.academic_search_enabled and settings.academic_search_provider == "semantic_scholar":
            try:
                return SemanticScholarCorpus(settings), backend_error
            except Exception as exc:
                backend_error = f"{backend_error}; Semantic Scholar fallback failed: {exc}"
        return (
            LocalCorpus(
                processed_dir,
                paper_limit=paper_limit,
                chunk_limit=chunk_limit,
                max_chunks_per_paper=max_chunks_per_paper,
            ),
            backend_error,
        )
    return (
        LocalCorpus(
            processed_dir,
            paper_limit=paper_limit,
            chunk_limit=chunk_limit,
            max_chunks_per_paper=max_chunks_per_paper,
        ),
        None,
    )
