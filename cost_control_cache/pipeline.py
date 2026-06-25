# 中文功能说明：旧版检索流水线兼容入口，保持 processed_dir 构造方式并转发到新分层实现。

from __future__ import annotations

from pathlib import Path

from apps.backend.scholar_api.bootstrap.container import _build_corpus, build_search_pipeline
from packages.scholar_core.pipeline import SearchPipeline as CoreSearchPipeline


class SearchPipeline(CoreSearchPipeline):
    def __init__(
        self,
        processed_dir: Path,
        *,
        paper_limit: int | None = None,
        chunk_limit: int | None = None,
        max_chunks_per_paper: int = 4,
        per_query_top_k: int = 30,
        backend: str = "auto",
        model_services=None,
        model_services_enabled: bool | None = None,
    ) -> None:
        if model_services is None:
            pipeline = build_search_pipeline(
                Path(processed_dir),
                paper_limit=paper_limit,
                chunk_limit=chunk_limit,
                max_chunks_per_paper=max_chunks_per_paper,
                per_query_top_k=per_query_top_k,
                backend=backend,
                model_services_enabled=model_services_enabled,
            )
            self.__dict__.update(pipeline.__dict__)
            return
        corpus, backend_error = _build_corpus(
            Path(processed_dir),
            backend=backend,
            paper_limit=paper_limit,
            chunk_limit=chunk_limit,
            max_chunks_per_paper=max_chunks_per_paper,
        )
        super().__init__(
            corpus,
            per_query_top_k=per_query_top_k,
            model_services=model_services,
            backend_error=backend_error,
        )


__all__ = ["SearchPipeline"]
