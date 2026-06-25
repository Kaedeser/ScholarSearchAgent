# 中文功能说明：旧版多源召回兼容入口，实际实现位于 packages/scholar_infra/retrieval_backends/retrieval.py。

from __future__ import annotations

from packages.scholar_infra.retrieval_backends.retrieval import BM25Index, CorpusBackend, DatabaseCorpus, LocalCorpus, RetrievalHit, TfidfIndex

__all__ = ["BM25Index", "CorpusBackend", "DatabaseCorpus", "LocalCorpus", "RetrievalHit", "TfidfIndex"]