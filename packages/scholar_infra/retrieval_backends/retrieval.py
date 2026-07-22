# 中文功能说明：多源召回模块，提供本地 JSONL 检索和真实数据库检索后端适配。

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
import re
from typing import Any

from packages.scholar_infra.io.jsonl import ensure_paths_exist, load_chunks_by_paper, load_papers, read_jsonl
from packages.scholar_core.models import Candidate, Paper, SearchAction
from packages.scholar_core.retrieval.weighted_query import (
    analyze_weighted_query,
    dense_retrieval_query_text,
    weighted_query_tokens,
    weighted_token_map,
)
from packages.scholar_core.text import best_snippet, cosine_sparse, token_counter, tokenize
from packages.scholar_infra.academic_search import AcademicSearchError, SemanticScholarClient
from packages.scholar_infra.embeddings import DenseEmbedder, build_dense_embedder
from packages.scholar_infra.persistence.neo4j import Neo4jGraphClient


@dataclass(frozen=True)
class RetrievalHit:
    paper_id: str
    score: float
    source: str
    snippet: str = ""


class BM25Index:
    def __init__(self, documents: dict[str, str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_freq: Counter[str] = Counter()
        for doc_id, text in documents.items():
            counts = token_counter(text)
            self.doc_tokens[doc_id] = counts
            self.doc_lengths[doc_id] = sum(counts.values())
            for token in counts:
                self.doc_freq[token] += 1
        self.doc_count = max(1, len(documents))
        self.avg_doc_len = sum(self.doc_lengths.values()) / self.doc_count if documents else 1.0

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        query_terms = weighted_query_tokens(query)
        if not query_terms:
            return []
        query_weights = weighted_token_map(query)
        scores: defaultdict[str, float] = defaultdict(float)
        query_term_set = set(query_terms)
        for doc_id, counts in self.doc_tokens.items():
            doc_len = self.doc_lengths.get(doc_id, 0) or 1
            score = 0.0
            for term in query_term_set:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += query_weights.get(term, 1.0) * idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scores[doc_id] = score
        return [
            RetrievalHit(doc_id, score, "bm25", best_snippet(self.documents[doc_id], set(query_terms)))
            for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]


class TfidfIndex:
    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents
        doc_counts = {doc_id: token_counter(text) for doc_id, text in documents.items()}
        doc_freq: Counter[str] = Counter()
        for counts in doc_counts.values():
            for token in counts:
                doc_freq[token] += 1
        doc_count = max(1, len(documents))
        self.vectors: dict[str, Counter[str]] = {}
        for doc_id, counts in doc_counts.items():
            vector: Counter[str] = Counter()
            for token, value in counts.items():
                vector[token] = value * log(1 + doc_count / (1 + doc_freq[token]))
            self.vectors[doc_id] = vector
        self.doc_count = doc_count
        self.doc_freq = doc_freq

    def vectorize_query(self, query: str) -> Counter[str]:
        counts: Counter[str] = Counter(weighted_query_tokens(query))
        query_weights = weighted_token_map(query)
        vector: Counter[str] = Counter()
        for token, value in counts.items():
            vector[token] = (
                value
                * query_weights.get(token, 1.0)
                * log(1 + self.doc_count / (1 + self.doc_freq.get(token, 0)))
            )
        return vector

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        query_vector = self.vectorize_query(query)
        if not query_vector:
            return []
        scored = []
        query_tokens = set(query_vector)
        for doc_id, vector in self.vectors.items():
            score = cosine_sparse(query_vector, vector)
            if score > 0:
                scored.append((doc_id, score))
        return [
            RetrievalHit(doc_id, score, "tfidf", best_snippet(self.documents[doc_id], query_tokens))
            for doc_id, score in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        ]


def _semantic_scholar_client_from_settings(
    settings: Any,
) -> tuple[SemanticScholarClient | None, str | None]:
    if not getattr(settings, "academic_search_enabled", False):
        return None, "academic search disabled"
    provider = getattr(settings, "academic_search_provider", "semantic_scholar")
    if provider != "semantic_scholar":
        return None, f"unsupported academic search provider: {provider}"
    return (
        SemanticScholarClient(
            base_url=settings.academic_search_base_url,
            api_key=settings.academic_search_api_key,
            timeout=settings.academic_search_timeout_sec,
            max_retries=getattr(settings, "academic_search_max_retries", 2),
            retry_backoff_sec=getattr(settings, "academic_search_retry_backoff_sec", 1.0),
            min_interval_sec=getattr(settings, "academic_search_min_interval_sec", 1.0),
            cache_size=getattr(settings, "academic_search_cache_size", 256),
            cache_path=getattr(settings, "academic_search_cache_path", ""),
        ),
        None,
    )


def _semantic_scholar_candidates(
    client: SemanticScholarClient,
    action: SearchAction,
) -> list[Candidate]:
    results = client.search(action.query, limit=action.top_k, filters=action.filters or None)
    candidates: list[Candidate] = []
    for rank, result in enumerate(results, start=1):
        raw_score = action.weight / rank
        aliases = {result.paper_id, *_external_aliases(result.external_ids or {})}
        if result.corpus_id:
            aliases.add(f"s2-corpus:{result.corpus_id}")
        candidates.append(
            Candidate(
                paper_id=result.paper_id,
                title=result.title,
                abstract=result.abstract,
                year=result.year,
                venue=result.venue,
                citation_count=result.citation_count,
                aliases=aliases,
                sources={action.source, "academic_api"},
                raw_scores={action.source: raw_score},
                snippets=[best_snippet(f"{result.title}. {result.abstract}", set(tokenize(action.query)))],
                metadata={
                    "source_query": action.query,
                    "academic_api_provider": "semantic_scholar",
                    "academic_api_url": result.url,
                    "external_ids": result.external_ids or {},
                    "semantic_scholar_corpus_id": result.corpus_id,
                    "academic_api_rank": rank,
                    "influential_citation_count": result.influential_citation_count,
                    "reference_count": result.reference_count,
                    "fields_of_study": list(result.fields_of_study),
                    "publication_types": list(result.publication_types),
                    "publication_date": result.publication_date,
                    "authors": list(result.authors),
                },
            )
        )
    return candidates


def _semantic_scholar_snippet_candidates(
    client: SemanticScholarClient,
    action: SearchAction,
) -> list[Candidate]:
    results = client.search_snippets(action.query, limit=action.top_k, filters=action.filters or None)
    candidates: list[Candidate] = []
    for rank, result in enumerate(results, start=1):
        paper_id = result.paper_id or f"s2-snippet:{rank}"
        aliases = {paper_id}
        if result.corpus_id:
            aliases.add(f"s2-corpus:{result.corpus_id}")
        candidates.append(
            Candidate(
                paper_id=paper_id,
                title=result.title,
                aliases=aliases,
                sources={action.source, "academic_api"},
                raw_scores={action.source: max(0.0, result.score) * action.weight},
                snippets=[result.text] if result.text else [],
                metadata={
                    "source_query": action.query,
                    "academic_api_provider": "semantic_scholar",
                    "academic_api_rank": rank,
                    "snippet_score": result.score,
                    "snippet_section": result.section,
                    "snippet_kind": result.snippet_kind,
                    "authors": list(result.authors),
                },
            )
        )
    return candidates


class SemanticScholarCorpus:
    """API-only corpus used when the submission has no local retrieval databases."""

    backend_name = "semantic_scholar"

    def __init__(self, settings: Any | None = None) -> None:
        if settings is None:
            from packages.scholar_infra.config import ScholarSearchSettings

            settings = ScholarSearchSettings.from_env()
        self.settings = settings
        self._semantic_scholar, self._academic_search_error = _semantic_scholar_client_from_settings(settings)
        if self._semantic_scholar is None:
            raise ValueError(self._academic_search_error or "Semantic Scholar is not configured")

    @property
    def supported_sources(self) -> set[str]:
        sources = {"semantic_scholar"}
        if getattr(self.settings, "academic_search_snippet_enabled", True):
            sources.add("semantic_scholar_snippet")
        return sources

    def run_action(self, action: SearchAction) -> list[Candidate]:
        if action.source not in self.supported_sources or self._semantic_scholar is None:
            return []
        try:
            if action.source == "semantic_scholar_snippet":
                return _semantic_scholar_snippet_candidates(self._semantic_scholar, action)
            return _semantic_scholar_candidates(self._semantic_scholar, action)
        except AcademicSearchError as exc:
            self._academic_search_error = str(exc)
            return []

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "backend": self.backend_name,
            "academic_search_status": "enabled",
            "academic_search_provider": "semantic_scholar",
        }
        if self._semantic_scholar is not None:
            stats.update(self._semantic_scholar.stats())
        if self._academic_search_error:
            stats["academic_search_error"] = self._academic_search_error
        return stats


class LocalCorpus:
    backend_name = "jsonl"

    def __init__(
        self,
        processed_dir: Path,
        *,
        paper_limit: int | None = None,
        chunk_limit: int | None = None,
        max_chunks_per_paper: int = 4,
    ) -> None:
        ensure_paths_exist(
            [
                processed_dir / "papers.jsonl",
                processed_dir / "paper_chunks.jsonl",
                processed_dir / "queries.jsonl",
                processed_dir / "eval_sets.jsonl",
            ]
        )
        self.processed_dir = processed_dir
        self.papers = load_papers(processed_dir, paper_limit)
        self.chunks_by_paper = load_chunks_by_paper(
            processed_dir,
            set(self.papers),
            limit=chunk_limit,
            max_chunks_per_paper=max_chunks_per_paper,
        )
        self.section_titles_by_paper = _load_section_titles_by_paper(
            processed_dir,
            set(self.papers),
            limit=chunk_limit,
            max_chunks_per_paper=max_chunks_per_paper,
        )
        title_docs = {paper_id: paper.title for paper_id, paper in self.papers.items()}
        chunk_docs = {
            paper_id: "\n".join(chunks) if chunks else self._paper_text(self.papers[paper_id])
            for paper_id, chunks in self.chunks_by_paper.items()
        }
        for paper_id, paper in self.papers.items():
            chunk_docs.setdefault(paper_id, self._paper_text(paper))
        semantic_docs = {paper_id: self._semantic_text(paper_id, paper) for paper_id, paper in self.papers.items()}
        self.title_bm25 = BM25Index(title_docs)
        self.chunk_bm25 = BM25Index(chunk_docs)
        self.tfidf = TfidfIndex(semantic_docs)
        self.semantic_docs = semantic_docs
        self.chunk_docs = chunk_docs
        from packages.scholar_infra.config import ScholarSearchSettings

        self._semantic_scholar, self._academic_search_error = _semantic_scholar_client_from_settings(
            ScholarSearchSettings.from_env()
        )

    def _paper_text(self, paper: Paper) -> str:
        return f"{paper.title}\n{paper.abstract or ''}"

    def _semantic_text(self, paper_id: str, paper: Paper) -> str:
        chunks = " ".join(self.chunks_by_paper.get(paper_id, [])[:2])
        return f"{paper.title}\n{paper.abstract or ''}\n{chunks}"

    def run_action(self, action: SearchAction) -> list[Candidate]:
        if action.source == "semantic_scholar":
            return self._search_semantic_scholar(action)
        if action.source == "semantic_scholar_snippet":
            return self._search_semantic_scholar_snippets(action)
        if action.source == "local_title_bm25":
            hits = self.title_bm25.search(action.query, top_k=action.top_k)
        elif action.source == "local_chunk_bm25":
            hits = self.chunk_bm25.search(action.query, top_k=action.top_k)
        elif action.source == "local_tfidf":
            hits = self.tfidf.search(action.query, top_k=action.top_k)
        else:
            return []
        return [self._candidate_from_hit(hit, action) for hit in hits]

    @property
    def supported_sources(self) -> set[str]:
        sources = {"local_title_bm25", "local_chunk_bm25", "local_tfidf"}
        if self._semantic_scholar is not None:
            sources.add("semantic_scholar")
            sources.add("semantic_scholar_snippet")
        return sources

    def stats(self) -> dict:
        stats: dict[str, Any] = {
            "backend": self.backend_name,
            "papers_loaded": len(self.papers),
            "chunks_loaded": sum(len(chunks) for chunks in self.chunks_by_paper.values()),
        }
        if self._semantic_scholar is not None:
            stats["academic_search_status"] = "enabled"
            stats["academic_search_provider"] = "semantic_scholar"
            stats.update(self._semantic_scholar.stats())
        elif self._academic_search_error is not None:
            stats["academic_search_error"] = self._academic_search_error
        return stats

    def _search_semantic_scholar(self, action: SearchAction) -> list[Candidate]:
        if self._semantic_scholar is None:
            return []
        try:
            return _semantic_scholar_candidates(self._semantic_scholar, action)
        except AcademicSearchError as exc:
            self._academic_search_error = str(exc)
            return []

    def _search_semantic_scholar_snippets(self, action: SearchAction) -> list[Candidate]:
        if self._semantic_scholar is None:
            return []
        try:
            return _semantic_scholar_snippet_candidates(self._semantic_scholar, action)
        except AcademicSearchError as exc:
            self._academic_search_error = str(exc)
            return []

    def _candidate_from_hit(self, hit: RetrievalHit, action: SearchAction) -> Candidate:
        paper = self.papers[hit.paper_id]
        score_key = action.source
        weighted_score = hit.score * action.weight
        candidate = Candidate(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            venue=paper.venue,
            citation_count=paper.citation_count,
            aliases={paper.paper_id},
            sources={action.source},
            raw_scores={score_key: weighted_score},
            snippets=[hit.snippet] if hit.snippet else [],
            metadata={
                "source_query": action.query,
                "paper_source": paper.source,
                "section_titles": self.section_titles_by_paper.get(paper.paper_id, []),
            },
        )
        return candidate


class DatabaseCorpus:
    backend_name = "database"

    def __init__(self) -> None:
        from packages.scholar_ingest.config import Settings
        from packages.scholar_infra.persistence.elasticsearch import ElasticsearchClient
        from packages.scholar_infra.persistence.mysql import MySQLClient
        from packages.scholar_infra.persistence.qdrant import QdrantClient

        self.settings = Settings.from_env()
        self.es = ElasticsearchClient(
            self.settings.elasticsearch_url,
            self.settings.elasticsearch_username,
            self.settings.elasticsearch_password,
        )
        self.qdrant = QdrantClient(self.settings.qdrant_url, self.settings.qdrant_api_key)
        self.mysql_cls = MySQLClient
        self._neo4j: Neo4jGraphClient | None = None
        self._neo4j_error: str | None = None
        self._dense_embedder: DenseEmbedder | None = None
        self._dense_error: str | None = None
        self._semantic_scholar: SemanticScholarClient | None = None
        self._academic_search_error: str | None = None
        self._dense_query_cache: dict[str, list[float]] = {}
        self._paper_cache: dict[str, dict] = {}
        self._chunk_cache: dict[str, dict] = {}
        self._check_connections()
        self._init_neo4j()
        self._init_dense_embedder()
        self._init_academic_search()

    def _check_connections(self) -> None:
        self.es.count(self.settings.papers_index)
        with self.mysql_cls.from_settings(self.settings) as mysql:
            mysql.use_database(self.settings.mysql_database)
            mysql.table_count("papers")

    def _init_neo4j(self) -> None:
        if not getattr(self.settings, "neo4j_retrieval_enabled", False):
            self._neo4j_error = "neo4j retrieval disabled"
            return
        try:
            client = Neo4jGraphClient.from_settings(self.settings)
            client.ping()
            self._neo4j = client
        except Exception as exc:
            self._neo4j = None
            self._neo4j_error = str(exc)

    def run_action(self, action: SearchAction) -> list[Candidate]:
        if action.source == "local_title_bm25":
            hits = self.es.search_papers(self.settings.papers_index, action.query, top_k=action.top_k)
            return [self._candidate_from_es_paper(hit, action) for hit in hits]
        if action.source == "local_chunk_bm25":
            hits = self.es.search_chunks(self.settings.chunks_index, action.query, top_k=action.top_k)
            return [self._candidate_from_es_chunk(hit, action) for hit in hits]
        if action.source == "local_tfidf":
            hits = self.qdrant.search_sparse(
                self.settings.qdrant_collection,
                action.query,
                vector_size=self.settings.qdrant_sparse_vector_size,
                sparse_vector_name=self.settings.qdrant_sparse_vector_name,
                top_k=action.top_k,
            )
            chunk_ids = [
                str((hit.get("payload") or {}).get("chunk_id") or "")
                for hit in hits
                if (hit.get("payload") or {}).get("chunk_id")
            ]
            chunks_by_id = self._chunks(chunk_ids)
            return [self._candidate_from_qdrant(hit, action, chunks_by_id=chunks_by_id) for hit in hits]
        if action.source == "qdrant_dense_paper":
            return self._search_qdrant_dense_papers(action)
        if action.source == "qdrant_sparse_paper":
            return self._search_qdrant_sparse_papers(action)
        if action.source == "neo4j_concept":
            return self._search_neo4j_concepts(action)
        if action.source == "neo4j_alias":
            return self._search_neo4j_aliases(action)
        if action.source == "semantic_scholar":
            return self._search_semantic_scholar(action)
        if action.source == "semantic_scholar_snippet":
            return self._search_semantic_scholar_snippets(action)
        return []

    @property
    def supported_sources(self) -> set[str]:
        sources = {"local_title_bm25", "local_chunk_bm25", "local_tfidf"}
        if self._dense_embedder is not None:
            sources.add("qdrant_dense_paper")
        if getattr(self.settings, "qdrant_sparse_paper_enabled", False):
            sources.add("qdrant_sparse_paper")
        if self._neo4j is not None:
            sources.update({"neo4j_concept", "neo4j_alias"})
        if self._semantic_scholar is not None:
            sources.add("semantic_scholar")
            if getattr(self.settings, "academic_search_snippet_enabled", True):
                sources.add("semantic_scholar_snippet")
        return sources

    def _init_academic_search(self) -> None:
        self._semantic_scholar, self._academic_search_error = _semantic_scholar_client_from_settings(
            self.settings
        )

    def _search_semantic_scholar(self, action: SearchAction) -> list[Candidate]:
        if self._semantic_scholar is None:
            return []
        try:
            return _semantic_scholar_candidates(self._semantic_scholar, action)
        except AcademicSearchError as exc:
            self._academic_search_error = str(exc)
            return []

    def _search_semantic_scholar_snippets(self, action: SearchAction) -> list[Candidate]:
        if self._semantic_scholar is None:
            return []
        try:
            return _semantic_scholar_snippet_candidates(self._semantic_scholar, action)
        except AcademicSearchError as exc:
            self._academic_search_error = str(exc)
            return []

    def _init_dense_embedder(self) -> None:
        if not getattr(self.settings, "qdrant_dense_paper_enabled", False):
            self._dense_error = "qdrant dense paper retrieval disabled"
            return
        model_name = getattr(self.settings, "dense_embedding_model", "")
        if not model_name:
            self._dense_error = "DENSE_EMBEDDING_MODEL is not configured"
            return
        try:
            self.qdrant.collection(self.settings.qdrant_dense_paper_collection)
            self._dense_embedder = build_dense_embedder(self.settings, model_name=model_name)
        except Exception as exc:
            self._dense_embedder = None
            self._dense_error = str(exc)

    def _search_qdrant_dense_papers(self, action: SearchAction) -> list[Candidate]:
        if self._dense_embedder is None:
            return []
        try:
            dense_query = dense_retrieval_query_text(action.query)
            cache_key = " ".join(dense_query.lower().split())
            vector = self._dense_query_cache.get(cache_key)
            if vector is None:
                vector = self._dense_embedder.encode_one(dense_query)
                self._dense_query_cache[cache_key] = vector
            hits = self.qdrant.search_dense(
                self.settings.qdrant_dense_paper_collection,
                vector,
                vector_name=self.settings.qdrant_dense_vector_name,
                top_k=action.top_k,
            )
        except Exception as exc:
            self._dense_error = str(exc)
            return []
        return [self._candidate_from_dense_paper(hit, action, dense_query=dense_query) for hit in hits]

    def _search_qdrant_sparse_papers(self, action: SearchAction) -> list[Candidate]:
        if not getattr(self.settings, "qdrant_sparse_paper_enabled", False):
            return []
        try:
            hits = self.qdrant.search_sparse(
                self.settings.qdrant_sparse_paper_collection,
                action.query,
                vector_size=self.settings.qdrant_sparse_vector_size,
                sparse_vector_name=self.settings.qdrant_sparse_vector_name,
                top_k=action.top_k,
            )
        except Exception:
            return []
        return [self._candidate_from_sparse_paper(hit, action) for hit in hits]

    def _search_neo4j_concepts(self, action: SearchAction) -> list[Candidate]:
        if self._neo4j is None:
            return []
        terms = _concept_search_terms(action.query)
        if not terms:
            return []
        try:
            hits = self._neo4j.search_concepts(
                terms,
                max_papers=action.top_k,
                min_concept_confidence=min(0.55, self.settings.neo4j_min_concept_confidence),
            )
        except Exception as exc:
            self._neo4j_error = str(exc)
            return []
        papers = self._papers([hit.paper_id for hit in hits])
        results: list[Candidate] = []
        for hit in hits:
            paper = papers.get(hit.paper_id)
            if paper is None:
                continue
            title = str(paper.get("title") or hit.paper_id)
            abstract = str(paper.get("abstract") or "")
            snippet = best_snippet(f"{title}. {abstract}", set(tokenize(action.query)))
            concept_snippet = f"Neo4j concepts: {', '.join(hit.concepts[:6])}" if hit.concepts else ""
            results.append(
                Candidate(
                    paper_id=hit.paper_id,
                    title=title,
                    abstract=abstract,
                    year=_safe_int(paper.get("year")),
                    venue=paper.get("venue"),
                    citation_count=_safe_int(paper.get("citation_count")),
                    aliases={hit.paper_id},
                    sources={action.source, "neo4j"},
                    raw_scores={action.source: float(hit.score) * action.weight},
                    snippets=[item for item in (snippet, concept_snippet) if item],
                    metadata={
                        "source_query": action.query,
                        "neo4j_concept_terms": terms[:12],
                        "graph_support": hit.support,
                        "graph_relations": list(hit.relations),
                        "graph_concepts": list(hit.concepts),
                    },
                )
            )
        return results

    def _search_neo4j_aliases(self, action: SearchAction) -> list[Candidate]:
        if self._neo4j is None:
            return []
        terms = _alias_search_terms(action.query)
        if not terms:
            return []
        try:
            hits = self._neo4j.search_aliases(terms, max_papers=action.top_k)
        except Exception as exc:
            self._neo4j_error = str(exc)
            return []
        papers = self._papers([hit.paper_id for hit in hits])
        results: list[Candidate] = []
        for hit in hits:
            paper = papers.get(hit.paper_id)
            if paper is None:
                continue
            title = str(paper.get("title") or hit.paper_id)
            abstract = str(paper.get("abstract") or "")
            snippet = best_snippet(f"{title}. {abstract}", set(tokenize(action.query)))
            matched_aliases = _matched_alias_terms(terms, hit.concepts)
            alias_snippet = f"Neo4j aliases: {', '.join(hit.concepts[:6])}" if hit.concepts else ""
            results.append(
                Candidate(
                    paper_id=hit.paper_id,
                    title=title,
                    abstract=abstract,
                    year=_safe_int(paper.get("year")),
                    venue=paper.get("venue"),
                    citation_count=_safe_int(paper.get("citation_count")),
                    aliases={hit.paper_id},
                    sources={action.source, "neo4j"},
                    raw_scores={action.source: float(hit.score) * action.weight},
                    snippets=[item for item in (snippet, alias_snippet) if item],
                    metadata={
                        "source_query": action.query,
                        "neo4j_alias_terms": terms[:12],
                        "graph_support": hit.support,
                        "graph_relations": list(hit.relations),
                        "graph_aliases": list(hit.concepts),
                        "alias_support": hit.support,
                        "alias_relations": list(hit.relations),
                        "alias_matched_terms": matched_aliases,
                        "alias_to_concept": any(relation == "concept_alias_fallback" for relation in hit.relations),
                    },
                )
            )
        return results

    def expand_graph_candidates(
        self,
        seed_candidates: list[Candidate],
        *,
        max_neighbors: int | None = None,
        min_concept_confidence: float | None = None,
    ) -> list[Candidate]:
        if self._neo4j is None:
            return []
        seed_ids = [candidate.paper_id for candidate in seed_candidates if candidate.paper_id]
        seed_ids = list(dict.fromkeys(seed_ids))[: max(1, self.settings.neo4j_max_seed_papers)]
        if not seed_ids:
            return []
        try:
            hits = self._neo4j.expand_papers(
                seed_ids,
                max_neighbors=max_neighbors or self.settings.neo4j_max_neighbors,
                min_concept_confidence=(
                    min_concept_confidence
                    if min_concept_confidence is not None
                    else self.settings.neo4j_min_concept_confidence
                ),
            )
        except Exception as exc:
            self._neo4j_error = str(exc)
            return []
        paper_ids = [hit.paper_id for hit in hits]
        papers = self._papers(paper_ids)
        results: list[Candidate] = []
        for hit in hits:
            paper = papers.get(hit.paper_id)
            if paper is None:
                continue
            candidate = Candidate(
                paper_id=hit.paper_id,
                title=str(paper.get("title") or hit.paper_id),
                abstract=str(paper.get("abstract") or ""),
                year=_safe_int(paper.get("year")),
                venue=paper.get("venue"),
                citation_count=_safe_int(paper.get("citation_count")),
                aliases={hit.paper_id},
                sources={"neo4j"},
                raw_scores={"neo4j_graph": float(hit.score)},
                snippets=[],
                metadata={
                    "graph_support": hit.support,
                    "graph_relations": list(hit.relations),
                    "graph_seed_ids": list(hit.seed_ids),
                    "graph_concepts": list(hit.concepts),
                },
            )
            results.append(candidate)
        return results

    def stats(self) -> dict:
        stats: dict[str, Any] = {"backend": self.backend_name}
        try:
            stats["es_papers"] = self.es.count(self.settings.papers_index)
            stats["es_chunks"] = self.es.count(self.settings.chunks_index)
        except Exception as exc:
            stats["es_error"] = str(exc)
        try:
            collection = self.qdrant.collection(self.settings.qdrant_collection)
            result = collection.get("result") or {}
            stats["qdrant_points"] = result.get("points_count")
            stats["qdrant_status"] = result.get("status")
        except Exception as exc:
            stats["qdrant_error"] = str(exc)
        if self._neo4j is not None:
            stats["neo4j_status"] = "connected"
        elif self._neo4j_error is not None:
            stats["neo4j_error"] = self._neo4j_error
        if self._dense_embedder is not None:
            stats["qdrant_dense_paper_status"] = "enabled"
            stats["qdrant_dense_paper_collection"] = self.settings.qdrant_dense_paper_collection
        elif self._dense_error is not None:
            stats["qdrant_dense_paper_error"] = self._dense_error
        if getattr(self.settings, "qdrant_sparse_paper_enabled", False):
            try:
                collection = self.qdrant.collection(self.settings.qdrant_sparse_paper_collection)
                result = collection.get("result") or {}
                stats["qdrant_sparse_paper_points"] = result.get("points_count")
                stats["qdrant_sparse_paper_status"] = result.get("status")
            except Exception as exc:
                stats["qdrant_sparse_paper_error"] = str(exc)
        if self._semantic_scholar is not None:
            stats["academic_search_status"] = "enabled"
            stats["academic_search_provider"] = "semantic_scholar"
            stats.update(self._semantic_scholar.stats())
        elif self._academic_search_error is not None:
            stats["academic_search_error"] = self._academic_search_error
        return stats

    def _candidate_from_es_paper(self, hit: dict, action: SearchAction) -> Candidate:
        paper_id = str(hit.get("paper_id") or hit.get("_id") or "")
        return Candidate(
            paper_id=paper_id,
            title=str(hit.get("title") or ""),
            abstract=str(hit.get("abstract") or ""),
            year=_safe_int(hit.get("year")),
            venue=hit.get("venue"),
            citation_count=_safe_int(hit.get("citation_count")),
            aliases={paper_id},
            sources={action.source, "elasticsearch"},
            raw_scores={action.source: float(hit.get("_score") or 0.0) * action.weight},
            snippets=[best_snippet(f"{hit.get('title') or ''}. {hit.get('abstract') or ''}", set(tokenize(action.query)))],
            metadata={"source_query": action.query, "index": hit.get("_index")},
        )

    def _candidate_from_es_chunk(self, hit: dict, action: SearchAction) -> Candidate:
        paper_id = str(hit.get("paper_id") or "")
        paper = self._paper(paper_id)
        title = str((paper or {}).get("title") or paper_id)
        abstract = str((paper or {}).get("abstract") or "")
        snippet = best_snippet(str(hit.get("text") or ""), set(tokenize(action.query)))
        return Candidate(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            year=_safe_int((paper or {}).get("year")),
            venue=(paper or {}).get("venue"),
            citation_count=_safe_int((paper or {}).get("citation_count")),
            aliases={paper_id, str(hit.get("chunk_id") or "")},
            sources={action.source, "elasticsearch"},
            raw_scores={action.source: float(hit.get("_score") or 0.0) * action.weight},
            snippets=[snippet] if snippet else [],
            metadata={
                "source_query": action.query,
                "chunk_id": hit.get("chunk_id"),
                "section_title": hit.get("section_title"),
                "index": hit.get("_index"),
            },
        )

    def _candidate_from_qdrant(
        self,
        hit: dict,
        action: SearchAction,
        *,
        chunks_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> Candidate:
        payload = hit.get("payload") or {}
        paper_id = str(payload.get("paper_id") or "")
        chunk_id = str(payload.get("chunk_id") or "")
        chunk = (chunks_by_id or {}).get(chunk_id) or {}
        paper = self._paper(paper_id)
        title = str((paper or {}).get("title") or paper_id)
        abstract = str((paper or {}).get("abstract") or "")
        chunk_text = str(chunk.get("text") or "")
        snippet = best_snippet(chunk_text, set(tokenize(action.query)))
        section_title = payload.get("section_title") or chunk.get("section_title")
        qdrant_score = float(hit.get("score") or 0.0)
        weighted_score = log(1.0 + max(0.0, qdrant_score)) * action.weight
        return Candidate(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            year=_safe_int((paper or {}).get("year")),
            venue=(paper or {}).get("venue"),
            citation_count=_safe_int((paper or {}).get("citation_count")),
            aliases={paper_id, chunk_id},
            sources={action.source, "qdrant"},
            raw_scores={action.source: weighted_score},
            snippets=[snippet] if snippet else [],
            metadata={
                "source_query": action.query,
                "chunk_id": chunk_id,
                "chunk_type": payload.get("chunk_type") or chunk.get("chunk_type"),
                "section_title": section_title,
                "point_id": hit.get("id"),
                "qdrant_score": qdrant_score,
            },
        )

    def _candidate_from_dense_paper(
        self,
        hit: dict,
        action: SearchAction,
        *,
        dense_query: str | None = None,
    ) -> Candidate:
        payload = hit.get("payload") or {}
        paper_id = str(payload.get("paper_id") or hit.get("id") or "")
        paper = self._paper(paper_id)
        title = str(payload.get("title") or (paper or {}).get("title") or paper_id)
        abstract = str((paper or {}).get("abstract") or payload.get("abstract") or "")
        qdrant_score = float(hit.get("score") or 0.0)
        weighted_score = max(0.0, qdrant_score) * action.weight
        return Candidate(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            year=_safe_int(payload.get("year") or (paper or {}).get("year")),
            venue=payload.get("venue") or (paper or {}).get("venue"),
            citation_count=_safe_int((paper or {}).get("citation_count")),
            aliases={paper_id},
            sources={action.source, "qdrant", "qdrant_dense"},
            raw_scores={action.source: weighted_score},
            snippets=[best_snippet(f"{title}. {abstract}", set(tokenize(action.query)))],
            metadata={
                "source_query": action.query,
                "dense_query": dense_query or action.query,
                "text_type": payload.get("text_type") or "title_abs",
                "point_id": hit.get("id"),
                "qdrant_score": qdrant_score,
                "qdrant_collection": self.settings.qdrant_dense_paper_collection,
                "dense_used": True,
            },
        )

    def _candidate_from_sparse_paper(self, hit: dict, action: SearchAction) -> Candidate:
        payload = hit.get("payload") or {}
        paper_id = str(payload.get("paper_id") or hit.get("id") or "")
        paper = self._paper(paper_id)
        title = str(payload.get("title") or (paper or {}).get("title") or paper_id)
        abstract = str((paper or {}).get("abstract") or "")
        qdrant_score = float(hit.get("score") or 0.0)
        weighted_score = log(1.0 + max(0.0, qdrant_score)) * action.weight
        return Candidate(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            year=_safe_int(payload.get("year") or (paper or {}).get("year")),
            venue=payload.get("venue") or (paper or {}).get("venue"),
            citation_count=_safe_int((paper or {}).get("citation_count")),
            aliases={paper_id},
            sources={action.source, "qdrant", "qdrant_sparse_paper"},
            raw_scores={action.source: weighted_score},
            snippets=[best_snippet(f"{title}. {abstract}", set(tokenize(action.query)))],
            metadata={
                "source_query": action.query,
                "text_type": payload.get("text_type") or "title_abs_sparse",
                "point_id": hit.get("id"),
                "qdrant_score": qdrant_score,
                "qdrant_collection": self.settings.qdrant_sparse_paper_collection,
                "sparse_paper_used": True,
            },
        )

    def _chunks(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        missing = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id and chunk_id not in self._chunk_cache]
        if missing:
            source_fields = ["chunk_id", "paper_id", "chunk_type", "section_title", "text"]
            body = {"docs": [{"_id": chunk_id, "_source": source_fields} for chunk_id in missing]}
            result = self.es.request("POST", f"/{self.settings.chunks_index}/_mget", body)
            for doc in (result or {}).get("docs") or []:
                if not doc.get("found"):
                    continue
                source = doc.get("_source") or {}
                chunk_id = str(source.get("chunk_id") or doc.get("_id") or "")
                if chunk_id:
                    self._chunk_cache[chunk_id] = source
        return {chunk_id: self._chunk_cache[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunk_cache}

    def _paper(self, paper_id: str) -> dict | None:
        if not paper_id:
            return None
        if paper_id in self._paper_cache:
            return self._paper_cache[paper_id]
        with self.mysql_cls.from_settings(self.settings) as mysql:
            mysql.use_database(self.settings.mysql_database)
            paper = mysql.fetch_paper(paper_id)
        if paper is not None:
            self._paper_cache[paper_id] = paper
        return paper

    def _papers(self, paper_ids: list[str]) -> dict[str, dict]:
        missing = [paper_id for paper_id in dict.fromkeys(paper_ids) if paper_id and paper_id not in self._paper_cache]
        if missing:
            with self.mysql_cls.from_settings(self.settings) as mysql:
                mysql.use_database(self.settings.mysql_database)
                fetched = mysql.fetch_papers(missing)
            self._paper_cache.update(fetched)
        return {paper_id: self._paper_cache[paper_id] for paper_id in paper_ids if paper_id in self._paper_cache}


def _safe_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _external_aliases(external_ids: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for key, value in external_ids.items():
        if value is None or value == "":
            continue
        normalized_key = str(key).lower()
        if normalized_key == "arxiv":
            aliases.add(f"arxiv:{value}")
        elif normalized_key == "doi":
            aliases.add(f"doi:{str(value).lower()}")
        elif normalized_key == "corpusid":
            aliases.add(f"s2-corpus:{value}")
        else:
            aliases.add(f"{normalized_key}:{value}")
    return aliases


_CONCEPT_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{1,}", re.IGNORECASE)
_CONCEPT_SPLIT_RE = re.compile(r"[|;,/]+")
_CONCEPT_STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "against",
    "also",
    "and",
    "any",
    "are",
    "based",
    "been",
    "being",
    "better",
    "between",
    "bigger",
    "but",
    "can",
    "cite",
    "compare",
    "different",
    "find",
    "for",
    "from",
    "give",
    "have",
    "help",
    "how",
    "large",
    "me",
    "more",
    "most",
    "paper",
    "papers",
    "provide",
    "research",
    "result",
    "results",
    "show",
    "simple",
    "smaller",
    "studies",
    "study",
    "such",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "using",
    "various",
    "were",
    "what",
    "where",
    "which",
    "with",
    "work",
    "works",
}
_CONCEPT_SINGLETONS = {
    "bert",
    "clip",
    "cot",
    "dpo",
    "gpt",
    "humaneval",
    "imo",
    "llm",
    "llms",
    "mbpp",
    "mdp",
    "nerf",
    "slam",
    "vlm",
    "vlms",
}
_CONCEPT_HINTS = {
    "agent",
    "agents",
    "animation",
    "benchmark",
    "bound",
    "bounds",
    "classification",
    "concept",
    "certification",
    "certified",
    "computer",
    "control",
    "controlled",
    "corpus",
    "data",
    "dataset",
    "datasets",
    "detection",
    "diffusion",
    "evaluation",
    "feature",
    "features",
    "generation",
    "game",
    "gameplay",
    "graph",
    "image",
    "language",
    "learning",
    "literature",
    "matching",
    "math",
    "mathematical",
    "model",
    "models",
    "multilingual",
    "navigation",
    "olympiad",
    "pretraining",
    "prediction",
    "proving",
    "ranking",
    "reasoning",
    "response",
    "retrieval",
    "robustness",
    "review",
    "search",
    "segmentation",
    "summarization",
    "survey",
    "synthesis",
    "state",
    "text",
    "theorem",
    "translation",
    "vision",
}


def _concept_search_terms(query: str) -> list[str]:
    weighted = analyze_weighted_query(query)
    terms: list[str] = [phrase.term for phrase in weighted.phrases]
    terms.extend(term.term for term in weighted.terms if term.term in _CONCEPT_SINGLETONS)
    for part in _CONCEPT_SPLIT_RE.split(query.lower()):
        tokens = [
            token.replace("-", " ")
            for token in _CONCEPT_TOKEN_RE.findall(part)
            if token not in _CONCEPT_STOPWORDS and (len(token) > 2 or token in _CONCEPT_SINGLETONS)
        ]
        expanded_tokens: list[str] = []
        for token in tokens:
            expanded_tokens.extend(token.split())
        tokens = [token for token in expanded_tokens if token and token not in _CONCEPT_STOPWORDS]
        if not tokens:
            continue
        if 1 < len(tokens) <= 5 and _useful_concept_tokens(tokens):
            terms.append(" ".join(tokens))
        for size in (4, 3, 2):
            if len(tokens) < size:
                continue
            for index in range(0, len(tokens) - size + 1):
                ngram = tokens[index : index + size]
                if _useful_concept_tokens(ngram):
                    terms.append(" ".join(ngram))
        for token in tokens:
            if token in _CONCEPT_SINGLETONS:
                terms.append(token)
    return _unique_terms(terms)[:40]


def _alias_search_terms(query: str) -> list[str]:
    terms = _concept_search_terms(query)
    weighted = analyze_weighted_query(query)
    terms.extend(phrase.term for phrase in weighted.phrases)
    for part in _CONCEPT_SPLIT_RE.split(query.lower()):
        tokens = [
            token.replace("-", " ")
            for token in _CONCEPT_TOKEN_RE.findall(part)
            if token not in _CONCEPT_STOPWORDS and (len(token) > 2 or token in _CONCEPT_SINGLETONS)
        ]
        expanded_tokens: list[str] = []
        for token in tokens:
            expanded_tokens.extend(token.split())
        tokens = [token for token in expanded_tokens if token and token not in _CONCEPT_STOPWORDS]
        if not tokens:
            continue
        for size in (5, 4, 3, 2):
            if len(tokens) < size:
                continue
            for index in range(0, len(tokens) - size + 1):
                ngram = tokens[index : index + size]
                if _useful_alias_tokens(ngram):
                    terms.append(" ".join(ngram))
    return _unique_terms(terms)[:60]


def _useful_alias_tokens(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return any(token in _CONCEPT_SINGLETONS for token in tokens)
    if any(token in _CONCEPT_SINGLETONS for token in tokens):
        return True
    return any(token in _CONCEPT_HINTS for token in tokens) or any(
        token
        in {
            "adapter",
            "benchmark",
            "classification",
            "dataset",
            "deblurring",
            "detectgpt",
            "navigation",
            "prompting",
            "pruning",
            "reranking",
            "segmentation",
            "token",
            "tokens",
        }
        for token in tokens
    )


def _useful_concept_tokens(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return any(token in _CONCEPT_SINGLETONS for token in tokens)
    return any(token in _CONCEPT_HINTS or token in _CONCEPT_SINGLETONS for token in tokens)


def _unique_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = " ".join(value.split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _matched_alias_terms(query_terms: list[str], aliases: tuple[str, ...]) -> list[str]:
    normalized_aliases = {_normalize_alias_text(alias) for alias in aliases if alias}
    matched: list[str] = []
    for term in query_terms:
        normalized = _normalize_alias_text(term)
        if not normalized:
            continue
        if normalized in normalized_aliases:
            matched.append(term)
            continue
        if any(normalized and normalized in alias for alias in normalized_aliases):
            matched.append(term)
    return _unique_terms(matched)[:12]


def _normalize_alias_text(value: str) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").split())


def _load_section_titles_by_paper(
    processed_dir: Path,
    known_paper_ids: set[str],
    *,
    limit: int | None,
    max_chunks_per_paper: int,
) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = {}
    for index, row in enumerate(read_jsonl(processed_dir / "paper_chunks.jsonl")):
        if limit is not None and index >= limit:
            break
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in known_paper_ids:
            continue
        count = counts.get(paper_id, 0)
        if count >= max_chunks_per_paper:
            continue
        counts[paper_id] = count + 1
        section_title = str(row.get("section_title") or "").strip()
        if section_title and section_title not in sections[paper_id]:
            sections[paper_id].append(section_title)
    return sections
