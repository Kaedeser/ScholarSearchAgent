# 中文功能说明：多源召回模块，提供本地 JSONL 检索和真实数据库检索后端适配。

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path

from packages.scholar_infra.io.jsonl import ensure_paths_exist, load_chunks_by_paper, load_papers, read_jsonl
from packages.scholar_core.models import Candidate, Paper, SearchAction
from packages.scholar_core.retrieval.ports import CorpusBackend
from packages.scholar_core.text import best_snippet, cosine_sparse, token_counter, tokenize

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
        query_terms = tokenize(query)
        if not query_terms:
            return []
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
                score += idf * (tf * (self.k1 + 1)) / denom
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
        counts = token_counter(query)
        vector: Counter[str] = Counter()
        for token, value in counts.items():
            vector[token] = value * log(1 + self.doc_count / (1 + self.doc_freq.get(token, 0)))
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

    def _paper_text(self, paper: Paper) -> str:
        return f"{paper.title}\n{paper.abstract or ''}"

    def _semantic_text(self, paper_id: str, paper: Paper) -> str:
        chunks = " ".join(self.chunks_by_paper.get(paper_id, [])[:2])
        return f"{paper.title}\n{paper.abstract or ''}\n{chunks}"

    def run_action(self, action: SearchAction) -> list[Candidate]:
        if action.source == "local_title_bm25":
            hits = self.title_bm25.search(action.query, top_k=action.top_k)
        elif action.source == "local_chunk_bm25":
            hits = self.chunk_bm25.search(action.query, top_k=action.top_k)
        elif action.source == "local_tfidf":
            hits = self.tfidf.search(action.query, top_k=action.top_k)
        else:
            return []
        return [self._candidate_from_hit(hit, action) for hit in hits]

    def stats(self) -> dict:
        return {
            "backend": self.backend_name,
            "papers_loaded": len(self.papers),
            "chunks_loaded": sum(len(chunks) for chunks in self.chunks_by_paper.values()),
        }

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
        self._paper_cache: dict[str, dict] = {}
        self._check_connections()

    def _check_connections(self) -> None:
        self.es.count(self.settings.papers_index)
        with self.mysql_cls.from_settings(self.settings) as mysql:
            mysql.use_database(self.settings.mysql_database)
            mysql.table_count("papers")

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
            return [self._candidate_from_qdrant(hit, action) for hit in hits]
        return []

    def stats(self) -> dict:
        stats = {"backend": self.backend_name}
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

    def _candidate_from_qdrant(self, hit: dict, action: SearchAction) -> Candidate:
        payload = hit.get("payload") or {}
        paper_id = str(payload.get("paper_id") or "")
        paper = self._paper(paper_id)
        title = str((paper or {}).get("title") or paper_id)
        abstract = str((paper or {}).get("abstract") or "")
        return Candidate(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            year=_safe_int((paper or {}).get("year")),
            venue=(paper or {}).get("venue"),
            citation_count=_safe_int((paper or {}).get("citation_count")),
            aliases={paper_id, str(payload.get("chunk_id") or "")},
            sources={action.source, "qdrant"},
            raw_scores={action.source: float(hit.get("score") or 0.0) * action.weight},
            snippets=[],
            metadata={
                "source_query": action.query,
                "chunk_id": payload.get("chunk_id"),
                "section_title": payload.get("section_title"),
                "point_id": hit.get("id"),
            },
        )

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


def _safe_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
