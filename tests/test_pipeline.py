from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from candidate_normalization.normalizer import CandidateNormalizer
from cost_control_cache.pipeline import SearchPipeline
from offline_evaluation.evaluation import score_prediction
from query_understanding_decomposition.query import QueryParser
from scholar_common.models import Candidate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def make_processed_dir(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    write_jsonl(
        processed / "papers.jsonl",
        [
            {
                "paper_id": "arxiv:1111.00001",
                "title": "Universal Image Text Representation Learning for Image Retrieval",
                "abstract": "A model for cross-modal image retrieval and visual search.",
                "year": 2020,
                "source": "pasa",
            },
            {
                "paper_id": "arxiv:2222.00002",
                "title": "A Survey of Database Query Optimizers",
                "abstract": "This survey discusses relational query processing.",
                "year": 2018,
                "source": "pasa",
            },
        ],
    )
    write_jsonl(
        processed / "paper_chunks.jsonl",
        [
            {
                "chunk_id": "arxiv:1111.00001#chunk:0",
                "paper_id": "arxiv:1111.00001",
                "text": "Title: Universal Image Text Representation Learning for Image Retrieval\nAbstract: cross-modal retrieval.",
            },
            {
                "chunk_id": "arxiv:2222.00002#chunk:0",
                "paper_id": "arxiv:2222.00002",
                "text": "Title: A Survey of Database Query Optimizers\nAbstract: relational query plans.",
            },
        ],
    )
    write_jsonl(
        processed / "queries.jsonl",
        [
            {
                "qid": "q1",
                "query_text": "What works are related to the field of image retrieval?",
                "split_name": "dev",
                "dataset_name": "mini",
            }
        ],
    )
    write_jsonl(
        processed / "eval_sets.jsonl",
        [
            {
                "qid": "q1",
                "gold_paper_ids": ["arxiv:1111.00001"],
                "split_name": "dev",
                "dataset_name": "mini",
            }
        ],
    )
    return processed


def test_query_parser_generates_constraints_and_subqueries():
    parsed = QueryParser().parse("Find image retrieval papers after 2020")
    assert "image retrieval" in parsed.must_have_constraints
    assert parsed.time_range == (2020, None)
    assert parsed.sub_queries


def test_candidate_normalizer_merges_arxiv_aliases():
    candidates = [
        Candidate("arxiv:2301.00001", "Paper A", sources={"local_title_bm25"}, raw_scores={"a": 1.0}),
        Candidate("arXiv:2301.00001", "Paper A", sources={"local_tfidf"}, raw_scores={"b": 2.0}),
    ]
    merged = CandidateNormalizer().merge(candidates)
    assert len(merged) == 1
    assert merged[0].sources == {"local_title_bm25", "local_tfidf"}
    assert merged[0].raw_scores["b"] == 2.0


def test_score_prediction_computes_basic_metrics():
    metrics = score_prediction(["a", "b", "c"], ["b", "d"], k=3)
    assert metrics.hits == 1
    assert round(metrics.recall_at_k, 3) == 0.5
    assert round(metrics.mrr, 3) == 0.5


def test_pipeline_returns_ranked_result(tmp_path):
    processed = make_processed_dir(tmp_path)
    pipeline = SearchPipeline(processed, per_query_top_k=5, backend="jsonl")
    response = pipeline.search("image retrieval representation learning", top_k=2)
    assert response.papers
    assert response.papers[0].paper_id == "arxiv:1111.00001"
    assert response.plan.expand_citations_for == ["arxiv:1111.00001"]
    assert response.cost["citation_expansion_seeds"][0]["paper_id"] == "arxiv:1111.00001"
    assert response.coverage.coverage
