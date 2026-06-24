from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ids import estimate_tokens, normalize_arxiv_id, paper_id_from_arxiv, slugify_title, title_hash
from .io_utils import load_json, read_jsonl, write_jsonl


QUERY_FILES = (
    ("AutoScholarQuery", "train", "AutoScholarQuery/train.jsonl"),
    ("AutoScholarQuery", "dev", "AutoScholarQuery/dev.jsonl"),
    ("AutoScholarQuery", "test", "AutoScholarQuery/test.jsonl"),
    ("RealScholarQuery", "test", "RealScholarQuery/test.jsonl"),
)


@dataclass(frozen=True)
class ConversionStats:
    queries: int = 0
    gold_labels: int = 0
    eval_sets: int = 0
    papers: int = 0
    paper_chunks: int = 0
    papers_with_zip_doc: int = 0


def _published_date(source_meta: dict[str, Any] | None) -> str | None:
    value = (source_meta or {}).get("published_time")
    if value is None or value == "":
        return None
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def convert_queries(pasa_root: Path, processed_dir: Path, limit: int | None = None) -> ConversionStats:
    query_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for dataset_name, split_name, rel_path in QUERY_FILES:
        path = pasa_root / rel_path
        if not path.exists():
            continue
        seen = 0
        for obj in read_jsonl(path):
            if limit is not None and seen >= limit:
                break
            seen += 1
            qid = obj.get("qid") or f"{dataset_name}_{split_name}_{seen - 1}"
            answers = _as_list(obj.get("answer"))
            answer_arxiv_ids = [normalize_arxiv_id(x) for x in _as_list(obj.get("answer_arxiv_id"))]
            answer_titles = [str(x) for x in answers]
            published_time = _published_date(obj.get("source_meta"))
            gold_paper_ids = [paper_id_from_arxiv(x) for x in answer_arxiv_ids if x]
            query_rows.append(
                {
                    "qid": qid,
                    "dataset_name": dataset_name,
                    "split_name": split_name,
                    "query_text": obj.get("question", ""),
                    "published_time": published_time,
                    "answer_count": len(gold_paper_ids),
                    "source_path": rel_path,
                }
            )
            for rank, paper_id in enumerate(gold_paper_ids, start=1):
                title = answer_titles[rank - 1] if rank - 1 < len(answer_titles) else None
                gold_rows.append(
                    {
                        "qid": qid,
                        "paper_id": paper_id,
                        "arxiv_id": paper_id.replace("arxiv:", "", 1),
                        "title": title,
                        "label_rank": rank,
                        "source": dataset_name,
                    }
                )
            eval_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split_name": split_name,
                    "qid": qid,
                    "gold_paper_ids": gold_paper_ids,
                    "published_time": published_time,
                }
            )

    write_jsonl(processed_dir / "queries.jsonl", query_rows)
    write_jsonl(processed_dir / "gold_labels.jsonl", gold_rows)
    write_jsonl(processed_dir / "eval_sets.jsonl", eval_rows)
    return ConversionStats(queries=len(query_rows), gold_labels=len(gold_rows), eval_sets=len(eval_rows))


def _load_zip_doc(zip_file: zipfile.ZipFile, fulltext_key: str | None) -> dict[str, Any] | None:
    if not fulltext_key or fulltext_key not in zip_file.NameToInfo:
        return None
    with zip_file.open(fulltext_key) as handle:
        return json.loads(handle.read().decode("utf-8"))


def _paper_rows_from_doc(arxiv_id: str, title: str, doc: dict[str, Any] | None, fulltext_key: str | None) -> dict[str, Any]:
    abstract = None
    if doc:
        abstract = doc.get("abstract")
    return {
        "paper_id": paper_id_from_arxiv(arxiv_id) or title_hash(title),
        "arxiv_id": normalize_arxiv_id(arxiv_id),
        "title": title,
        "abstract": abstract,
        "year": _year_from_arxiv(arxiv_id),
        "published_time": None,
        "venue": None,
        "authors": [],
        "citation_count": None,
        "source": "pasa",
        "fulltext_key": fulltext_key,
        "has_fulltext": bool(doc),
    }


def _year_from_arxiv(arxiv_id: str) -> int | None:
    normalized = normalize_arxiv_id(arxiv_id)
    if not normalized or len(normalized) < 2:
        return None
    prefix = normalized.split(".", 1)[0]
    if len(prefix) >= 2 and prefix[:2].isdigit():
        yy = int(prefix[:2])
        return 2000 + yy if yy < 90 else 1900 + yy
    return None


def _chunks_for_paper(paper: dict[str, Any], doc: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
    paper_id = paper["paper_id"]
    idx = 0
    abstract = paper.get("abstract")
    if abstract:
        text = f"Title: {paper['title']}\nAbstract: {abstract}"
    else:
        text = f"Title: {paper['title']}"
    yield {
        "chunk_id": f"{paper_id}#chunk:{idx}",
        "paper_id": paper_id,
        "chunk_index": idx,
        "chunk_type": "title_abstract",
        "section_title": None,
        "text": text,
        "token_estimate": estimate_tokens(text),
        "source": "pasa",
    }
    idx += 1
    sections = (doc or {}).get("sections") or {}
    if isinstance(sections, dict):
        for section_title, references in sections.items():
            if not references:
                continue
            if isinstance(references, list):
                body = "; ".join(str(x) for x in references[:80])
            else:
                body = str(references)
            text = f"Title: {paper['title']}\nSection: {section_title}\nReferenced papers: {body}"
            yield {
                "chunk_id": f"{paper_id}#chunk:{idx}",
                "paper_id": paper_id,
                "chunk_index": idx,
                "chunk_type": "section_references",
                "section_title": str(section_title),
                "text": text,
                "token_estimate": estimate_tokens(text),
                "source": "pasa",
            }
            idx += 1


def convert_papers(pasa_root: Path, processed_dir: Path, limit: int | None = None) -> ConversionStats:
    id2paper_path = pasa_root / "paper_database" / "id2paper.json"
    zip_path = pasa_root / "paper_database" / "cs_paper_2nd.zip"
    id2paper: dict[str, str] = load_json(id2paper_path)

    paper_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    papers_with_doc = 0

    with zipfile.ZipFile(zip_path) as zip_file:
        for index, (arxiv_id, title) in enumerate(id2paper.items()):
            if limit is not None and index >= limit:
                break
            key = slugify_title(title)
            doc = _load_zip_doc(zip_file, key)
            if doc:
                papers_with_doc += 1
            paper = _paper_rows_from_doc(arxiv_id, title, doc, key if doc else None)
            paper_rows.append(paper)
            chunk_rows.extend(_chunks_for_paper(paper, doc))

    write_jsonl(processed_dir / "papers.jsonl", paper_rows)
    write_jsonl(processed_dir / "paper_chunks.jsonl", chunk_rows)
    return ConversionStats(papers=len(paper_rows), paper_chunks=len(chunk_rows), papers_with_zip_doc=papers_with_doc)

