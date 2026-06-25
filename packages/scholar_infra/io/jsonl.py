# 中文功能说明：共享文件 IO 工具，负责 JSONL 读写、默认数据目录和本地处理数据加载。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from packages.scholar_core.models import Paper


def default_processed_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data_ingestion_indexing" / "data_processed"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_papers(processed_dir: Path, limit: int | None = None) -> dict[str, Paper]:
    papers: dict[str, Paper] = {}
    for index, row in enumerate(read_jsonl(processed_dir / "papers.jsonl")):
        if limit is not None and index >= limit:
            break
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            continue
        papers[paper_id] = Paper(
            paper_id=paper_id,
            title=str(row.get("title") or ""),
            abstract=str(row.get("abstract") or ""),
            year=row.get("year"),
            venue=row.get("venue"),
            citation_count=row.get("citation_count"),
            source=str(row.get("source") or "pasa"),
            metadata=row,
        )
    return papers


def load_chunks_by_paper(
    processed_dir: Path,
    known_paper_ids: set[str],
    *,
    limit: int | None = None,
    max_chunks_per_paper: int = 4,
) -> dict[str, list[str]]:
    chunks: dict[str, list[str]] = {}
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
        text = str(row.get("text") or "")
        if text:
            chunks.setdefault(paper_id, []).append(text)
            counts[paper_id] = count + 1
    return chunks


def load_queries(processed_dir: Path, split: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(processed_dir / "queries.jsonl"):
        if split and row.get("split_name") != split:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_eval_sets(processed_dir: Path, split: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(processed_dir / "eval_sets.jsonl"):
        if split and row.get("split_name") != split:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required demo data files: " + ", ".join(missing))
