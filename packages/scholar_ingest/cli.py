# 中文功能说明：数据导入命令行工具，提供转换、建库、导入、检索和校验命令。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .es import CHUNKS_MAPPING, PAPERS_MAPPING, ElasticsearchClient
from .mysql import MySQLClient, sql_value
from .io_utils import dump_json
from .pasa import convert_papers, convert_queries
from .qdrant import QdrantClient, qdrant_point_from_chunk, qdrant_point_from_dense_paper, qdrant_point_from_sparse_paper
from .io_utils import read_jsonl
from packages.scholar_infra.embeddings import build_dense_embedder


def print_json(data: object) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", "replace").decode("utf-8", "replace").encode("ascii", "backslashreplace").decode("ascii"))


def cmd_doctor(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    report = {
        "module_root": str(settings.module_root),
        "workspace_root": str(settings.workspace_root),
        "config_path": str(settings.config_path),
        "config_exists": settings.config_path.exists(),
        "pasa_data_root": str(settings.pasa_data_root),
        "pasa_data_exists": settings.pasa_data_root.exists(),
        "processed_dir": str(settings.processed_dir),
        "mysql_host": settings.mysql_host,
        "mysql_port": settings.mysql_port,
        "mysql_database": settings.mysql_database,
        "mysql_username": settings.mysql_username,
        "mysql_password_set": bool(settings.mysql_password),
        "elasticsearch_url": settings.elasticsearch_url,
        "elasticsearch_username": settings.elasticsearch_username,
        "papers_index": settings.papers_index,
        "chunks_index": settings.chunks_index,
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_api_key_set": bool(settings.qdrant_api_key),
        "qdrant_sparse_vector_name": settings.qdrant_sparse_vector_name,
        "qdrant_sparse_vector_size": settings.qdrant_sparse_vector_size,
    }
    if args.check_mysql:
        with MySQLClient.from_settings(settings) as client:
            report["mysql_version"] = client.scalar("SELECT VERSION()")
            client.execute(
                f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
            )
            client.use_database(settings.mysql_database)
            report["mysql_database"] = settings.mysql_database
    if args.check_qdrant:
        client = QdrantClient(settings.qdrant_url, settings.qdrant_api_key)
        report["qdrant_health"] = client.health()
        report["qdrant_collections"] = client.collections()
    if args.check_es:
        client = ElasticsearchClient(
            settings.elasticsearch_url,
            settings.elasticsearch_username,
            settings.elasticsearch_password,
        )
        report["elasticsearch_health"] = client.health()
    print_json(report)


def cmd_convert_pasa(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    processed_dir = Path(args.output_dir) if args.output_dir else settings.processed_dir
    query_stats = convert_queries(settings.pasa_data_root, processed_dir, args.limit)
    paper_stats = convert_papers(settings.pasa_data_root, processed_dir, args.limit)
    stats = {
        "processed_dir": str(processed_dir),
        "queries": query_stats.queries,
        "gold_labels": query_stats.gold_labels,
        "eval_sets": query_stats.eval_sets,
        "papers": paper_stats.papers,
        "paper_chunks": paper_stats.paper_chunks,
        "papers_with_zip_doc": paper_stats.papers_with_zip_doc,
    }
    dump_json(processed_dir / "conversion_stats.json", stats)
    print_json(stats)


def cmd_init_es(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = ElasticsearchClient(
        settings.elasticsearch_url,
        settings.elasticsearch_username,
        settings.elasticsearch_password,
    )
    result = client.init_indices(args.papers_index, args.chunks_index, reset=args.reset)
    result["papers_index"] = args.papers_index
    result["chunks_index"] = args.chunks_index
    result["reset"] = args.reset
    print_json(result)


def cmd_load_es(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = ElasticsearchClient(
        settings.elasticsearch_url,
        settings.elasticsearch_username,
        settings.elasticsearch_password,
    )
    processed_dir = Path(args.processed_dir) if args.processed_dir else settings.processed_dir
    if args.kind == "papers":
        count = client.bulk_jsonl(args.index, processed_dir / "papers.jsonl", "paper_id")
    else:
        count = client.bulk_jsonl(args.index, processed_dir / "paper_chunks.jsonl", "chunk_id")
    print_json({"kind": args.kind, "index": args.index, "count": count})


def cmd_init_qdrant(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = QdrantClient(settings.qdrant_url, settings.qdrant_api_key)
    collection = args.collection or settings.qdrant_collection
    if args.sparse:
        result = client.init_sparse_collection(
            collection,
            sparse_vector_name=args.sparse_vector_name or settings.qdrant_sparse_vector_name,
            reset=args.reset,
        )
    else:
        if args.reset:
            client.delete_collection(collection)
        result = client.create_collection(collection, args.vector_size, args.distance)
    print_json({"collection": collection, "reset": args.reset, "result": result})


def cmd_init_mysql(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    schema_path = Path(args.schema) if args.schema else settings.module_root / "sql" / "schema.sql"
    with MySQLClient.from_settings(settings) as client:
        result = client.init_schema(schema_path, settings.mysql_database, reset=args.reset)
    print_json(result)


def cmd_load_mysql(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    processed_dir = Path(args.processed_dir) if args.processed_dir else settings.processed_dir
    stats: dict[str, int] = {}
    with MySQLClient.from_settings(settings) as client:
        client.use_database(settings.mysql_database)
        stats["queries"] = client.load_jsonl(
            "queries",
            processed_dir / "queries.jsonl",
            ["qid", "dataset_name", "split_name", "query_text", "published_time", "answer_count", "source_path"],
            ["dataset_name", "split_name", "query_text", "published_time", "answer_count", "source_path"],
            batch_size=args.batch_size,
            limit=args.limit,
        )
        stats["papers"] = client.load_jsonl(
            "papers",
            processed_dir / "papers.jsonl",
            [
                "paper_id",
                "arxiv_id",
                "title",
                "abstract",
                "year",
                "published_time",
                "venue",
                "authors",
                "citation_count",
                "source",
                "fulltext_key",
                "has_fulltext",
            ],
            [
                "arxiv_id",
                "title",
                "abstract",
                "year",
                "published_time",
                "venue",
                "authors",
                "citation_count",
                "source",
                "fulltext_key",
                "has_fulltext",
            ],
            batch_size=args.paper_batch_size,
            limit=args.limit,
        )
        stats["paper_chunks"] = client.load_jsonl(
            "paper_chunks",
            processed_dir / "paper_chunks.jsonl",
            ["chunk_id", "paper_id", "chunk_index", "chunk_type", "section_title", "text", "token_estimate", "source"],
            ["paper_id", "chunk_index", "chunk_type", "section_title", "text", "token_estimate", "source"],
            batch_size=args.chunk_batch_size,
            limit=args.limit,
        )
        stats["gold_labels"] = client.load_jsonl(
            "gold_labels",
            processed_dir / "gold_labels.jsonl",
            ["qid", "paper_id", "arxiv_id", "title", "label_rank", "source"],
            ["arxiv_id", "title", "label_rank", "source"],
            batch_size=args.batch_size,
            limit=args.limit,
        )
        stats["eval_sets"] = client.load_jsonl(
            "eval_sets",
            processed_dir / "eval_sets.jsonl",
            ["dataset_name", "split_name", "qid", "gold_paper_ids", "published_time"],
            ["gold_paper_ids", "published_time"],
            batch_size=args.batch_size,
            limit=args.limit,
        )
    print_json({"processed_dir": str(processed_dir), "stats": stats})


def cmd_load_qdrant(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    processed_dir = Path(args.processed_dir) if args.processed_dir else settings.processed_dir
    input_path = Path(args.input_file) if args.input_file else processed_dir / "paper_chunks.jsonl"
    client = QdrantClient(settings.qdrant_url, settings.qdrant_api_key)
    collection = args.collection or settings.qdrant_collection
    vector_size = args.vector_size or settings.qdrant_sparse_vector_size
    sparse_name = args.sparse_vector_name or settings.qdrant_sparse_vector_name
    count = 0
    batch = []
    for line_number, row in enumerate(read_jsonl(input_path), start=1):
        if line_number < args.start_line:
            continue
        if args.end_line is not None and line_number > args.end_line:
            break
        batch.append(qdrant_point_from_chunk(row, vector_size=vector_size, sparse_vector_name=sparse_name))
        count += 1
        if len(batch) >= args.batch_size:
            client.upsert_points(collection, batch, wait=args.wait)
            batch = []
        if args.limit is not None and count >= args.limit:
            break
    if batch:
        client.upsert_points(collection, batch, wait=args.wait)
    print_json(
        {
            "collection": collection,
            "input_file": str(input_path),
            "count": count,
            "vector_size": vector_size,
            "sparse_vector_name": sparse_name,
        }
    )


def cmd_load_qdrant_dense_papers(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    model_name = args.model or settings.dense_embedding_model
    if not model_name:
        raise SystemExit("--model or DENSE_EMBEDDING_MODEL is required")
    if args.paper_shard_count < 1:
        raise SystemExit("--paper-shard-count must be >= 1")
    if args.paper_shard_index < 0 or args.paper_shard_index >= args.paper_shard_count:
        raise SystemExit("--paper-shard-index must satisfy 0 <= index < count")

    client = QdrantClient(settings.qdrant_url, settings.qdrant_api_key)
    collection = args.collection or settings.qdrant_dense_paper_collection
    vector_name = args.vector_name if args.vector_name is not None else settings.qdrant_dense_vector_name
    if args.reset:
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_collection(
            collection,
            args.vector_size or settings.qdrant_dense_vector_size,
            args.distance,
            vector_name=vector_name,
        )

    try:
        embedder = build_dense_embedder(settings, model_name=model_name, device=args.device or settings.dense_embedding_device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    count = 0
    batch: list[dict[str, object]] = []
    with MySQLClient.from_settings(settings) as mysql:
        mysql.use_database(settings.mysql_database)
        last_id = str(args.start_after_paper_id or "")
        while True:
            where_parts = [f"paper_id > {sql_value(last_id)}" if last_id else "1=1"]
            if args.paper_shard_count > 1:
                where_parts.append(
                    f"MOD(CRC32(paper_id), {int(args.paper_shard_count)}) = {int(args.paper_shard_index)}"
                )
            if args.end_before_paper_id:
                where_parts.append(f"paper_id < {sql_value(args.end_before_paper_id)}")
            where = " AND ".join(where_parts)
            sql = (
                "SELECT paper_id, title, abstract, year, venue, source "
                f"FROM papers WHERE {where} ORDER BY paper_id LIMIT {int(args.page_size)}"
            )
            result = mysql.execute(sql)
            if not result.rows:
                break
            rows = [dict(zip(result.columns, values)) for values in result.rows]
            texts = [_dense_paper_text(row) for row in rows]
            vectors = embedder.encode_batch(texts, batch_size=args.encode_batch_size)
            for row, vector in zip(rows, vectors):
                batch.append(qdrant_point_from_dense_paper(row, [float(value) for value in vector], vector_name=vector_name))
                count += 1
                last_id = str(row["paper_id"])
                if len(batch) >= args.batch_size:
                    client.upsert_points(collection, batch, wait=args.wait)
                    batch = []
                if args.progress_every and count % args.progress_every == 0:
                    print_json(
                        {
                            "event": "dense_paper_load_progress",
                            "collection": collection,
                            "count": count,
                            "last_paper_id": last_id,
                            "backend": settings.dense_embedding_backend,
                            "model": model_name,
                        }
                    )
                if args.limit is not None and count >= args.limit:
                    break
            if args.limit is not None and count >= args.limit:
                break
    if batch:
        client.upsert_points(collection, batch, wait=args.wait)
    print_json(
        {
            "collection": collection,
            "count": count,
            "backend": settings.dense_embedding_backend,
            "model": model_name,
            "start_after_paper_id": args.start_after_paper_id or "",
            "end_before_paper_id": args.end_before_paper_id or "",
            "paper_shard_count": args.paper_shard_count,
            "paper_shard_index": args.paper_shard_index,
            "last_paper_id": last_id,
            "vector_name": vector_name,
            "vector_size": args.vector_size or settings.qdrant_dense_vector_size,
        }
    )


def cmd_load_qdrant_sparse_papers(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = QdrantClient(settings.qdrant_url, settings.qdrant_api_key)
    collection = args.collection or settings.qdrant_sparse_paper_collection
    sparse_name = args.sparse_vector_name or settings.qdrant_sparse_vector_name
    vector_size = args.vector_size or settings.qdrant_sparse_vector_size
    if args.reset:
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_sparse_collection(collection, sparse_name)

    count = 0
    batch: list[dict[str, object]] = []
    with MySQLClient.from_settings(settings) as mysql:
        mysql.use_database(settings.mysql_database)
        last_id = ""
        while True:
            where = f"paper_id > {sql_value(last_id)}" if last_id else "1=1"
            sql = (
                "SELECT paper_id, title, abstract, year, venue, source "
                f"FROM papers WHERE {where} ORDER BY paper_id LIMIT {int(args.page_size)}"
            )
            result = mysql.execute(sql)
            if not result.rows:
                break
            for values in result.rows:
                row = dict(zip(result.columns, values))
                batch.append(
                    qdrant_point_from_sparse_paper(
                        row,
                        vector_size=vector_size,
                        sparse_vector_name=sparse_name,
                    )
                )
                count += 1
                last_id = str(row["paper_id"])
                if len(batch) >= args.batch_size:
                    client.upsert_points(collection, batch, wait=args.wait)
                    batch = []
                if args.limit is not None and count >= args.limit:
                    break
            if args.limit is not None and count >= args.limit:
                break
    if batch:
        client.upsert_points(collection, batch, wait=args.wait)
    print_json(
        {
            "collection": collection,
            "count": count,
            "vector_size": vector_size,
            "sparse_vector_name": sparse_name,
        }
    )


def cmd_init_all(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    schema_path = settings.module_root / "sql" / "schema.sql"
    result: dict[str, object] = {}
    with MySQLClient.from_settings(settings) as mysql:
        result["mysql"] = mysql.init_schema(schema_path, settings.mysql_database, reset=args.reset)
    es_client = ElasticsearchClient(
        settings.elasticsearch_url,
        settings.elasticsearch_username,
        settings.elasticsearch_password,
    )
    result["elasticsearch"] = es_client.init_indices(settings.papers_index, settings.chunks_index, reset=args.reset)
    qdrant_client = QdrantClient(settings.qdrant_url, settings.qdrant_api_key)
    result["qdrant"] = qdrant_client.init_sparse_collection(
        settings.qdrant_collection,
        sparse_vector_name=settings.qdrant_sparse_vector_name,
        reset=args.reset,
    )
    print_json(result)


def cmd_verify_all(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    report: dict[str, object] = {}
    try:
        with MySQLClient.from_settings(settings) as mysql:
            mysql.use_database(settings.mysql_database)
            table_counts: dict[str, int | str] = {}
            for table in ("queries", "gold_labels", "eval_sets", "papers", "paper_chunks"):
                try:
                    table_counts[table] = mysql.table_count(table)
                except Exception as exc:
                    table_counts[table] = f"error: {exc}"
            report["mysql"] = {"database": settings.mysql_database, "tables": table_counts}
    except Exception as exc:
        report["mysql"] = {"error": str(exc)}
    try:
        es_client = ElasticsearchClient(
            settings.elasticsearch_url,
            settings.elasticsearch_username,
            settings.elasticsearch_password,
        )
        es_counts: dict[str, int | str] = {}
        for index in (settings.papers_index, settings.chunks_index):
            try:
                es_counts[index] = es_client.count(index)
            except Exception as exc:
                es_counts[index] = f"error: {exc}"
        report["elasticsearch"] = {"health": es_client.health(), "counts": es_counts}
    except Exception as exc:
        report["elasticsearch"] = {"error": str(exc)}
    try:
        qdrant_client = QdrantClient(settings.qdrant_url, settings.qdrant_api_key)
        report["qdrant"] = qdrant_client.collection(settings.qdrant_collection)
        try:
            report["qdrant_dense_papers"] = qdrant_client.collection(settings.qdrant_dense_paper_collection)
        except Exception as exc:
            report["qdrant_dense_papers"] = {"error": str(exc)}
    except Exception as exc:
        report["qdrant"] = {"error": str(exc)}
    print_json(report)


def cmd_search_es(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = ElasticsearchClient(
        settings.elasticsearch_url,
        settings.elasticsearch_username,
        settings.elasticsearch_password,
    )
    if args.kind == "papers":
        hits = client.search_papers(args.index or settings.papers_index, args.query, top_k=args.top_k)
    else:
        hits = client.search_chunks(args.index or settings.chunks_index, args.query, top_k=args.top_k)
    print_json(hits)


def cmd_search_qdrant(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = QdrantClient(settings.qdrant_url, settings.qdrant_api_key)
    hits = client.search_sparse(
        args.collection or settings.qdrant_collection,
        args.query,
        vector_size=args.vector_size or settings.qdrant_sparse_vector_size,
        sparse_vector_name=args.sparse_vector_name or settings.qdrant_sparse_vector_name,
        top_k=args.top_k,
    )
    print_json(hits)


def _dense_paper_text(row: dict[str, object]) -> str:
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    venue = str(row.get("venue") or "").strip()
    year = str(row.get("year") or "").strip()
    parts = [f"title: {title}"]
    if venue or year:
        parts.append(f"metadata: {venue} {year}".strip())
    if abstract:
        parts.append(f"abstract: {abstract[:2200]}")
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SaiTi3 data ingestion and indexing utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Print resolved config and optional service health")
    doctor.add_argument("--check-qdrant", action="store_true")
    doctor.add_argument("--check-es", action="store_true")
    doctor.add_argument("--check-mysql", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    convert = subparsers.add_parser("convert-pasa", help="Convert PaSa data into processed JSONL files")
    convert.add_argument("--limit", type=int, default=None, help="Limit rows per source file for smoke tests")
    convert.add_argument("--output-dir", default=None)
    convert.set_defaults(func=cmd_convert_pasa)

    init_mysql = subparsers.add_parser("init-mysql", help="Create MySQL database and tables")
    init_mysql.add_argument("--schema", default=None)
    init_mysql.add_argument("--reset", action="store_true", help="Drop known tables before recreating schema")
    init_mysql.set_defaults(func=cmd_init_mysql)

    load_mysql = subparsers.add_parser("load-mysql", help="Load processed JSONL into MySQL")
    load_mysql.add_argument("--processed-dir", default=None)
    load_mysql.add_argument("--batch-size", type=int, default=500)
    load_mysql.add_argument("--paper-batch-size", type=int, default=500)
    load_mysql.add_argument("--chunk-batch-size", type=int, default=100)
    load_mysql.add_argument("--limit", type=int, default=None, help="Limit rows per table for smoke tests")
    load_mysql.set_defaults(func=cmd_load_mysql)

    init_es = subparsers.add_parser("init-es", help="Create Elasticsearch indices")
    init_es.add_argument("--papers-index", default=Settings.from_env().papers_index)
    init_es.add_argument("--chunks-index", default=Settings.from_env().chunks_index)
    init_es.add_argument("--reset", action="store_true", help="Delete indices before recreating them")
    init_es.set_defaults(func=cmd_init_es)

    load_es = subparsers.add_parser("load-es", help="Bulk load processed JSONL into Elasticsearch")
    load_es.add_argument("--kind", choices=("papers", "chunks"), required=True)
    load_es.add_argument("--index", required=True)
    load_es.add_argument("--processed-dir", default=None)
    load_es.set_defaults(func=cmd_load_es)

    init_qdrant = subparsers.add_parser("init-qdrant", help="Create a Qdrant collection")
    init_qdrant.add_argument("--collection", default=None)
    init_qdrant.add_argument("--sparse", action="store_true", help="Create sparse lexical baseline collection")
    init_qdrant.add_argument("--sparse-vector-name", default=None)
    init_qdrant.add_argument("--vector-size", type=int, default=768)
    init_qdrant.add_argument("--distance", choices=("Cosine", "Dot", "Euclid"), default="Cosine")
    init_qdrant.add_argument("--reset", action="store_true", help="Delete collection before recreating it")
    init_qdrant.set_defaults(func=cmd_init_qdrant)

    load_qdrant = subparsers.add_parser("load-qdrant", help="Load lexical sparse chunk vectors into Qdrant")
    load_qdrant.add_argument("--processed-dir", default=None)
    load_qdrant.add_argument("--input-file", default=None)
    load_qdrant.add_argument("--collection", default=None)
    load_qdrant.add_argument("--batch-size", type=int, default=1024)
    load_qdrant.add_argument("--vector-size", type=int, default=None)
    load_qdrant.add_argument("--sparse-vector-name", default=None)
    load_qdrant.add_argument("--start-line", type=int, default=1)
    load_qdrant.add_argument("--end-line", type=int, default=None)
    load_qdrant.add_argument("--limit", type=int, default=None)
    load_qdrant.add_argument("--no-wait", dest="wait", action="store_false")
    load_qdrant.set_defaults(wait=True)
    load_qdrant.set_defaults(func=cmd_load_qdrant)

    load_dense_papers = subparsers.add_parser(
        "load-qdrant-dense-papers",
        help="Embed paper title+abstract rows and load a dense Qdrant paper collection",
    )
    load_dense_papers.add_argument("--collection", default=None)
    load_dense_papers.add_argument("--model", default=None)
    load_dense_papers.add_argument("--device", default=None)
    load_dense_papers.add_argument("--vector-name", default=None)
    load_dense_papers.add_argument("--vector-size", type=int, default=None)
    load_dense_papers.add_argument("--distance", choices=("Cosine", "Dot", "Euclid"), default="Cosine")
    load_dense_papers.add_argument("--page-size", type=int, default=512)
    load_dense_papers.add_argument("--batch-size", type=int, default=512)
    load_dense_papers.add_argument("--encode-batch-size", type=int, default=32)
    load_dense_papers.add_argument("--limit", type=int, default=None)
    load_dense_papers.add_argument("--start-after-paper-id", default="")
    load_dense_papers.add_argument("--end-before-paper-id", default="")
    load_dense_papers.add_argument("--paper-shard-count", type=int, default=1)
    load_dense_papers.add_argument("--paper-shard-index", type=int, default=0)
    load_dense_papers.add_argument("--progress-every", type=int, default=5000)
    load_dense_papers.add_argument("--reset", action="store_true")
    load_dense_papers.add_argument("--no-wait", dest="wait", action="store_false")
    load_dense_papers.set_defaults(wait=True)
    load_dense_papers.set_defaults(func=cmd_load_qdrant_dense_papers)

    load_sparse_papers = subparsers.add_parser(
        "load-qdrant-sparse-papers",
        help="Load paper title+abstract sparse vectors into a Qdrant paper collection",
    )
    load_sparse_papers.add_argument("--collection", default=None)
    load_sparse_papers.add_argument("--sparse-vector-name", default=None)
    load_sparse_papers.add_argument("--vector-size", type=int, default=None)
    load_sparse_papers.add_argument("--page-size", type=int, default=1024)
    load_sparse_papers.add_argument("--batch-size", type=int, default=1024)
    load_sparse_papers.add_argument("--limit", type=int, default=None)
    load_sparse_papers.add_argument("--reset", action="store_true")
    load_sparse_papers.add_argument("--no-wait", dest="wait", action="store_false")
    load_sparse_papers.set_defaults(wait=True)
    load_sparse_papers.set_defaults(func=cmd_load_qdrant_sparse_papers)

    init_all = subparsers.add_parser("init-all", help="Initialize MySQL, Elasticsearch and Qdrant")
    init_all.add_argument("--reset", action="store_true", help="Drop/recreate all configured stores")
    init_all.set_defaults(func=cmd_init_all)

    verify_all = subparsers.add_parser("verify-all", help="Verify configured database/index/collection status")
    verify_all.set_defaults(func=cmd_verify_all)

    search_es = subparsers.add_parser("search-es", help="Smoke search Elasticsearch")
    search_es.add_argument("--query", required=True)
    search_es.add_argument("--kind", choices=("papers", "chunks"), default="papers")
    search_es.add_argument("--index", default=None)
    search_es.add_argument("--top-k", type=int, default=5)
    search_es.set_defaults(func=cmd_search_es)

    search_qdrant = subparsers.add_parser("search-qdrant", help="Smoke search Qdrant sparse collection")
    search_qdrant.add_argument("--query", required=True)
    search_qdrant.add_argument("--collection", default=None)
    search_qdrant.add_argument("--sparse-vector-name", default=None)
    search_qdrant.add_argument("--vector-size", type=int, default=None)
    search_qdrant.add_argument("--top-k", type=int, default=5)
    search_qdrant.set_defaults(func=cmd_search_qdrant)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
