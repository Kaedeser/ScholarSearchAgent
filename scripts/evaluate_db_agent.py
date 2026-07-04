"""Evaluate ScholarSearchAgent against test labels stored in the configured MySQL DB."""

from __future__ import annotations

import argparse
import zlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.scholar_api.bootstrap.container import build_search_pipeline  # noqa: E402
from packages.scholar_eval.evaluation import score_prediction  # noqa: E402
from packages.scholar_ingest.config import Settings  # noqa: E402
from packages.scholar_infra.io.jsonl import write_json  # noqa: E402
from packages.scholar_infra.persistence.mysql import MySQLClient, sql_value  # noqa: E402


DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_DATASETS = ("AutoScholarQuery:test", "RealScholarQuery:test")


@dataclass(frozen=True)
class QueryResult:
    dataset_name: str
    split_name: str
    qid: str
    query: str
    gold_count: int
    predicted_count: int
    latency_sec: float
    top_predictions: list[str]
    retrieved_papers: list[dict[str, Any]]
    gold_ids: list[str]
    gold_papers: list[dict[str, Any]]
    gold_hit_ranks: dict[str, int]
    pool_metrics: dict[str, Any]
    metrics: dict[str, dict[str, float | int]]
    diagnostics: dict[str, Any]
    query_type: str
    rewrite_used: bool
    dense_used: bool
    alias_used: bool
    model_events: dict[str, Any]
    pipeline_cost: dict[str, Any]


def _avg(values: Iterable[float]) -> float:
    materialized = list(values)
    return round(mean(materialized), 6) if materialized else 0.0


def load_eval_rows(
    settings: Settings,
    dataset_specs: list[str],
    *,
    max_queries_per_dataset: int | None,
    sample_order: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with MySQLClient.from_settings(settings) as mysql:
        mysql.use_database(settings.mysql_database)
        for spec in dataset_specs:
            dataset_name, split_name = spec.split(":", 1)
            order_sql = "ORDER BY q.qid"
            if sample_order == "hash":
                order_sql = "ORDER BY CRC32(CONCAT(q.dataset_name, '/', q.split_name, '/', q.qid)), q.qid"
            elif sample_order != "qid":
                raise ValueError(f"Unsupported sample order: {sample_order}")
            sql = (
                "SELECT q.dataset_name, q.split_name, q.qid, q.query_text, e.gold_paper_ids "
                "FROM queries q JOIN eval_sets e ON q.qid=e.qid "
                f"WHERE q.dataset_name={sql_value(dataset_name)} AND q.split_name={sql_value(split_name)} "
                f"{order_sql}"
            )
            if max_queries_per_dataset is not None:
                sql += f" LIMIT {int(max_queries_per_dataset)}"
            result = mysql.execute(sql)
            columns = result.columns
            for values in result.rows:
                item = dict(zip(columns, values))
                gold_raw = item.get("gold_paper_ids") or "[]"
                if isinstance(gold_raw, str):
                    gold_ids = json.loads(gold_raw)
                else:
                    gold_ids = gold_raw
                rows.append(
                    {
                        "dataset_name": item["dataset_name"],
                        "split_name": item["split_name"],
                        "qid": item["qid"],
                        "query": item["query_text"],
                        "gold_ids": list(gold_ids or []),
                    }
                )
    return rows


def load_paper_details(settings: Settings, paper_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    unique_ids = [paper_id for paper_id in dict.fromkeys(str(item) for item in paper_ids) if paper_id]
    if not unique_ids:
        return details
    with MySQLClient.from_settings(settings) as mysql:
        mysql.use_database(settings.mysql_database)
        # Batch only the fields needed for the report. Fetching full abstracts one
        # paper at a time makes setup painfully slow on large multi-answer samples.
        for start in range(0, len(unique_ids), 100):
            chunk = unique_ids[start : start + 100]
            id_sql = ",".join(sql_value(paper_id) for paper_id in chunk)
            result = mysql.execute(f"SELECT paper_id, title, year, venue FROM papers WHERE paper_id IN ({id_sql})")
            for values in result.rows:
                item = dict(zip(result.columns, values))
                paper_id = str(item.get("paper_id") or "")
                if paper_id:
                    details[paper_id] = item
    return details


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings.from_env()
    cutoffs = sorted(set(args.cutoffs))
    top_k = max(cutoffs)
    pool_cutoffs = sorted(set(args.pool_cutoffs or [100, 200, 500]))
    diagnostic_top_k = max(top_k, int(args.diagnostic_top_k or top_k), max(pool_cutoffs))
    rows = load_eval_rows(
        settings,
        args.datasets,
        max_queries_per_dataset=args.max_queries_per_dataset,
        sample_order=args.sample_order,
    )
    rows = select_sample_rows(rows, max_total_queries=args.max_total_queries, sample_profile=args.sample_profile)
    unsharded_query_count = len(rows)
    rows = shard_eval_rows(rows, shard_count=args.shard_count, shard_index=args.shard_index)
    gold_paper_details = load_paper_details(settings, _iter_gold_ids(rows))
    if args.progress_every >= 0:
        print(
            f"[setup] loaded_eval_rows={len(rows)} unsharded_eval_rows={unsharded_query_count} "
            f"shard={args.shard_index}/{args.shard_count} gold_detail_rows={len(gold_paper_details)}",
            flush=True,
        )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_search_pipeline(
        Path(args.processed_dir),
        paper_limit=None,
        chunk_limit=None,
        max_chunks_per_paper=args.max_chunks_per_paper,
        per_query_top_k=args.per_query_top_k,
        backend="database",
        model_services_enabled=False if args.disable_model_services else None,
    )

    started = time.perf_counter()
    per_query: list[QueryResult] = []
    failures: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        query = str(row["query"] or "")
        qid = str(row["qid"])
        if not query:
            failures.append({"qid": qid, "error": "empty query"})
            continue
        try:
            query_started = time.perf_counter()
            response = pipeline.search(query, top_k=diagnostic_top_k)
            latency = round(time.perf_counter() - query_started, 4)
        except Exception as exc:
            failures.append({"qid": qid, "error": str(exc)})
            continue
        predicted = [candidate.canonical_id or candidate.paper_id for candidate in response.papers]
        diagnostic_pool = _diagnostic_pool_from_response(response)
        pool_predicted = [str(item.get("paper_id") or "") for item in diagnostic_pool if item.get("paper_id")]
        gold = list(row["gold_ids"])
        gold_hit_ranks = _gold_hit_ranks(predicted, gold)
        pool_gold_hit_ranks = _gold_hit_ranks(pool_predicted, gold)
        source_hit_ranks = _source_gold_hit_ranks_from_pool(diagnostic_pool, gold)
        pool_metrics = build_pool_metrics(
            gold_ids=gold,
            predicted_ids=pool_predicted,
            source_hit_ranks=source_hit_ranks,
            pool_cutoffs=pool_cutoffs,
            diagnostic_pool=diagnostic_pool,
        )
        metrics: dict[str, dict[str, float | int]] = {}
        for cutoff in cutoffs:
            scored = score_prediction(predicted, gold, k=cutoff)
            metrics[f"@{cutoff}"] = {
                "hits": scored.hits,
                "precision": scored.precision_at_k,
                "recall": scored.recall_at_k,
                "f1": scored.f1_at_k,
                "mrr": scored.mrr,
                "hit_rate": 1.0 if scored.hits else 0.0,
            }
        diagnostics = diagnose_query(
            gold_ids=gold,
            predicted_ids=predicted,
            gold_hit_ranks=gold_hit_ranks,
            pool_gold_hit_ranks=pool_gold_hit_ranks,
            response_cost=response.cost,
            model_events=response.cost.get("model_services") or {},
            coverage_missing=response.coverage.missing_constraints,
            parsed_sub_queries=response.parsed_query.sub_queries,
            metrics=metrics,
            pool_metrics=pool_metrics,
            cutoffs=cutoffs,
        )
        per_query.append(
            QueryResult(
                dataset_name=str(row["dataset_name"]),
                split_name=str(row["split_name"]),
                qid=qid,
                query=query,
                gold_count=len(gold),
                predicted_count=len(predicted),
                latency_sec=latency,
                top_predictions=predicted[: args.output_paper_limit],
                retrieved_papers=[
                    _candidate_snapshot(candidate, rank)
                    for rank, candidate in enumerate(response.papers[: args.output_paper_limit], start=1)
                ],
                gold_ids=gold,
                gold_papers=[_paper_snapshot(paper_id, gold_paper_details.get(paper_id)) for paper_id in gold],
                gold_hit_ranks=gold_hit_ranks,
                pool_metrics=pool_metrics,
                metrics=metrics,
                diagnostics=diagnostics,
                query_type=str(response.cost.get("query_type") or ""),
                rewrite_used=bool(response.cost.get("rewrite_used")),
                dense_used=bool(response.cost.get("dense_used")),
                alias_used=bool(response.cost.get("alias_used")),
                model_events=response.cost.get("model_services") or {},
                pipeline_cost=_compact_cost(response.cost),
            )
        )
        if args.progress_every > 0 and index % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[progress] processed={index}/{len(rows)} elapsed={elapsed:.2f}s avg={elapsed / index:.3f}s",
                flush=True,
            )

    elapsed = round(time.perf_counter() - started, 3)
    by_dataset: dict[str, list[QueryResult]] = {}
    for item in per_query:
        by_dataset.setdefault(f"{item.dataset_name}/{item.split_name}", []).append(item)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ScholarSearchAgent database-backed retrieval evaluation",
        "config": {
            "mysql_host": settings.mysql_host,
            "mysql_database": settings.mysql_database,
            "elasticsearch_url": settings.elasticsearch_url,
            "papers_index": settings.papers_index,
            "chunks_index": settings.chunks_index,
            "qdrant_url": settings.qdrant_url,
            "qdrant_collection": settings.qdrant_collection,
            "datasets": args.datasets,
            "cutoffs": cutoffs,
            "pool_cutoffs": pool_cutoffs,
            "top_k": top_k,
            "diagnostic_top_k": diagnostic_top_k,
            "output_paper_limit": args.output_paper_limit,
            "per_query_top_k": args.per_query_top_k,
            "model_services": "disabled" if args.disable_model_services else "config-default",
            "max_queries_per_dataset": args.max_queries_per_dataset,
            "max_total_queries": args.max_total_queries,
            "sample_profile": args.sample_profile,
            "sample_order": args.sample_order,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
        "data": {
            "unsharded_queries": unsharded_query_count,
            "requested_queries": len(rows),
            "evaluated_queries": len(per_query),
            "failed_queries": len(failures),
            "failures": failures[:200],
            "gold_labels": sum(item.gold_count for item in per_query),
        },
        "runtime": {
            "elapsed_sec": elapsed,
            "avg_latency_sec": _avg([item.latency_sec for item in per_query]),
        },
        "summary": summarize(per_query, cutoffs),
        "by_dataset": {key: summarize(value, cutoffs) for key, value in by_dataset.items()},
        "diagnostics": summarize_diagnostics(per_query, cutoffs),
        "per_query": [asdict(item) for item in per_query],
    }
    write_json(report_dir / args.json_output, report)
    (report_dir / args.markdown_output).write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return report


def shard_eval_rows(rows: list[dict[str, Any]], *, shard_count: int, shard_index: int) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    if shard_count == 1:
        return rows
    return [
        row
        for row in rows
        if stable_query_shard(row, shard_count=shard_count) == shard_index
    ]


def stable_query_shard(row: dict[str, Any], *, shard_count: int) -> int:
    key = f"{row.get('dataset_name', '')}/{row.get('split_name', '')}/{row.get('qid', '')}"
    return zlib.crc32(key.encode("utf-8")) % shard_count


def select_sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_total_queries: int | None,
    sample_profile: str,
) -> list[dict[str, Any]]:
    if max_total_queries is None or max_total_queries <= 0 or len(rows) <= max_total_queries:
        return rows
    if sample_profile == "head":
        return rows[:max_total_queries]
    by_dataset: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset_name") or ""), str(row.get("split_name") or ""))
        by_dataset.setdefault(key, []).append(row)
    if sample_profile == "balanced":
        allocations = _balanced_allocations(by_dataset, max_total_queries)
    elif sample_profile == "proportional":
        allocations = _proportional_allocations(by_dataset, max_total_queries)
    else:
        raise ValueError(f"Unsupported sample profile: {sample_profile}")
    selected: list[dict[str, Any]] = []
    for key, group in by_dataset.items():
        selected.extend(group[: allocations.get(key, 0)])
    selected.sort(key=lambda row: _stable_query_key(row))
    return selected[:max_total_queries]


def _balanced_allocations(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    total: int,
) -> dict[tuple[str, str], int]:
    allocations = {key: 0 for key in groups}
    remaining = min(total, sum(len(group) for group in groups.values()))
    keys = sorted(groups)
    while remaining > 0:
        progressed = False
        for key in keys:
            if remaining <= 0:
                break
            if allocations[key] >= len(groups[key]):
                continue
            allocations[key] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return allocations


def _proportional_allocations(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    total: int,
) -> dict[tuple[str, str], int]:
    available = sum(len(group) for group in groups.values())
    target = min(total, available)
    raw: list[tuple[tuple[str, str], int, float]] = []
    for key, group in groups.items():
        exact = target * len(group) / available
        base = min(len(group), int(exact))
        raw.append((key, base, exact - base))
    allocations = {key: base for key, base, _ in raw}
    if target >= len(groups):
        for key, group in groups.items():
            if allocations[key] == 0 and group:
                allocations[key] = 1
    remaining = target - sum(allocations.values())
    for key, _, _ in sorted(raw, key=lambda item: item[2], reverse=True):
        if remaining <= 0:
            break
        capacity = len(groups[key]) - allocations[key]
        if capacity <= 0:
            continue
        add = min(capacity, remaining)
        allocations[key] += add
        remaining -= add
    while remaining < 0:
        for key, _, _ in sorted(raw, key=lambda item: item[2]):
            if remaining == 0:
                break
            minimum = 1 if target >= len(groups) and groups[key] else 0
            if allocations[key] > minimum:
                allocations[key] -= 1
                remaining += 1
    return allocations


def _stable_query_key(row: dict[str, Any]) -> int:
    key = f"{row.get('dataset_name', '')}/{row.get('split_name', '')}/{row.get('qid', '')}"
    return zlib.crc32(key.encode("utf-8"))


def _iter_gold_ids(rows: list[dict[str, Any]]) -> Iterable[str]:
    for row in rows:
        yield from list(row.get("gold_ids") or [])


def _gold_hit_ranks(predicted_ids: list[str], gold_ids: list[str]) -> dict[str, int]:
    gold = set(gold_ids)
    hit_ranks: dict[str, int] = {}
    for rank, paper_id in enumerate(predicted_ids, start=1):
        if paper_id in gold and paper_id not in hit_ranks:
            hit_ranks[paper_id] = rank
    return hit_ranks


def _diagnostic_pool_from_response(response: Any) -> list[dict[str, Any]]:
    pool = response.cost.get("diagnostic_pool_candidates") or []
    if isinstance(pool, list) and pool:
        return [item for item in pool if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(response.papers, start=1):
        rows.append(
            {
                "rank": rank,
                "paper_id": candidate.canonical_id or candidate.paper_id,
                "sources": sorted(candidate.sources),
                "source_ranks": candidate.metadata.get("source_ranks") or {},
            }
        )
    return rows


def _source_gold_hit_ranks(candidates: list[Any], gold_ids: list[str]) -> dict[str, dict[str, int]]:
    gold = set(gold_ids)
    result: dict[str, dict[str, int]] = {}
    for rank, candidate in enumerate(candidates, start=1):
        paper_id = candidate.canonical_id or candidate.paper_id
        if paper_id not in gold:
            continue
        source_ranks = candidate.metadata.get("source_ranks") or {}
        if not isinstance(source_ranks, dict) or not source_ranks:
            for source in candidate.sources:
                result.setdefault(str(source), {}).setdefault(paper_id, rank)
            continue
        for source, source_rank in source_ranks.items():
            try:
                rank_value = int(source_rank)
            except (TypeError, ValueError):
                rank_value = rank
            source_hits = result.setdefault(str(source), {})
            source_hits[paper_id] = min(source_hits.get(paper_id, rank_value), rank_value)
    return result


def _source_gold_hit_ranks_from_pool(pool: list[dict[str, Any]], gold_ids: list[str]) -> dict[str, dict[str, int]]:
    gold = set(gold_ids)
    result: dict[str, dict[str, int]] = {}
    for item in pool:
        paper_id = str(item.get("paper_id") or "")
        if paper_id not in gold:
            continue
        source_ranks = item.get("source_ranks") or {}
        if isinstance(source_ranks, dict) and source_ranks:
            for source, source_rank in source_ranks.items():
                try:
                    rank_value = int(source_rank)
                except (TypeError, ValueError):
                    rank_value = int(item.get("rank") or 0)
                source_hits = result.setdefault(str(source), {})
                source_hits[paper_id] = min(source_hits.get(paper_id, rank_value), rank_value)
            continue
        for source in item.get("sources") or []:
            source_hits = result.setdefault(str(source), {})
            rank_value = int(item.get("rank") or 0)
            source_hits[paper_id] = min(source_hits.get(paper_id, rank_value), rank_value)
    return result


def build_pool_metrics(
    *,
    gold_ids: list[str],
    predicted_ids: list[str],
    source_hit_ranks: dict[str, dict[str, int]],
    pool_cutoffs: list[int],
    diagnostic_pool: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gold = set(gold_ids)
    pool = diagnostic_pool or []
    metrics: dict[str, Any] = {
        "source_gold_hit_ranks": source_hit_ranks,
        "source_gold_hits": {},
        "pool_recall": {},
        "pool_hits": {},
        "late_gold_count": 0,
        "dense_gold_hits": {},
        "alias_gold_hits": {},
        "alias_false_positive_top20": 0,
        "alias_to_concept_coverage": 0,
    }
    for cutoff in pool_cutoffs:
        hits = len(gold & set(predicted_ids[:cutoff]))
        metrics["pool_hits"][f"@{cutoff}"] = hits
        metrics["pool_recall"][f"@{cutoff}"] = round(hits / len(gold), 6) if gold else 0.0
    if pool_cutoffs:
        first_cutoff = min(pool_cutoffs)
        last_cutoff = max(pool_cutoffs)
        metrics["late_gold_count"] = max(
            0,
            len(gold & set(predicted_ids[:last_cutoff])) - len(gold & set(predicted_ids[:first_cutoff])),
        )
    for source, ranks in source_hit_ranks.items():
        source_row: dict[str, int] = {}
        for cutoff in pool_cutoffs:
            source_row[f"@{cutoff}"] = sum(1 for rank in ranks.values() if rank <= cutoff)
        metrics["source_gold_hits"][source] = source_row
    for cutoff in pool_cutoffs:
        dense_hits = 0
        alias_hits = 0
        for item in pool[:cutoff]:
            paper_id = str(item.get("paper_id") or "")
            if paper_id not in gold:
                continue
            if item.get("dense_used"):
                dense_hits += 1
            if item.get("alias_used"):
                alias_hits += 1
        metrics["dense_gold_hits"][f"@{cutoff}"] = dense_hits
        metrics["alias_gold_hits"][f"@{cutoff}"] = alias_hits
    alias_top20 = [item for item in pool[:20] if item.get("alias_used")]
    metrics["alias_false_positive_top20"] = sum(
        1 for item in alias_top20 if str(item.get("paper_id") or "") not in gold
    )
    metrics["alias_to_concept_coverage"] = sum(1 for item in alias_top20 if item.get("alias_relations"))
    return metrics


def _candidate_snapshot(candidate, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "paper_id": candidate.canonical_id or candidate.paper_id,
        "title": candidate.title,
        "year": candidate.year,
        "score": round(candidate.final_score, 6),
        "relevance": candidate.relevance,
        "sources": sorted(candidate.sources),
        "raw_scores": {key: round(float(value), 6) for key, value in candidate.raw_scores.items()},
        "source_ranks": candidate.metadata.get("source_ranks") or {},
        "source_rank_backfill": candidate.metadata.get("source_rank_backfill"),
        "matched_constraints": candidate.matched_constraints,
        "missing_constraints": candidate.missing_constraints,
        "evidence": candidate.snippets[:2],
        "crawler_strategy": candidate.metadata.get("crawler_strategy"),
    }


def _paper_snapshot(paper_id: str, paper: dict[str, Any] | None) -> dict[str, Any]:
    if not paper:
        return {"paper_id": paper_id, "title": ""}
    return {
        "paper_id": paper_id,
        "title": paper.get("title") or "",
        "year": paper.get("year"),
        "venue": paper.get("venue"),
    }


def _compact_cost(cost: dict[str, Any]) -> dict[str, Any]:
    # Keep per-query reports compact while preserving the stage counters needed for diagnosis.
    keys = (
        "latency_sec",
        "rounds",
        "actions_executed",
        "raw_candidates",
        "unique_candidates",
        "backend",
        "es_papers",
        "es_chunks",
        "qdrant_points",
        "qdrant_status",
        "qdrant_dense_paper_status",
        "qdrant_dense_paper_collection",
        "qdrant_dense_paper_error",
        "qdrant_sparse_paper_status",
        "qdrant_sparse_paper_points",
        "qdrant_sparse_paper_error",
        "backend_fallback_reason",
        "query_type",
        "rewrite_used",
        "dense_used",
        "sparse_paper_used",
        "alias_used",
    )
    compact = {key: cost.get(key) for key in keys if key in cost}
    if "diagnostic_pool_candidates" in cost:
        compact["diagnostic_pool_candidates"] = cost.get("diagnostic_pool_candidates", [])[:100]
    return compact


def diagnose_query(
    *,
    gold_ids: list[str],
    predicted_ids: list[str],
    gold_hit_ranks: dict[str, int],
    pool_gold_hit_ranks: dict[str, int],
    response_cost: dict[str, Any],
    model_events: dict[str, Any],
    coverage_missing: list[str],
    parsed_sub_queries: list[str],
    metrics: dict[str, dict[str, float | int]],
    pool_metrics: dict[str, Any],
    cutoffs: list[int],
) -> dict[str, Any]:
    max_cutoff = max(cutoffs)
    max_key = f"@{max_cutoff}"
    max_metrics = metrics[max_key]
    gold_count = len(gold_ids)
    hits_at_max = int(max_metrics["hits"])
    recall_at_max = float(max_metrics["recall"])
    topk_recall_ceiling = min(1.0, max_cutoff / gold_count) if gold_count else 0.0
    missed_gold = [paper_id for paper_id in gold_ids if paper_id not in gold_hit_ranks]
    late_hits = {
        paper_id: rank
        for paper_id, rank in gold_hit_ranks.items()
        if rank > max_cutoff
    }
    model_errors = list(model_events.get("errors") or [])
    raw_candidates = int(response_cost.get("raw_candidates") or 0)
    unique_candidates = int(response_cost.get("unique_candidates") or 0)
    pool_recall = pool_metrics.get("pool_recall") or {}
    max_pool_key = _max_pool_key(pool_recall)
    max_pool_recall = float(pool_recall.get(max_pool_key, 0.0)) if max_pool_key else recall_at_max
    max_pool_hits = int((pool_metrics.get("pool_hits") or {}).get(max_pool_key, len(pool_gold_hit_ranks))) if max_pool_key else len(pool_gold_hit_ranks)

    # Diagnosis is stage-oriented: first find hard failures, then separate
    # retrieval misses from ranking/cutoff misses and top-K ceiling effects.
    if not parsed_sub_queries:
        reason = "query_intent_blocked_or_empty_plan"
        stage = "query_intent"
    elif raw_candidates == 0 or unique_candidates == 0 or not predicted_ids:
        reason = "retrieval_empty"
        stage = "multi_source_retrieval"
    elif hits_at_max == 0 and (late_hits or max_pool_hits > 0):
        reason = "gold_retrieved_but_ranked_below_cutoff"
        stage = "selector_reranker_or_rule_ranker"
    elif hits_at_max == 0:
        reason = "gold_not_found_in_diagnostic_pool"
        stage = "query_understanding_and_retrieval"
    elif recall_at_max < 0.5 and topk_recall_ceiling <= 0.5:
        reason = "large_gold_set_topk_ceiling"
        stage = "coverage_iteration_or_evaluation_k"
    elif coverage_missing:
        reason = "top_results_miss_required_constraints"
        stage = "query_understanding_or_coverage_iteration"
    elif model_errors:
        reason = "model_service_errors_present"
        stage = "model_service_infra"
    elif recall_at_max < 0.5:
        reason = "partial_recall_needs_better_ranking_or_expansion"
        stage = "selector_reranker_or_retrieval_expansion"
    else:
        reason = "acceptable_for_current_cutoff"
        stage = "monitor"

    return {
        "primary_reason": reason,
        "priority_stage": stage,
        "gold_hit_ranks": gold_hit_ranks,
        "pool_gold_hit_ranks": pool_gold_hit_ranks,
        "late_gold_hits": late_hits,
        "missed_gold_count": len(missed_gold),
        "missed_gold_ids": missed_gold[:20],
        "topk_recall_ceiling": round(topk_recall_ceiling, 6),
        "diagnostic_pool_hit_count": len(gold_hit_ranks),
        "pool_recall": pool_recall,
        "pool_hit_count": max_pool_hits,
        "pool_recall_at_max_pool": round(max_pool_recall, 6),
        "source_gold_hits": pool_metrics.get("source_gold_hits") or {},
        "late_gold_count": pool_metrics.get("late_gold_count", 0),
        "raw_candidates": raw_candidates,
        "unique_candidates": unique_candidates,
        "model_errors": model_errors,
        "coverage_missing": coverage_missing,
    }


def _max_pool_key(pool_recall: dict[str, Any]) -> str | None:
    keys = [key for key in pool_recall if key.startswith("@")]
    if not keys:
        return None
    return max(keys, key=lambda key: int(key[1:]))


def summarize(items: list[QueryResult], cutoffs: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "query_count": len(items),
        "gold_labels": sum(item.gold_count for item in items),
        "avg_gold_per_query": _avg([item.gold_count for item in items]),
        "avg_latency_sec": _avg([item.latency_sec for item in items]),
    }
    for cutoff in cutoffs:
        key = f"@{cutoff}"
        result[key] = {
            "precision": _avg([item.metrics[key]["precision"] for item in items]),
            "recall": _avg([item.metrics[key]["recall"] for item in items]),
            "f1": _avg([item.metrics[key]["f1"] for item in items]),
            "mrr": _avg([item.metrics[key]["mrr"] for item in items]),
            "hit_rate": _avg([item.metrics[key]["hit_rate"] for item in items]),
            "hits": sum(int(item.metrics[key]["hits"]) for item in items),
        }
    return result


def summarize_diagnostics(items: list[QueryResult], cutoffs: list[int]) -> dict[str, Any]:
    max_cutoff = max(cutoffs)
    max_key = f"@{max_cutoff}"
    reason_counts = Counter(item.diagnostics["primary_reason"] for item in items)
    stage_counts = Counter(item.diagnostics["priority_stage"] for item in items)
    model_error_count = sum(len(item.diagnostics.get("model_errors") or []) for item in items)
    no_hit_items = [item for item in items if int(item.metrics[max_key]["hits"]) == 0]
    pool_summary = summarize_pool_metrics(items)
    return {
        "cutoff_used_for_diagnosis": max_cutoff,
        "primary_reason_counts": dict(reason_counts.most_common()),
        "priority_stage_counts": dict(stage_counts.most_common()),
        "model_error_count": model_error_count,
        "zero_hit_queries": len(no_hit_items),
        "zero_hit_rate": _avg([1.0 if item in no_hit_items else 0.0 for item in items]),
        "avg_topk_recall_ceiling": _avg([item.diagnostics["topk_recall_ceiling"] for item in items]),
        "pool_recall": pool_summary["pool_recall"],
        "source_gold_hits": pool_summary["source_gold_hits"],
        "dense_gold_hits": pool_summary["dense_gold_hits"],
        "alias_gold_hits": pool_summary["alias_gold_hits"],
        "alias_false_positive_top20": pool_summary["alias_false_positive_top20"],
        "alias_to_concept_coverage": pool_summary["alias_to_concept_coverage"],
        "query_type_counts": dict(Counter(item.query_type or "unknown" for item in items).most_common()),
        "query_type_source_gold_hits": summarize_query_type_source_hits(items),
        "feature_usage": summarize_feature_usage(items),
        "selector_preselector": summarize_selector_preselection(items),
        "late_gold_count": pool_summary["late_gold_count"],
        "recommendations": build_recommendations(reason_counts, stage_counts, model_error_count),
    }


def summarize_pool_metrics(items: list[QueryResult]) -> dict[str, Any]:
    pool_keys: list[str] = []
    for item in items:
        for key in (item.pool_metrics.get("pool_recall") or {}):
            if key not in pool_keys:
                pool_keys.append(key)
    pool_keys.sort(key=lambda key: int(key[1:]))
    source_hits: dict[str, Counter[str]] = {}
    dense_hits: Counter[str] = Counter()
    alias_hits: Counter[str] = Counter()
    alias_false_positive_top20 = 0
    alias_to_concept_coverage = 0
    for item in items:
        for source, hits_by_cutoff in (item.pool_metrics.get("source_gold_hits") or {}).items():
            counter = source_hits.setdefault(source, Counter())
            for key, value in hits_by_cutoff.items():
                counter[key] += int(value)
        for key, value in (item.pool_metrics.get("dense_gold_hits") or {}).items():
            dense_hits[key] += int(value)
        for key, value in (item.pool_metrics.get("alias_gold_hits") or {}).items():
            alias_hits[key] += int(value)
        alias_false_positive_top20 += int(item.pool_metrics.get("alias_false_positive_top20") or 0)
        alias_to_concept_coverage += int(item.pool_metrics.get("alias_to_concept_coverage") or 0)
    return {
        "pool_recall": {
            key: _avg([float((item.pool_metrics.get("pool_recall") or {}).get(key, 0.0)) for item in items])
            for key in pool_keys
        },
        "source_gold_hits": {
            source: {key: counter.get(key, 0) for key in pool_keys}
            for source, counter in sorted(source_hits.items())
        },
        "dense_gold_hits": {key: dense_hits.get(key, 0) for key in pool_keys},
        "alias_gold_hits": {key: alias_hits.get(key, 0) for key in pool_keys},
        "alias_false_positive_top20": alias_false_positive_top20,
        "alias_to_concept_coverage": alias_to_concept_coverage,
        "late_gold_count": sum(int(item.pool_metrics.get("late_gold_count") or 0) for item in items),
    }


def summarize_feature_usage(items: list[QueryResult]) -> dict[str, Any]:
    total = max(1, len(items))
    return {
        "rewrite_used": sum(1 for item in items if item.rewrite_used),
        "dense_used": sum(1 for item in items if item.dense_used),
        "alias_used": sum(1 for item in items if item.alias_used),
        "rewrite_used_rate": round(sum(1 for item in items if item.rewrite_used) / total, 6),
        "dense_used_rate": round(sum(1 for item in items if item.dense_used) / total, 6),
        "alias_used_rate": round(sum(1 for item in items if item.alias_used) / total, 6),
    }


def summarize_selector_preselection(items: list[QueryResult]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    query_count = 0
    for item in items:
        item_events = _selector_preselector_events(item.model_events)
        if item_events:
            query_count += 1
            events.extend(item_events)
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for event in events:
        reason_counts.update({str(key): int(value) for key, value in (event.get("reason_counts") or {}).items()})
        source_counts.update({str(key): int(value) for key, value in (event.get("selected_source_counts") or {}).items()})
    return {
        "event_count": len(events),
        "query_count": query_count,
        "avg_input_candidates": _avg([float(event.get("input_candidates") or 0) for event in events]),
        "avg_selected_candidates": _avg([float(event.get("selected_candidates") or 0) for event in events]),
        "avg_target_candidates": _avg([float(event.get("target_candidates") or 0) for event in events]),
        "avg_pool_limit": _avg([float(event.get("pool_limit") or 0) for event in events]),
        "avg_compression_ratio": _avg(
            [
                float(event.get("selected_candidates") or 0) / max(1.0, float(event.get("input_candidates") or 0))
                for event in events
            ]
        ),
        "reason_counts": dict(reason_counts.most_common()),
        "selected_source_counts": dict(sorted(source_counts.items())),
    }


def _selector_preselector_events(model_events: dict[str, Any]) -> list[dict[str, Any]]:
    raw_events = model_events.get("selector_preselector") if isinstance(model_events, dict) else None
    if isinstance(raw_events, dict):
        raw_events = [raw_events]
    if not isinstance(raw_events, list):
        return []
    return [event for event in raw_events if isinstance(event, dict)]


def summarize_query_type_source_hits(items: list[QueryResult]) -> dict[str, Any]:
    summary: dict[str, dict[str, Counter[str]]] = {}
    for item in items:
        query_type = item.query_type or "unknown"
        type_summary = summary.setdefault(query_type, {})
        for source, hits_by_cutoff in (item.pool_metrics.get("source_gold_hits") or {}).items():
            counter = type_summary.setdefault(source, Counter())
            for key, value in hits_by_cutoff.items():
                counter[key] += int(value)
    return {
        query_type: {
            source: dict(counter)
            for source, counter in sorted(source_hits.items())
        }
        for query_type, source_hits in sorted(summary.items())
    }


def build_recommendations(reason_counts: Counter[str], stage_counts: Counter[str], model_error_count: int) -> list[str]:
    recommendations: list[str] = []
    if model_error_count:
        recommendations.append("先修复模型服务错误；错误会让评测混入规则降级结果，影响三模型效果判断。")
    if stage_counts.get("query_understanding_and_retrieval", 0) or stage_counts.get("multi_source_retrieval", 0):
        recommendations.append("优先优化查询理解和多源召回：扩展 SearchPlanner 子查询、ES 字段权重/同义词、chunk 召回和 Qdrant 覆盖。")
    if stage_counts.get("selector_reranker_or_rule_ranker", 0):
        recommendations.append("重新训练或校准 Selector Reranker：加入 hard negative、同主题近邻负样本，并检查 CrossEncoder 分数是否压低 gold。")
    if reason_counts.get("large_gold_set_topk_ceiling", 0):
        recommendations.append("RealScholarQuery 多答案场景需要关注 Recall@50/@100；Recall@10 常被 gold 数量上限压低。")
    if stage_counts.get("query_intent", 0):
        recommendations.append("重新检查 Query Intent gate 的 false negative，避免论文检索请求被误判为 non_paper_search。")
    if not recommendations:
        recommendations.append("当前主要问题不是硬故障，继续扩大样本并做 reranker/召回消融比较。")
    recommendations.append("Crawler Strategy 只影响 top 论文后续章节展开元数据，不是当前 Recall/F1 的首要优化点。")
    return recommendations


def render_markdown(report: dict[str, Any]) -> str:
    cutoffs = report["config"]["cutoffs"]
    lines = [
        "# ScholarSearchAgent Database Evaluation Report",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Configuration",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Datasets | {', '.join(report['config']['datasets'])} |",
        f"| MySQL | `{report['config']['mysql_host']}/{report['config']['mysql_database']}` |",
        f"| Elasticsearch | `{report['config']['elasticsearch_url']}` |",
        f"| ES indices | `{report['config']['papers_index']}`, `{report['config']['chunks_index']}` |",
        f"| Qdrant | `{report['config']['qdrant_url']}` / `{report['config']['qdrant_collection']}` |",
        f"| Model services | {report['config']['model_services']} |",
        f"| Cutoffs | {', '.join('@' + str(item) for item in cutoffs)} |",
        f"| Pool cutoffs | {', '.join(report['diagnostics']['pool_recall'].keys())} |",
        f"| Diagnostic top K | {report['config']['diagnostic_top_k']} |",
        f"| Output paper limit / query | {report['config']['output_paper_limit']} |",
        "",
        "## Runtime",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Requested queries | {report['data']['requested_queries']} |",
        f"| Evaluated queries | {report['data']['evaluated_queries']} |",
        f"| Failed queries | {report['data']['failed_queries']} |",
        f"| Gold labels | {report['data']['gold_labels']} |",
        f"| Total elapsed sec | {report['runtime']['elapsed_sec']} |",
        f"| Avg latency sec | {report['runtime']['avg_latency_sec']} |",
        "",
        "## Overall Metrics",
        "",
    ]
    lines.extend(metric_table(report["summary"], cutoffs))
    lines.extend(["", "## Low Recall Diagnosis", ""])
    lines.extend(diagnostic_section(report))
    lines.extend(["", "## Metrics by Dataset", ""])
    for name, summary in report["by_dataset"].items():
        lines.extend([f"### {name}", ""])
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Query count | {summary['query_count']} |",
                f"| Gold labels | {summary['gold_labels']} |",
                f"| Avg gold / query | {summary['avg_gold_per_query']} |",
                f"| Avg latency sec | {summary['avg_latency_sec']} |",
                "",
            ]
        )
        lines.extend(metric_table(summary, cutoffs))
        lines.append("")
    lines.extend(
        [
            "## Metric Notes",
            "",
            "- Precision@K uses a fixed denominator K.",
            "- Recall@K uses each query's gold paper set from `eval_sets.gold_paper_ids`.",
            "- F1@K is the harmonic mean of Precision@K and Recall@K.",
            "- MRR@K is the reciprocal rank of the first hit within top-K, macro-averaged.",
            "- HitRate@K is the query ratio with at least one hit in top-K.",
            "- The full JSON report stores per-query `retrieved_papers`, gold titles, model events, and stage diagnostics.",
        ]
    )
    return "\n".join(lines) + "\n"


def diagnostic_section(report: dict[str, Any]) -> list[str]:
    diagnostics = report["diagnostics"]
    lines = [
        "| Item | Value |",
        "| --- | --- |",
        f"| Diagnosis cutoff | @{diagnostics['cutoff_used_for_diagnosis']} |",
        f"| Zero-hit queries | {diagnostics['zero_hit_queries']} ({diagnostics['zero_hit_rate']:.6f}) |",
        f"| Avg Recall ceiling at K | {diagnostics['avg_topk_recall_ceiling']:.6f} |",
        f"| Late gold count | {diagnostics['late_gold_count']} |",
        f"| Model service errors | {diagnostics['model_error_count']} |",
        "",
        "### Pool Recall",
        "",
        "| Pool | Recall |",
        "| --- | ---: |",
    ]
    for key, value in diagnostics.get("pool_recall", {}).items():
        lines.append(f"| {key} | {value:.6f} |")
    lines.extend(
        [
            "",
            "### Source Gold Hits",
            "",
            "| Source | " + " | ".join(diagnostics.get("pool_recall", {}).keys()) + " |",
            "| --- | " + " | ".join("---:" for _ in diagnostics.get("pool_recall", {})) + " |",
        ]
    )
    for source, row in diagnostics.get("source_gold_hits", {}).items():
        values = [str(row.get(key, 0)) for key in diagnostics.get("pool_recall", {})]
        lines.append(f"| {source} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "### Dense / Alias Contribution",
            "",
            "| Signal | " + " | ".join(diagnostics.get("pool_recall", {}).keys()) + " |",
            "| --- | " + " | ".join("---:" for _ in diagnostics.get("pool_recall", {})) + " |",
        ]
    )
    for signal, row in (
        ("dense_gold_hits", diagnostics.get("dense_gold_hits", {})),
        ("alias_gold_hits", diagnostics.get("alias_gold_hits", {})),
    ):
        values = [str(row.get(key, 0)) for key in diagnostics.get("pool_recall", {})]
        lines.append(f"| {signal} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "| Alias diagnostic | Value |",
            "| --- | ---: |",
            f"| alias_false_positive_top20 | {diagnostics.get('alias_false_positive_top20', 0)} |",
            f"| alias_to_concept_coverage | {diagnostics.get('alias_to_concept_coverage', 0)} |",
            "",
            "### Query Type / Feature Usage",
            "",
            "| Feature | Count | Rate |",
            "| --- | ---: | ---: |",
        ]
    )
    feature_usage = diagnostics.get("feature_usage", {})
    for feature in ("rewrite_used", "dense_used", "alias_used"):
        lines.append(
            f"| {feature} | {feature_usage.get(feature, 0)} | {feature_usage.get(feature + '_rate', 0.0):.6f} |"
        )
    selector_preselector = diagnostics.get("selector_preselector") or {}
    if selector_preselector.get("event_count"):
        lines.extend(
            [
                "",
                "### Selector Preselection",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Event count | {selector_preselector.get('event_count', 0)} |",
                f"| Query count | {selector_preselector.get('query_count', 0)} |",
                f"| Avg input candidates | {selector_preselector.get('avg_input_candidates', 0.0):.6f} |",
                f"| Avg selected candidates | {selector_preselector.get('avg_selected_candidates', 0.0):.6f} |",
                f"| Avg target candidates | {selector_preselector.get('avg_target_candidates', 0.0):.6f} |",
                f"| Avg pool limit | {selector_preselector.get('avg_pool_limit', 0.0):.6f} |",
                f"| Avg compression ratio | {selector_preselector.get('avg_compression_ratio', 0.0):.6f} |",
                "",
                "| Reason | Count |",
                "| --- | ---: |",
            ]
        )
        for reason, count in (selector_preselector.get("reason_counts") or {}).items():
            lines.append(f"| {reason} | {count} |")
    lines.extend(["", "| Query type | Queries |", "| --- | ---: |"])
    for query_type, count in diagnostics.get("query_type_counts", {}).items():
        lines.append(f"| {query_type} | {count} |")
    lines.extend(
        [
            "",
        "### Priority Stages",
        "",
        "| Stage | Queries |",
        "| --- | ---: |",
        ]
    )
    for stage, count in diagnostics["priority_stage_counts"].items():
        lines.append(f"| {stage} | {count} |")
    lines.extend(["", "### Primary Reasons", "", "| Reason | Queries |", "| --- | ---: |"])
    for reason, count in diagnostics["primary_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "### Recommendations", ""])
    for item in diagnostics["recommendations"]:
        lines.append(f"- {item}")
    lines.extend(["", "### Worst Query Examples", ""])
    lines.extend(worst_query_examples(report))
    return lines


def worst_query_examples(report: dict[str, Any], limit: int = 8) -> list[str]:
    cutoffs = report["config"]["cutoffs"]
    max_key = f"@{max(cutoffs)}"
    examples = sorted(
        report["per_query"],
        key=lambda item: (
            float(item["metrics"][max_key]["recall"]),
            -item["gold_count"],
            item["latency_sec"],
        ),
    )[:limit]
    lines: list[str] = []
    for item in examples:
        metric = item["metrics"][max_key]
        lines.extend(
            [
                f"#### {item['dataset_name']}/{item['split_name']} `{item['qid']}`",
                "",
                f"- Query: {item['query']}",
                f"- Hits/Gold: {metric['hits']}/{item['gold_count']}; Recall={metric['recall']:.6f}; F1={metric['f1']:.6f}",
                f"- Diagnosis: `{item['diagnostics']['primary_reason']}`; priority stage: `{item['diagnostics']['priority_stage']}`",
                "- Top retrieved papers:",
            ]
        )
        for paper in item["retrieved_papers"][:5]:
            title = str(paper.get("title") or "").replace("\n", " ")
            lines.append(f"  - #{paper['rank']} `{paper['paper_id']}` {title}")
        missed = item["diagnostics"].get("missed_gold_ids") or []
        if missed:
            lines.append(f"- Missed gold ids: {', '.join(missed[:8])}")
        lines.append("")
    return lines


def metric_table(summary: dict[str, Any], cutoffs: list[int]) -> list[str]:
    lines = [
        "| K | Precision@K | Recall@K | F1@K | MRR@K | HitRate@K | Hits |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cutoff in cutoffs:
        row = summary[f"@{cutoff}"]
        lines.append(
            f"| {cutoff} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {row['mrr']:.6f} | {row['hit_rate']:.6f} | {row['hits']} |"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ScholarSearchAgent using configured database test data.")
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data_ingestion_indexing" / "data_processed"))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--json-output", default="db_eval_results.json")
    parser.add_argument("--markdown-output", default="db_eval_report.md")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="dataset:split pairs")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--pool-cutoffs", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--diagnostic-top-k", type=int, default=100)
    parser.add_argument("--output-paper-limit", type=int, default=20)
    parser.add_argument("--per-query-top-k", type=int, default=60)
    parser.add_argument("--max-chunks-per-paper", type=int, default=4)
    parser.add_argument("--max-queries-per-dataset", type=int, default=None)
    parser.add_argument("--max-total-queries", type=int, default=None)
    parser.add_argument("--sample-profile", choices=["head", "balanced", "proportional"], default="head")
    parser.add_argument("--sample-order", choices=["qid", "hash"], default="hash")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--disable-model-services", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(args)
    print(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "data": report["data"],
                "runtime": report["runtime"],
                "summary": report["summary"],
                "by_dataset": report["by_dataset"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
