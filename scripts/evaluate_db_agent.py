"""Evaluate ScholarSearchAgent against test labels stored in the configured MySQL DB."""

from __future__ import annotations

import argparse
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
    metrics: dict[str, dict[str, float | int]]
    diagnostics: dict[str, Any]
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with MySQLClient.from_settings(settings) as mysql:
        mysql.use_database(settings.mysql_database)
        for spec in dataset_specs:
            dataset_name, split_name = spec.split(":", 1)
            sql = (
                "SELECT q.dataset_name, q.split_name, q.qid, q.query_text, e.gold_paper_ids "
                "FROM queries q JOIN eval_sets e ON q.qid=e.qid "
                f"WHERE q.dataset_name={sql_value(dataset_name)} AND q.split_name={sql_value(split_name)} "
                "ORDER BY q.qid"
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
    diagnostic_top_k = max(top_k, int(args.diagnostic_top_k or top_k))
    rows = load_eval_rows(
        settings,
        args.datasets,
        max_queries_per_dataset=args.max_queries_per_dataset,
    )
    gold_paper_details = load_paper_details(settings, _iter_gold_ids(rows))
    if args.progress_every >= 0:
        print(
            f"[setup] loaded_eval_rows={len(rows)} gold_detail_rows={len(gold_paper_details)}",
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
        gold = list(row["gold_ids"])
        gold_hit_ranks = _gold_hit_ranks(predicted, gold)
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
            response_cost=response.cost,
            model_events=response.cost.get("model_services") or {},
            coverage_missing=response.coverage.missing_constraints,
            parsed_sub_queries=response.parsed_query.sub_queries,
            metrics=metrics,
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
                metrics=metrics,
                diagnostics=diagnostics,
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
            "top_k": top_k,
            "diagnostic_top_k": diagnostic_top_k,
            "output_paper_limit": args.output_paper_limit,
            "per_query_top_k": args.per_query_top_k,
            "model_services": "disabled" if args.disable_model_services else "config-default",
            "max_queries_per_dataset": args.max_queries_per_dataset,
        },
        "data": {
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
        "backend_fallback_reason",
    )
    return {key: cost.get(key) for key in keys if key in cost}


def diagnose_query(
    *,
    gold_ids: list[str],
    predicted_ids: list[str],
    gold_hit_ranks: dict[str, int],
    response_cost: dict[str, Any],
    model_events: dict[str, Any],
    coverage_missing: list[str],
    parsed_sub_queries: list[str],
    metrics: dict[str, dict[str, float | int]],
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

    # Diagnosis is stage-oriented: first find hard failures, then separate
    # retrieval misses from ranking/cutoff misses and top-K ceiling effects.
    if not parsed_sub_queries:
        reason = "query_intent_blocked_or_empty_plan"
        stage = "query_intent"
    elif raw_candidates == 0 or unique_candidates == 0 or not predicted_ids:
        reason = "retrieval_empty"
        stage = "multi_source_retrieval"
    elif hits_at_max == 0 and late_hits:
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
        "late_gold_hits": late_hits,
        "missed_gold_count": len(missed_gold),
        "missed_gold_ids": missed_gold[:20],
        "topk_recall_ceiling": round(topk_recall_ceiling, 6),
        "diagnostic_pool_hit_count": len(gold_hit_ranks),
        "raw_candidates": raw_candidates,
        "unique_candidates": unique_candidates,
        "model_errors": model_errors,
        "coverage_missing": coverage_missing,
    }


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
    return {
        "cutoff_used_for_diagnosis": max_cutoff,
        "primary_reason_counts": dict(reason_counts.most_common()),
        "priority_stage_counts": dict(stage_counts.most_common()),
        "model_error_count": model_error_count,
        "zero_hit_queries": len(no_hit_items),
        "zero_hit_rate": _avg([1.0 if item in no_hit_items else 0.0 for item in items]),
        "avg_topk_recall_ceiling": _avg([item.diagnostics["topk_recall_ceiling"] for item in items]),
        "recommendations": build_recommendations(reason_counts, stage_counts, model_error_count),
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
        f"| Model service errors | {diagnostics['model_error_count']} |",
        "",
        "### Priority Stages",
        "",
        "| Stage | Queries |",
        "| --- | ---: |",
    ]
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
    parser.add_argument("--diagnostic-top-k", type=int, default=100)
    parser.add_argument("--output-paper-limit", type=int, default=20)
    parser.add_argument("--per-query-top-k", type=int, default=60)
    parser.add_argument("--max-chunks-per-paper", type=int, default=4)
    parser.add_argument("--max-queries-per-dataset", type=int, default=None)
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
