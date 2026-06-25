# 中文功能说明：离线评测模块，基于 PaSa 标注计算 Precision、Recall、F1 和 MRR。

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

from packages.scholar_core.pipeline import SearchPipeline
from packages.scholar_infra.io.jsonl import load_eval_sets, load_queries


@dataclass(frozen=True)
class QueryMetrics:
    qid: str
    hits: int
    gold_count: int
    predicted_count: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    mrr: float
    latency_sec: float


def score_prediction(predicted_ids: list[str], gold_ids: list[str], *, k: int) -> QueryMetrics:
    predicted_top = predicted_ids[:k]
    gold = set(gold_ids)
    hits = sum(1 for paper_id in predicted_top if paper_id in gold)
    precision = hits / k if k else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    mrr = 0.0
    for rank, paper_id in enumerate(predicted_top, start=1):
        if paper_id in gold:
            mrr = 1 / rank
            break
    return QueryMetrics(
        qid="",
        hits=hits,
        gold_count=len(gold),
        predicted_count=len(predicted_top),
        precision_at_k=precision,
        recall_at_k=recall,
        f1_at_k=f1,
        mrr=mrr,
        latency_sec=0.0,
    )


class Evaluator:
    def __init__(self, processed_dir: Path, pipeline: SearchPipeline) -> None:
        self.processed_dir = processed_dir
        self.pipeline = pipeline

    def evaluate(self, *, split: str | None = None, max_queries: int | None = None, top_k: int = 20) -> dict[str, Any]:
        query_rows = {row["qid"]: row for row in load_queries(self.processed_dir, split=split)}
        eval_rows = load_eval_sets(self.processed_dir, split=split, limit=max_queries)
        per_query: list[QueryMetrics] = []
        for row in eval_rows:
            qid = row["qid"]
            query_text = str(query_rows.get(qid, {}).get("query_text") or "")
            if not query_text:
                continue
            response = self.pipeline.search(query_text, top_k=top_k)
            predicted = [candidate.canonical_id or candidate.paper_id for candidate in response.papers]
            metrics = score_prediction(predicted, list(row.get("gold_paper_ids") or []), k=top_k)
            per_query.append(
                QueryMetrics(
                    qid=qid,
                    hits=metrics.hits,
                    gold_count=metrics.gold_count,
                    predicted_count=metrics.predicted_count,
                    precision_at_k=metrics.precision_at_k,
                    recall_at_k=metrics.recall_at_k,
                    f1_at_k=metrics.f1_at_k,
                    mrr=metrics.mrr,
                    latency_sec=float(response.cost["latency_sec"]),
                )
            )
        return {
            "split": split or "all",
            "top_k": top_k,
            "query_count": len(per_query),
            "macro_precision_at_k": _avg(item.precision_at_k for item in per_query),
            "macro_recall_at_k": _avg(item.recall_at_k for item in per_query),
            "macro_f1_at_k": _avg(item.f1_at_k for item in per_query),
            "mrr": _avg(item.mrr for item in per_query),
            "avg_latency_sec": _avg(item.latency_sec for item in per_query),
            "per_query": [asdict(item) for item in per_query],
        }


def _avg(values) -> float:
    materialized = list(values)
    return round(mean(materialized), 6) if materialized else 0.0
