"""Run offline retrieval evaluation for PaSa test queries using local papers only.

This script avoids network dependency and evaluates the current agent retrieval
logic with a lightweight text baseline over local title metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from math import log
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.scholar_core.text import cosine_sparse, normalize_title, token_counter  # noqa: E402
from packages.scholar_eval.evaluation import score_prediction  # noqa: E402
from packages.scholar_infra.io.jsonl import read_jsonl, write_json  # noqa: E402


DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "数据集" / "pasa" / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_SPLITS = ("AutoScholarQuery/test", "RealScholarQuery/test")


@dataclass(frozen=True)
class QueryResult:
    dataset: str
    split: str
    qid: str
    query: str
    gold_count: int
    predicted_count: int
    latency_sec: float
    top_predictions: list[str]
    gold_ids: list[str]
    predicted_ids: list[str]
    metrics: dict[str, dict[str, float | int]]


def normalize_arxiv_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    cleaned = raw.replace("arXiv:", "").replace("arxiv:", "").strip()
    return cleaned.split("v", 1)[0]


def paper_id_from_raw(value: Any) -> str:
    norm = normalize_arxiv_id(value)
    return f"arxiv:{norm}" if norm else ""


def _read_id2paper(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    result: dict[str, str] = {}
    for arxiv_id, title in raw.items():
        pid = paper_id_from_raw(arxiv_id)
        if pid:
            result[pid] = str(title or "")
    return result


def build_local_index(papers: dict[str, str]) -> dict[str, Counter[str]]:
    return {pid: token_counter(normalize_title(title), keep_stopwords=True) for pid, title in papers.items()}


def build_query_cache(papers: dict[str, str]) -> dict[str, Counter[str]]:
    # Query-side tf counter, includes stop words for consistent overlap in formulas.
    return {pid: token_counter(title, keep_stopwords=True) for pid, title in papers.items()}


def build_inverted_index(index: dict[str, Counter[str]]) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    doc_count = max(1, len(index))
    df: dict[str, int] = defaultdict(int)
    for counter in index.values():
        for token in counter:
            df[token] += 1
    idf: dict[str, float] = {token: log(1 + doc_count / (1 + freq)) for token, freq in df.items()}
    inv: dict[str, dict[str, int]] = defaultdict(dict)
    for pid, counter in index.items():
        for token, freq in counter.items():
            inv[token][pid] = freq
    return inv, idf


def rank_local(
    query_tokens: list[str],
    query_counter: Counter[str],
    paper_index: dict[str, Counter[str]],
    inv: dict[str, dict[str, int]],
    idf: dict[str, float],
    *,
    top_k: int,
) -> list[str]:
    candidate_scores: dict[str, float] = defaultdict(float)
    unique_terms = set(query_tokens)
    if not unique_terms:
        return []

    for term in unique_terms:
        posting = inv.get(term, {})
        term_idf = idf.get(term, 0.0)
        qf = query_counter.get(term, 0) or 1
        for pid, tf in posting.items():
            candidate_scores[pid] += (tf * qf * term_idf)

    if not candidate_scores:
        # Fallback to cosine over small candidate set by random tie baseline.
        return []

    # Length-normalized ranking via cosine to reduce keyword inflation.
    # Keep top few thousand for stable runtime.
    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[: min(5000, len(candidate_scores))]
    query_vec = Counter({token: query_counter.get(token, 0) for token in unique_terms})
    scored: list[tuple[str, float]] = []
    for pid, _ in ranked:
        scored.append((pid, cosine_sparse(query_vec, paper_index[pid]) + 0.01 * _))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [pid for pid, _ in scored[:top_k]]


def iter_pasa_queries(pasa_root: Path, splits: list[str], max_queries: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_spec in splits:
        dataset, split = split_spec.split("/", 1)
        file_path = pasa_root / dataset / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"missing paas query file: {file_path}")
        seen = 0
        for obj in read_jsonl(file_path):
            if max_queries is not None and seen >= max_queries:
                break
            seen += 1
            qid = str(obj.get("qid") or f"{dataset}_{split}_{seen - 1}")
            gold = [paper_id_from_raw(v) for v in (obj.get("answer_arxiv_id") or [])]
            gold = [v for v in gold if v]
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "qid": qid,
                    "query": str(obj.get("question") or ""),
                    "gold": gold,
                }
            )
    return rows


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(mean(values), 6)


def summarize_group(items: list[QueryResult], cutoffs: list[int]) -> dict[str, Any]:
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.pasa_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    ids_path = data_root / "paper_database" / "id2paper.json"
    if not ids_path.exists():
        raise FileNotFoundError(f"missing id2paper file: {ids_path}")
    paper_ids = _read_id2paper(ids_path)
    paper_titles = {pid: normalize_title(title) for pid, title in paper_ids.items()}
    paper_index = build_local_index(paper_ids)
    inv, idf = build_inverted_index(paper_index)

    cutoffs = sorted(set(int(x) for x in args.cutoffs))
    max_k = max(cutoffs)
    all_queries = iter_pasa_queries(data_root, args.splits, args.max_queries_per_split)

    per_query: list[QueryResult] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for idx, row in enumerate(all_queries, start=1):
        qid = row["qid"]
        query = str(row["query"] or "")
        if not query.strip():
            failures.append({"qid": qid, "error": "empty query"})
            continue
        q_counter = token_counter(normalize_title(query))
        q_tokens = list(q_counter.keys())
        q_started = time.perf_counter()

        predicted_ids = []
        try:
            predicted_ids = rank_local(q_tokens, q_counter, paper_index, inv, idf, top_k=max_k)
        except Exception as exc:  # keep evaluator robust.
            failures.append({"qid": qid, "error": str(exc)})
            continue
        latency = round(time.perf_counter() - q_started, 4)

        gold = list(row["gold"])
        metrics: dict[str, dict[str, float | int]] = {}
        for cutoff in cutoffs:
            scored = score_prediction(predicted_ids, gold, k=cutoff)
            metrics[f"@{cutoff}"] = {
                "hits": scored.hits,
                "precision": scored.precision_at_k,
                "recall": scored.recall_at_k,
                "f1": scored.f1_at_k,
                "mrr": scored.mrr,
                "hit_rate": 1.0 if scored.hits > 0 else 0.0,
            }

        per_query.append(
            QueryResult(
                dataset=row["dataset"],
                split=row["split"],
                qid=qid,
                query=query,
                gold_count=len(gold),
                predicted_count=len(predicted_ids),
                latency_sec=latency,
                top_predictions=predicted_ids[: min(10, len(predicted_ids))],
                gold_ids=gold,
                predicted_ids=predicted_ids,
                metrics=metrics,
            )
        )
        if args.progress_every > 0 and idx % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[progress] processed={idx}/{len(all_queries)} elapsed={elapsed:.2f}s"
                f" avg_latency={(elapsed / idx):.3f}s"
            )

    elapsed = time.perf_counter() - started

    by_dataset: dict[str, list[QueryResult]] = {}
    for item in per_query:
        by_dataset.setdefault(f"{item.dataset}/{item.split}", []).append(item)
    overall_summary = summarize_group(per_query, cutoffs)
    by_dataset_summary = {name: summarize_group(items, cutoffs) for name, items in by_dataset.items()}
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ScholarSearchAgent PaSa offline local-title evaluation",
        "config": {
            "pasa_root": str(data_root),
            "cutoffs": cutoffs,
            "splits": list(args.splits),
            "per_query_top_k": max_k,
            "max_queries_per_split": args.max_queries_per_split,
            "paper_count": len(paper_ids),
            "index_type": "title token overlap + cosine rerank",
        },
        "data": {
            "requested_queries": len(all_queries),
            "evaluated_queries": len(per_query),
            "failed_queries": len(failures),
            "failures": failures[:200],
            "total_gold_labels": sum(item.gold_count for item in per_query),
        },
        "runtime": {
            "elapsed_sec": round(elapsed, 3),
            "avg_latency_sec": _avg([item.latency_sec for item in per_query]),
            "backend_observed": "local_title_index",
        },
        "summary": overall_summary,
        "by_dataset": by_dataset_summary,
        "per_query": [asdict(item) for item in per_query],
    }

    json_path = report_dir / args.json_output
    md_path = report_dir / args.markdown_output
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    cutoffs: list[int] = report["config"]["cutoffs"]
    lines: list[str] = [
        "# ScholarSearchAgent PaSa Offline Evaluation Report",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## 1. Configuration",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Dataset | {', '.join(report['config']['splits'])} |",
        f"| Data root | `{report['config']['pasa_root']}` |",
        f"| Index | Local title index (`paper_database/id2paper.json`) |",
        f"| Candidate pool | PaSa paper list (title-only baseline) |",
        f"| Cutoffs | {', '.join('@'+str(x) for x in cutoffs)} |",
        f"| Per-query top K used | {report['config']['per_query_top_k']} |",
        f"| Max queries per split | {report['config']['max_queries_per_split']} |",
        "",
        "## 2. Runtime",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Requested queries | {report['data']['requested_queries']} |",
        f"| Evaluated queries | {report['data']['evaluated_queries']} |",
        f"| Failed queries | {report['data']['failed_queries']} |",
        f"| Gold labels total | {report['data']['total_gold_labels']} |",
        f"| Total elapsed (s) | {report['runtime']['elapsed_sec']} |",
        f"| Avg latency per query (s) | {report['runtime']['avg_latency_sec']} |",
        "",
        "## 3. Overall Results",
        "",
    ]
    lines.extend(build_markdown_metric_table(report["summary"], cutoffs))
    lines.extend(["", "## 4. Results by dataset", ""])
    for name, summary in report["by_dataset"].items():
        lines.extend([f"### {name}", ""])
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Query count | {summary['query_count']} |",
                f"| Gold labels | {summary['gold_labels']} |",
                f"| Avg gold / query | {summary['avg_gold_per_query']} |",
                f"| Avg latency / query (s) | {summary['avg_latency_sec']} |",
                "",
            ]
        )
        lines.extend(build_markdown_metric_table(summary, cutoffs))
        lines.append("")
    lines.extend(
        [
            "## 5. Metric definition",
            "",
            "- Precision@K: share of gold papers in top-K predictions.",
            "- Recall@K: share of gold papers retrieved in top-K predictions.",
            "- F1@K: harmonic mean of Precision@K and Recall@K.",
            "- MRR: average reciprocal rank of first relevant paper within top-K.",
            "- HitRate@K: fraction of queries with at least one hit in top-K.",
            "",
            "## 6. Notes",
            "",
            "- This is a title-only local baseline and does not use database/ES/Qdrant.",
            "- It is suitable for offline reproducibility in restricted network environments.",
            "- Gold set comes from PaSa `answer_arxiv_id`; mismatched ID formats are normalized.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_markdown_metric_table(summary: dict[str, Any], cutoffs: list[int]) -> list[str]:
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
    parser = argparse.ArgumentParser(description="Evaluate ScholarSearchAgent offline using local PaSa data.")
    parser.add_argument("--pasa-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--json-output", default="pasa_eval_local.json")
    parser.add_argument("--markdown-output", default="pasa_eval_local.md")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--max-queries-per-split", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = evaluate(args)
    print(
        json.dumps(
            {"generated_at": report["generated_at"], "data": report["data"], "runtime": report["runtime"], "summary": report["summary"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
