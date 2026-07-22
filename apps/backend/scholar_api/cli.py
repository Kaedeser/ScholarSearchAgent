# 中文功能说明：命令行入口，负责构建检索流水线并启动 API 服务、检索命令和评测命令。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps.backend.scholar_api.bootstrap.container import build_search_pipeline
from .main import run_server
from packages.scholar_core.pipeline import SearchPipeline
from packages.scholar_eval.evaluation import Evaluator
from packages.scholar_core.composition.composer import ResultComposer
from packages.scholar_infra.io.jsonl import default_processed_dir, write_json, write_text


def build_pipeline(args: argparse.Namespace) -> SearchPipeline:
    return build_search_pipeline(
        Path(args.processed_dir),
        paper_limit=args.paper_limit,
        chunk_limit=args.chunk_limit,
        max_chunks_per_paper=args.max_chunks_per_paper,
        per_query_top_k=args.per_query_top_k,
        backend=args.backend,
        model_services_enabled=False if args.disable_model_services else None,
    )


def cmd_search(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(args)
    composer = ResultComposer()
    response = pipeline.search(args.query, top_k=args.top_k)
    jsonable = composer.to_jsonable(response)
    markdown = composer.to_markdown(response)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        write_json(output_dir / "result.json", jsonable)
        write_text(output_dir / "result.md", markdown)
        write_text(output_dir / "result.bib", composer.to_bibtex(response.papers))
    if args.format == "markdown":
        print(markdown)
    else:
        print(json.dumps(jsonable, ensure_ascii=False, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(args)
    evaluator = Evaluator(Path(args.processed_dir), pipeline)
    report = evaluator.evaluate(split=args.split, max_queries=args.max_queries, top_k=args.top_k)
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    run_server(
        Path(args.processed_dir),
        host=args.host,
        port=args.port,
        paper_limit=args.paper_limit,
        chunk_limit=args.chunk_limit,
        max_chunks_per_paper=args.max_chunks_per_paper,
        per_query_top_k=args.per_query_top_k,
        backend=args.backend,
        model_services_enabled=False if args.disable_model_services else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ScholarSearch-Agent demo")
    parser.add_argument("--processed-dir", default=str(default_processed_dir()))
    parser.add_argument("--paper-limit", type=int, default=None)
    parser.add_argument("--chunk-limit", type=int, default=None)
    parser.add_argument("--max-chunks-per-paper", type=int, default=4)
    parser.add_argument("--per-query-top-k", type=int, default=60)
    parser.add_argument(
        "--backend",
        choices=("auto", "jsonl", "database", "semantic_scholar"),
        default="auto",
    )
    parser.add_argument("--disable-model-services", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Run one query through the demo pipeline")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--format", choices=("json", "markdown"), default="json")
    search.add_argument("--output-dir", default=None)
    search.set_defaults(func=cmd_search)

    evaluate = subparsers.add_parser("eval", help="Evaluate against PaSa eval_sets")
    evaluate.add_argument("--split", default=None)
    evaluate.add_argument("--max-queries", type=int, default=None)
    evaluate.add_argument("--top-k", type=int, default=20)
    evaluate.add_argument("--output", default=None)
    evaluate.set_defaults(func=cmd_eval)

    serve = subparsers.add_parser("serve", help="Start the JSON API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
