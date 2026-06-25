# 中文功能说明：后端 API 服务入口，基于 Python 标准库 HTTPServer 提供健康检查和检索接口。

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from apps.backend.scholar_api.api.routes.health import health_response
from apps.backend.scholar_api.api.routes.search import search_response
from apps.backend.scholar_api.api.schemas.search import parse_search_query
from apps.backend.scholar_api.bootstrap.container import build_search_pipeline
from packages.scholar_core.pipeline import SearchPipeline
from packages.scholar_core.composition.composer import ResultComposer


class ApiServer:
    def __init__(self, pipeline: SearchPipeline, *, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self.composer = ResultComposer()

    def serve_forever(self) -> None:
        pipeline = self.pipeline
        composer = self.composer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/health"}:
                    self._send_json(health_response())
                    return
                if parsed.path == "/api/search":
                    self._handle_search(parsed.query)
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            def do_OPTIONS(self) -> None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self._send_cors_headers()
                self.end_headers()

            def log_message(self, format: str, *args) -> None:
                return

            def _handle_search(self, raw_query: str) -> None:
                query, top_k = parse_search_query(raw_query)
                if not query:
                    self._send_json({"error": "query is required"}, HTTPStatus.BAD_REQUEST)
                    return
                try:
                    payload = search_response(pipeline, composer, query, top_k)
                except Exception as exc:
                    self._send_json(
                        {"error": "search failed", "detail": str(exc)},
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return
                self._send_json(payload)

            def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_cors_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"ScholarSearch API server running at http://{self.host}:{self.port}")
        server.serve_forever()


def run_server(
    processed_dir: Path,
    *,
    host: str,
    port: int,
    paper_limit: int | None,
    chunk_limit: int | None,
    max_chunks_per_paper: int,
    per_query_top_k: int,
    backend: str = "auto",
    model_services_enabled: bool | None = None,
) -> None:
    pipeline = build_search_pipeline(
        processed_dir,
        paper_limit=paper_limit,
        chunk_limit=chunk_limit,
        max_chunks_per_paper=max_chunks_per_paper,
        per_query_top_k=per_query_top_k,
        backend=backend,
        model_services_enabled=model_services_enabled,
    )
    ApiServer(pipeline, host=host, port=port).serve_forever()
