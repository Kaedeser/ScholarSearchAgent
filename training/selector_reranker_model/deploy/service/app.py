# 中文功能说明：Selector Reranker 模型 HTTP 服务，加载 CrossEncoder 并提供打分与重排接口。

from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LOGGER = logging.getLogger("selector-reranker-service")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/selector-reranker"))
PORT = int(os.environ.get("PORT", "8000"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.0006931035313755274"))
USE_FP16 = os.environ.get("USE_FP16", "false").lower() in {"1", "true", "yes"}

TOKENIZER: Any | None = None
MODEL: Any | None = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS: dict[str, Any] = {}


def load_metrics() -> dict[str, Any]:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model() -> None:
    global TOKENIZER, MODEL, METRICS
    LOGGER.info("Loading model from %s on %s", MODEL_DIR, DEVICE)
    METRICS = load_metrics()
    TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    MODEL = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
    MODEL.to(DEVICE)
    if USE_FP16 and DEVICE.type == "cuda":
        MODEL.half()
    MODEL.eval()
    LOGGER.info("Model loaded. fp16=%s threshold=%s max_length=%s", USE_FP16, THRESHOLD, MAX_LENGTH)


def document_text(document: Any) -> str:
    if isinstance(document, str):
        return document
    if not isinstance(document, dict):
        return str(document)
    if document.get("text"):
        return str(document["text"])
    title = str(document.get("title", "")).strip()
    abstract = str(document.get("abstract", "")).strip()
    if title or abstract:
        return f"Title: {title}\nAbstract: {abstract}"
    return json.dumps(document, ensure_ascii=False)


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    if TOKENIZER is None or MODEL is None:
        raise RuntimeError("Model is not loaded")
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start : start + BATCH_SIZE]
            encoded = TOKENIZER(
                [query for query, _ in batch],
                [doc for _, doc in batch],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = MODEL(**encoded).logits.squeeze(-1)
            batch_scores = torch.sigmoid(logits).detach().cpu().float().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(float(score) for score in batch_scores)
    return scores


def json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def parse_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    return json.loads(raw.decode("utf-8"))


class Handler(BaseHTTPRequestHandler):
    server_version = "SelectorReranker/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/health", "/ready"}:
            json_response(
                self,
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_loaded": MODEL is not None,
                    "device": str(DEVICE),
                    "model_dir": str(MODEL_DIR),
                    "threshold": THRESHOLD,
                    "max_length": MAX_LENGTH,
                },
            )
            return
        if self.path == "/metrics":
            json_response(self, HTTPStatus.OK, {"metrics": METRICS})
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/score":
                payload = parse_json(self)
                if "pairs" in payload:
                    pairs = [
                        (str(item["query"]), document_text(item.get("document", "")))
                        for item in payload.get("pairs", [])
                    ]
                else:
                    pairs = [(str(payload["query"]), document_text(payload.get("document", "")))]
                scores = score_pairs(pairs)
                json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "scores": scores,
                        "threshold": THRESHOLD,
                        "relevant": [score >= THRESHOLD for score in scores],
                    },
                )
                return
            if self.path == "/rerank":
                payload = parse_json(self)
                query = str(payload["query"])
                documents = payload.get("documents", [])
                top_k = int(payload.get("top_k", len(documents)))
                pairs = [(query, document_text(document)) for document in documents]
                scores = score_pairs(pairs)
                ranked = []
                for index, (document, score) in enumerate(zip(documents, scores)):
                    item = dict(document) if isinstance(document, dict) else {"text": str(document)}
                    item["index"] = index
                    item["score"] = score
                    item["relevant"] = score >= THRESHOLD
                    ranked.append(item)
                ranked.sort(key=lambda item: item["score"], reverse=True)
                json_response(
                    self,
                    HTTPStatus.OK,
                    {"results": ranked[:top_k], "threshold": THRESHOLD, "count": len(ranked)},
                )
                return
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except KeyError as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"missing field: {exc}"})
        except Exception as exc:  # pragma: no cover - last-resort HTTP boundary
            LOGGER.exception("Request failed")
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    load_model()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    LOGGER.info("Serving on 0.0.0.0:%s", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
