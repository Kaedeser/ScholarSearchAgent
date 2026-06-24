from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SequenceClassifier:
    def __init__(self, model_dir: Path, max_length: int = 256) -> None:
        self.model_dir = model_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)

        id2label = self.model.config.id2label
        results = []
        for text, probs in zip(texts, probabilities):
            label_id = int(torch.argmax(probs).item())
            label = str(id2label.get(label_id, id2label.get(str(label_id), str(label_id))))
            results.append(
                {
                    "text": text,
                    "label": label,
                    "score": float(probs[label_id].item()),
                    "scores": {
                        str(id2label.get(index, id2label.get(str(index), index))): float(value.item())
                        for index, value in enumerate(probs)
                    },
                }
            )
        return results


class QueryIntentService:
    def __init__(self, model_root: Path) -> None:
        gate_dir = model_root / "query_gate_biobert"
        intent_dir = model_root / "intent_biobert"
        self.gate = SequenceClassifier(gate_dir)
        self.intent = SequenceClassifier(intent_dir)

    def predict(self, texts: list[str], mode: str = "auto") -> list[dict[str, Any]]:
        if mode == "gate":
            return [{"gate": item} for item in self.gate.predict(texts)]
        if mode == "intent":
            return [{"intent": item} for item in self.intent.predict(texts)]
        if mode != "auto":
            raise ValueError("mode must be one of: auto, gate, intent")

        gate_results = self.gate.predict(texts)
        intent_inputs = [item["text"] for item in gate_results if item["label"] == "paper_search"]
        intent_predictions = self.intent.predict(intent_inputs) if intent_inputs else []
        intent_by_text = {item["text"]: item for item in intent_predictions}

        results = []
        for gate_item in gate_results:
            payload: dict[str, Any] = {"gate": gate_item}
            if gate_item["label"] == "paper_search":
                payload["intent"] = intent_by_text.get(gate_item["text"])
            else:
                payload["intent"] = None
            results.append(payload)
        return results


def make_handler(service: QueryIntentService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "QueryIntentHTTP/1.0"

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] == "/health":
                self.write_json(200, {"status": "ok", "service": "query-intent-service"})
                return
            self.write_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/predict":
                self.write_json(404, {"error": "not found"})
                return

            try:
                payload = self.read_json()
                texts = normalize_texts(payload)
                mode = str(payload.get("mode", "auto"))
                predictions = service.predict(texts, mode=mode)
                self.write_json(200, {"results": predictions})
            except Exception as exc:
                self.write_json(400, {"error": str(exc)})

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.add_common_headers()
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - - %s" % (self.address_string(), fmt % args), flush=True)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0:
                raise ValueError("request body is empty")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.add_common_headers()
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def add_common_headers(self) -> None:
            self.send_header("access-control-allow-origin", "*")
            self.send_header("access-control-allow-methods", "GET,POST,OPTIONS")
            self.send_header("access-control-allow-headers", "content-type")

    return Handler


def normalize_texts(payload: dict[str, Any]) -> list[str]:
    if "texts" in payload:
        texts = payload["texts"]
        if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
            raise ValueError("texts must be a list of strings")
    elif "text" in payload:
        text = payload["text"]
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        texts = [text]
    else:
        raise ValueError("request must include text or texts")

    texts = [item.strip() for item in texts if item.strip()]
    if not texts:
        raise ValueError("no non-empty text values were provided")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve query intent classifiers over HTTP.")
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT", "/app/models"))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()

    service = QueryIntentService(Path(args.model_root))
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"query-intent-service listening on {args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
