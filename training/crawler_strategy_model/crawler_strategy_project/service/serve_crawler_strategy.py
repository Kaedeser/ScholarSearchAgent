#!/usr/bin/env python3
# 中文功能说明：Crawler Strategy 模型 HTTP 服务，加载 Qwen LoRA 并预测章节展开动作。

"""HTTP service for the crawler strategy LoRA adapter."""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPAND_RE = re.compile(r"\[Expand\](.*?)(?=\[Expand\]|\[StopExpand\])", re.DOTALL)


def trim_action(text: str) -> str:
    stop_token = "[StopExpand]"
    stop_at = text.find(stop_token)
    if stop_at >= 0:
        return text[: stop_at + len(stop_token)].strip()
    return text.strip()


def parse_action(text: str) -> dict[str, Any]:
    text = trim_action(text)
    return {
        "parse_success": text.endswith("[StopExpand]"),
        "sections": [item.strip() for item in EXPAND_RE.findall(text) if item.strip()],
    }


def build_prompt(payload: dict[str, Any]) -> str:
    if payload.get("prompt"):
        return str(payload["prompt"])

    query = payload.get("query") or payload.get("question") or payload.get("research_question")
    title = payload.get("title", "")
    abstract = payload.get("abstract", "")
    sections = payload.get("sections", [])
    if not query:
        raise ValueError("Missing required field: query")
    if isinstance(sections, str):
        sections_text = sections
    else:
        sections_text = json.dumps(sections, ensure_ascii=False)

    return (
        f"你正在围绕“{query}”开展学术论文检索与阅读。"
        "请根据当前论文的标题、摘要和候选章节列表，判断下一步是否需要展开阅读某些章节，"
        "以便找到更相关的论文线索。"
        "输出必须严格使用动作格式：[Expand]章节标题[Expand]章节标题[StopExpand]；"
        "如果不需要继续展开，请只输出：[StopExpand]。\n"
        f"论文标题：{title}\n"
        f"论文摘要：{abstract}\n"
        f"候选章节：{sections_text}"
    )


class CrawlerStrategyModel:
    def __init__(self, base_model: str, adapter_dir: str, max_new_tokens: int) -> None:
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=Path(adapter_dir).exists(),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=Path(base_model).exists(),
        )
        self.model = PeftModel.from_pretrained(model, adapter_dir)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        prompt = build_prompt(payload)
        max_new_tokens = int(payload.get("max_new_tokens") or self.max_new_tokens)
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prediction = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        prediction = trim_action(prediction)
        parsed = parse_action(prediction)
        return {
            "prediction": prediction,
            **parsed,
            "latency_ms": round((time.time() - started) * 1000, 2),
            "model": {
                "base_model": self.base_model,
                "adapter_dir": self.adapter_dir,
            },
        }


class Handler(BaseHTTPRequestHandler):
    model: CrawlerStrategyModel

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self.send_json(200, {"status": "ok", "model_loaded": True})
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in {"/predict", "/generate"}:
            self.send_json(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.model.predict(payload)
        except Exception as exc:  # Return structured errors for service diagnostics.
            self.send_json(400, {"error": type(exc).__name__, "message": str(exc)})
            return
        self.send_json(200, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    Handler.model = CrawlerStrategyModel(args.base_model, args.adapter_dir, args.max_new_tokens)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"crawler strategy service listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
