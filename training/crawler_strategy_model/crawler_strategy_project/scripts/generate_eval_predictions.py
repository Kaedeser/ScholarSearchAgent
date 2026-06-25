#!/usr/bin/env python3
# 中文功能说明：Crawler Strategy 评估预测生成脚本，加载 LoRA 模型生成动作串。

"""Generate crawler strategy predictions for action-level evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_rows(path: Path, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and len(rows) >= max_samples:
                break
            rows.append(json.loads(line))
    return rows


def split_messages(row: dict) -> tuple[list[dict], str]:
    messages = row["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected the final message to be the assistant label.")
    return messages[:-1], messages[-1].get("content", "")


def trim_action(text: str) -> str:
    stop_token = "[StopExpand]"
    stop_at = text.find(stop_token)
    if stop_at >= 0:
        return text[: stop_at + len(stop_token)].strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--eval-file", default="data/crawler_sft_eval.jsonl")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    local_files_only = Path(args.base_model).exists()

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    device = next(model.parameters()).device

    rows = iter_rows(Path(args.eval_file), args.max_samples)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            labels = []
            prompt_texts = []
            for row in batch:
                prompt_messages, label = split_messages(row)
                labels.append(label)
                prompt_texts.append(
                    tokenizer.apply_chat_template(
                        prompt_messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )

            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for offset, label in enumerate(labels):
                prompt_len = int(inputs["attention_mask"][offset].sum().item())
                prediction = tokenizer.decode(
                    generated[offset, inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
                prediction = trim_action(prediction)
                handle.write(
                    json.dumps(
                        {
                            "index": start + offset,
                            "label": label,
                            "prediction": prediction,
                            "prompt_tokens": prompt_len,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            done = min(start + len(batch), len(rows))
            if done % 25 == 0 or done == len(rows):
                print(f"generated={done}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
