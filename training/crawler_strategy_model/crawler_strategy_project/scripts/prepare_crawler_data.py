#!/usr/bin/env python3
# 中文功能说明：Crawler Strategy 数据准备脚本，清洗 PaSa crawler SFT 数据并划分训练/评估集。

"""Prepare PaSa crawler SFT data for LLaMA-Factory."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any


EXPAND_RE = re.compile(r"\[Expand\](.*?)(?=\[Expand\]|\[StopExpand\])", re.DOTALL)


def parse_actions(text: str) -> list[str] | None:
    text = text.strip()
    if not text.endswith("[StopExpand]"):
        return None
    sections = [match.strip() for match in EXPAND_RE.findall(text)]
    if text != "[StopExpand]" and not sections:
        return None
    return sections


def validate_example(example: dict[str, Any], max_expands: int) -> tuple[bool, str, int]:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "bad_messages", 0

    user_msg = messages[0]
    assistant_msg = messages[-1]
    if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
        return False, "bad_roles", 0

    user_content = user_msg.get("content")
    assistant_content = assistant_msg.get("content")
    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        return False, "bad_content", 0
    if "Sections:" not in user_content:
        return False, "missing_sections", 0

    sections = parse_actions(assistant_content)
    if sections is None:
        return False, "bad_action_format", 0
    if len(sections) > max_expands:
        return False, "too_many_expands", len(sections)

    return True, "ok", len(sections)


def load_valid_examples(source: Path, max_expands: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    skipped: collections.Counter[str] = collections.Counter()
    expand_hist: collections.Counter[int] = collections.Counter()
    total = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            total += 1
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                skipped["bad_json"] += 1
                continue

            ok, reason, expand_count = validate_example(example, max_expands=max_expands)
            if not ok:
                skipped[reason] += 1
                continue

            expand_hist[expand_count] += 1
            examples.append(example)

    stats = {
        "source": str(source),
        "total_lines": total,
        "kept": len(examples),
        "skipped": dict(skipped),
        "expand_count_histogram": {str(key): value for key, value in sorted(expand_hist.items())},
    }
    return examples, stats


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/sft_crawler_train.jsonl", type=Path)
    parser.add_argument("--out-dir", default="data", type=Path)
    parser.add_argument("--eval-size", default=500, type=int)
    parser.add_argument("--seed", default=20260624, type=int)
    parser.add_argument("--max-expands", default=6, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    examples, stats = load_valid_examples(args.source, max_expands=args.max_expands)
    if args.eval_size <= 0 or args.eval_size >= len(examples):
        raise ValueError("--eval-size must be between 1 and the number of valid examples - 1")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    eval_examples = examples[: args.eval_size]
    train_examples = examples[args.eval_size :]

    stats.update(
        {
            "seed": args.seed,
            "eval_size": len(eval_examples),
            "train_size": len(train_examples),
            "max_expands": args.max_expands,
        }
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "crawler_sft_train.jsonl", train_examples)
    write_jsonl(args.out_dir / "crawler_sft_eval.jsonl", eval_examples)
    (args.out_dir / "crawler_sft_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
