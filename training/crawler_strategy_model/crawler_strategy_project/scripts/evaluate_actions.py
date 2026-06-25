#!/usr/bin/env python3
# 中文功能说明：Crawler Strategy 动作评估脚本，解析动作串并计算章节级指标。

"""Evaluate crawler action strings with section-level metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPAND_RE = re.compile(r"\[Expand\](.*?)(?=\[Expand\]|\[StopExpand\])", re.DOTALL)


def parse_action(text: str) -> tuple[bool, set[str]]:
    text = (text or "").strip()
    if not text.endswith("[StopExpand]"):
        return False, set()
    return True, {item.strip() for item in EXPAND_RE.findall(text) if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="JSONL with label and prediction fields.")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--prediction-field", default="prediction")
    args = parser.parse_args()

    total = exact = parse_ok = stop_correct = 0
    tp = fp = fn = 0

    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = row.get(args.label_field, "")
            prediction = row.get(args.prediction_field, "")
            total += 1

            label_ok, gold_sections = parse_action(label)
            pred_ok, pred_sections = parse_action(prediction)
            parse_ok += int(pred_ok)
            exact += int(label.strip() == prediction.strip())
            stop_correct += int((len(gold_sections) == 0) == (len(pred_sections) == 0))

            tp += len(gold_sections & pred_sections)
            fp += len(pred_sections - gold_sections)
            fn += len(gold_sections - pred_sections)

            if not label_ok:
                raise ValueError(f"Bad label action string: {label!r}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "total": total,
        "exact_match": exact / total if total else 0.0,
        "parse_success_rate": parse_ok / total if total else 0.0,
        "stop_accuracy": stop_correct / total if total else 0.0,
        "section_precision": precision,
        "section_recall": recall,
        "section_f1": f1,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
