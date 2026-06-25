# 中文功能说明：Selector Reranker 评估脚本，加载模型并计算验证集指标。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from .paths import sentence_transformers_source_dir


FRAMEWORK_DIR = sentence_transformers_source_dir()
if FRAMEWORK_DIR.exists():
    sys.path.insert(0, str(FRAMEWORK_DIR))

from sentence_transformers.cross_encoder import CrossEncoder  # noqa: E402
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator  # noqa: E402


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CrossEncoder on a selector JSONL file.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=Path("data/processed/dev.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--name", default="pasa-dev")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.eval_file))
    pairs = [(str(row["query"]), str(row["document"])) for row in rows]
    labels = [int(row["label"]) for row in rows]
    model = CrossEncoder(str(args.model_dir))
    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=pairs,
        labels=labels,
        name=args.name,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    metrics = evaluator(model)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
