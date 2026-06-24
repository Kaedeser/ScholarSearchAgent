from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .paths import sentence_transformers_source_dir


FRAMEWORK_DIR = sentence_transformers_source_dir()
if FRAMEWORK_DIR.exists():
    sys.path.insert(0, str(FRAMEWORK_DIR))

from sentence_transformers.cross_encoder import CrossEncoder  # noqa: E402
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator  # noqa: E402
from sentence_transformers.sentence_transformer.readers import InputExample  # noqa: E402


LOGGER = logging.getLogger(__name__)


class MetricSelectingClassificationEvaluator(CrossEncoderClassificationEvaluator):
    def __init__(self, *args: Any, selection_metric: str = "average_precision", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.selection_metric = selection_metric

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        metrics = super().__call__(*args, **kwargs)
        metric_by_suffix = {key.rsplit("_", 1)[-1]: value for key, value in metrics.items()}
        accuracy = metric_by_suffix.get("accuracy")
        f1 = metric_by_suffix.get("f1")
        if accuracy is not None and f1 is not None:
            prefix = next((key[: -len("accuracy")] for key in metrics if key.endswith("accuracy")), "")
            metrics[f"{prefix}min_accuracy_f1"] = min(accuracy, f1)
            metrics[f"{prefix}mean_accuracy_f1"] = (accuracy + f1) / 2
        for key in metrics:
            if key == self.selection_metric or key.endswith(f"_{self.selection_metric}"):
                self.primary_metric = key
                break
        return metrics


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_dataset(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        label = int(row["label"])
        rows.append(
            {
                "query": str(row["query"]),
                "document": str(row["document"]),
                "label": label,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def labels_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {str(label): count for label, count in sorted(counts.items())}


def classification_metrics(scores: np.ndarray, labels: list[int]) -> dict[str, float]:
    label_array = np.asarray(labels)
    thresholds = np.unique(scores)
    if len(thresholds) == 0:
        thresholds = np.asarray([0.5])

    best_acc: tuple[float, float] = (-1.0, 0.5)
    best_f1: tuple[float, float, float, float] = (-1.0, 0.0, 0.0, 0.5)
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        acc = accuracy_score(label_array, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_array,
            preds,
            average="binary",
            zero_division=0,
        )
        if acc > best_acc[0]:
            best_acc = (float(acc), float(threshold))
        if f1 > best_f1[0]:
            best_f1 = (float(f1), float(precision), float(recall), float(threshold))

    return {
        "accuracy": best_acc[0],
        "accuracy_threshold": best_acc[1],
        "f1": best_f1[0],
        "f1_threshold": best_f1[3],
        "precision_at_f1_threshold": best_f1[1],
        "recall_at_f1_threshold": best_f1[2],
        "average_precision": float(average_precision_score(label_array, scores)),
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a CrossEncoder selector/reranker.")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/processed/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bge-reranker-base-pasa"))
    parser.add_argument("--model-name", default="BAAI/bge-reranker-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-dev-rows", type=int, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--classifier-dropout", type=float, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("average_precision", "f1", "accuracy", "min_accuracy_f1", "mean_accuracy_f1"),
        default="average_precision",
        help="Validation metric used for best-checkpoint saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    train_rows = load_dataset(args.train_file, limit=args.max_train_rows)
    dev_rows = load_dataset(args.dev_file, limit=args.max_dev_rows)
    LOGGER.info("Loaded train=%s labels=%s", len(train_rows), labels_summary(train_rows))
    LOGGER.info("Loaded dev=%s labels=%s", len(dev_rows), labels_summary(dev_rows))

    train_examples = [
        InputExample(texts=[row["query"], row["document"]], label=float(row["label"]))
        for row in train_rows
    ]
    if args.dataloader_num_workers != 0:
        LOGGER.warning("old_fit moves labels to CUDA in collate; forcing dataloader_num_workers=0.")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size, num_workers=0)
    dev_pairs = [(row["query"], row["document"]) for row in dev_rows]
    dev_labels = [int(row["label"]) for row in dev_rows]

    config_kwargs = {}
    if args.classifier_dropout is not None:
        config_kwargs["classifier_dropout"] = args.classifier_dropout
    model = CrossEncoder(
        args.model_name,
        num_labels=1,
        max_length=args.max_length,
        config_kwargs=config_kwargs or None,
    )
    LOGGER.info("Model=%s max_length=%s num_labels=%s", args.model_name, model.max_length, model.num_labels)

    label_counts = Counter(int(row["label"]) for row in train_rows)
    if label_counts[1] > 0:
        pos_weight_value = max(1.0, label_counts[0] / label_counts[1])
    else:
        pos_weight_value = 1.0
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=model.device))
    LOGGER.info("Using BCE pos_weight=%.4f", pos_weight_value)

    evaluator = MetricSelectingClassificationEvaluator(
        sentence_pairs=dev_pairs,
        labels=dev_labels,
        name="pasa-dev",
        batch_size=max(1, args.batch_size * 2),
        show_progress_bar=False,
        selection_metric=args.selection_metric,
    )

    train_steps_per_epoch = max(1, math.ceil(len(train_rows) / max(1, args.batch_size)))
    eval_steps = min(args.eval_steps, train_steps_per_epoch) if args.eval_steps > 0 else train_steps_per_epoch
    warmup_steps = int(train_steps_per_epoch * args.epochs * args.warmup_ratio)
    if args.gradient_accumulation_steps != 1:
        LOGGER.warning("old_fit path does not support gradient accumulation; using batch_size=%s directly.", args.batch_size)

    evaluator(model, output_path=str(args.output_dir / "eval"), epoch=0, steps=0)
    model.old_fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        loss_fct=loss,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        evaluation_steps=eval_steps,
        output_path=str(args.output_dir),
        save_best_model=True,
        max_grad_norm=1,
        use_amp=args.fp16,
        show_progress_bar=True,
    )
    best_model = CrossEncoder(str(args.output_dir))
    evaluator(best_model, output_path=str(args.output_dir / "eval"), epoch=-1, steps=-1)
    scores = np.asarray(
        best_model.predict(
            dev_pairs,
            batch_size=max(1, args.batch_size * 2),
            show_progress_bar=False,
        )
    )
    metrics = classification_metrics(scores, dev_labels)
    run_config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
        "classifier_dropout": args.classifier_dropout,
        "selection_metric": args.selection_metric,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_labels": labels_summary(train_rows),
        "dev_labels": labels_summary(dev_rows),
    }
    save_json(args.output_dir / "metrics.json", {"metrics": metrics, "run_config": run_config})
    LOGGER.info("Best dev metrics: %s", json.dumps(metrics, ensure_ascii=False))

    final_dir = args.output_dir / "final"
    best_model.save_pretrained(str(final_dir))
    save_json(final_dir / "metrics.json", {"metrics": metrics, "run_config": run_config})
    LOGGER.info("Saved final model to %s", final_dir)


if __name__ == "__main__":
    main()
