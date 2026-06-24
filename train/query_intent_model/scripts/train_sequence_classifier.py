from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def main() -> None:
    args = parse_args()
    if args.config:
        args = merge_config(args, Path(args.config))

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {"train": args.train_file, "validation": args.validation_file}
    if args.test_file:
        data_files["test"] = args.test_file

    dataset = load_dataset("json", data_files=data_files)
    label_list = sorted(set(dataset["train"]["label"]) | set(dataset["validation"]["label"]))
    if "test" in dataset:
        label_list = sorted(set(label_list) | set(dataset["test"]["label"]))
    label2id = {label: index for index, label in enumerate(label_list)}
    id2label = {index: label for label, index in label2id.items()}

    tokenizer = load_tokenizer(args.model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        encoded["labels"] = [label2id[label] for label in batch["label"]]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
    )

    training_args = build_training_arguments(args)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "compute_metrics": make_compute_metrics(len(label_list)),
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {"train": train_result.metrics}
    metrics["validation"] = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    if "test" in tokenized:
        metrics["test"] = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    write_json(output_dir / "label_map.json", {"labels": label_list, "label2id": label2id, "id2label": id2label})
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sequence classifier on JSONL text/label data.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--train-file", required=False)
    parser.add_argument("--validation-file", required=False)
    parser.add_argument("--test-file")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--output-dir", required=False)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def merge_config(args: argparse.Namespace, config_path: Path) -> argparse.Namespace:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in config.items():
        attr = key.replace("-", "_")
        setattr(args, attr, value)
    required = ("train_file", "validation_file", "output_dir")
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    return args


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": args.fp16,
        "report_to": "none",
        "save_strategy": "epoch",
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def make_compute_metrics(num_labels: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = float(np.mean(predictions == labels))
        f1_scores = []
        precisions = []
        recalls = []
        for label_id in range(num_labels):
            true_positive = int(np.sum((predictions == label_id) & (labels == label_id)))
            false_positive = int(np.sum((predictions == label_id) & (labels != label_id)))
            false_negative = int(np.sum((predictions != label_id) & (labels == label_id)))
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        return {
            "accuracy": accuracy,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1_scores)),
        }

    return compute_metrics


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
