# 中文功能说明：Selector Reranker 数据构建脚本，将 PaSa selector 数据转换为 CrossEncoder JSONL。

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import default_pasa_data_dir


USER_QUERY_RE = re.compile(r"\nUser Query:\s*(.*?)\n\nOutput format:", re.DOTALL | re.IGNORECASE)
TITLE_RE = re.compile(r"\nTitle:\s*(.*?)\nAbstract:", re.DOTALL | re.IGNORECASE)
ABSTRACT_RE = re.compile(r"\nAbstract:\s*(.*?)\n\nUser Query:", re.DOTALL | re.IGNORECASE)
DECISION_RE = re.compile(r"^(?:Decision:\s*)?(True|False)\b", re.IGNORECASE)


@dataclass(frozen=True)
class BuildStats:
    written: int
    labels: Counter
    skipped: int


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def _parse_label(assistant_content: str) -> int | None:
    first_line = assistant_content.strip().splitlines()[0] if assistant_content.strip() else ""
    match = DECISION_RE.match(first_line.strip())
    if not match:
        match = DECISION_RE.search(assistant_content.strip())
    if not match:
        return None
    return 1 if match.group(1).lower() == "true" else 0


def _truncate_chars(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def parse_sft_selector_row(row: dict[str, Any], *, max_abstract_chars: int) -> dict[str, Any] | None:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None

    user_content = str(messages[0].get("content") or "")
    assistant_content = str(messages[1].get("content") or "")
    label = _parse_label(assistant_content)
    query = _first_match(USER_QUERY_RE, user_content)
    title = _first_match(TITLE_RE, user_content)
    abstract = _first_match(ABSTRACT_RE, user_content)

    if label is None or not query or not title:
        return None

    document = f"Title: {title}"
    if abstract:
        document += f"\nAbstract: {_truncate_chars(abstract, max_abstract_chars)}"

    return {
        "query": query,
        "document": document,
        "label": label,
        "source": "sft_selector",
        "title": title,
    }


def build_from_sft_selector(
    input_path: Path,
    output_path: Path,
    *,
    max_rows: int | None,
    max_abstract_chars: int,
    seed: int,
    negative_ratio: float | None,
) -> BuildStats:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    skipped = 0

    for row in read_jsonl(input_path):
        parsed = parse_sft_selector_row(row, max_abstract_chars=max_abstract_chars)
        if parsed is None:
            skipped += 1
            continue
        if parsed["label"] == 1:
            positives.append(parsed)
        else:
            negatives.append(parsed)

    rng = random.Random(seed)
    if negative_ratio is not None and positives:
        max_negatives = int(round(len(positives) * negative_ratio))
        if max_negatives < len(negatives):
            negatives = rng.sample(negatives, max_negatives)

    rows = positives + negatives
    rng.shuffle(rows)
    if max_rows is not None:
        rows = rows[:max_rows]

    written = write_jsonl(output_path, rows)
    labels = Counter(row["label"] for row in rows)
    return BuildStats(written=written, labels=labels, skipped=skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ScholarSearch selector reranker JSONL datasets.")
    parser.add_argument("--pasa-data-dir", type=Path, default=default_pasa_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-dev", type=int, default=None)
    parser.add_argument("--max-abstract-chars", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=None,
        help="Optional negative:positive ratio. Leave unset to keep all sft_selector rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sft_dir = args.pasa_data_dir / "sft_selector"
    train_path = sft_dir / "train.jsonl"
    test_path = sft_dir / "test.jsonl"
    missing = [path for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PaSa sft_selector files: " + ", ".join(str(path) for path in missing))

    train_stats = build_from_sft_selector(
        train_path,
        args.output_dir / "train.jsonl",
        max_rows=args.max_train,
        max_abstract_chars=args.max_abstract_chars,
        seed=args.seed,
        negative_ratio=args.negative_ratio,
    )
    dev_stats = build_from_sft_selector(
        test_path,
        args.output_dir / "dev.jsonl",
        max_rows=args.max_dev,
        max_abstract_chars=args.max_abstract_chars,
        seed=args.seed + 1,
        negative_ratio=args.negative_ratio,
    )
    metadata = {
        "source": "pasa/data/sft_selector",
        "train": {
            "rows": train_stats.written,
            "labels": dict(train_stats.labels),
            "skipped": train_stats.skipped,
        },
        "dev": {
            "rows": dev_stats.written,
            "labels": dict(dev_stats.labels),
            "skipped": dev_stats.skipped,
        },
        "max_abstract_chars": args.max_abstract_chars,
        "negative_ratio": args.negative_ratio,
    }
    write_jsonl(args.output_dir / "metadata.jsonl", [metadata])
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
