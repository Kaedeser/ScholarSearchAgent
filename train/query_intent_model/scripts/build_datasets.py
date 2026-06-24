from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from query_intent.labeling import normalize_text, weak_intent_label


MANUAL_NEGATIVES = (
    "What is the weather like today?",
    "Write a Python function that sorts a list of numbers.",
    "How do I configure nginx reverse proxy for a Flask app?",
    "Translate this paragraph into Chinese.",
    "Summarize this meeting transcript.",
    "What is the capital city of France?",
    "Create a weekly study plan for machine learning.",
    "Debug this JavaScript error in my browser console.",
    "How do I install CUDA on Ubuntu?",
    "Generate a polite email asking for an extension.",
    "Explain how to use pandas groupby.",
    "What does this SQL query do?",
    "Help me write a project proposal outline.",
    "Convert these coordinates from WGS84 to GCJ02.",
    "How can I reset my Linux password?",
    "Build a REST API endpoint in FastAPI.",
    "What are the symptoms of dehydration?",
    "Plan a three-day trip to Shanghai.",
    "Calculate the monthly payment for a loan.",
    "Explain recursion with a simple example.",
    "How do I merge two Excel sheets?",
    "Write unit tests for this helper function.",
    "What is the difference between TCP and UDP?",
    "Make this sentence sound more professional.",
    "How can I speed up this SQL join?",
    "Create a Dockerfile for a Node.js service.",
    "What is the exchange rate from USD to CNY?",
    "Help me brainstorm startup names.",
    "How do I center a div with CSS?",
    "Explain the plot of this movie.",
    "What is Kubernetes used for?",
    "Write a bash script to back up a folder.",
    "How can I parse JSON in Go?",
    "Give me a recipe for tomato soup.",
    "What is the time complexity of binary search?",
    "Fix this TypeScript type error.",
    "How do I connect to MySQL from Python?",
    "Create a regex for validating phone numbers.",
    "What should I prepare for a job interview?",
    "Explain OAuth 2.0 in plain language.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query gate and intent datasets.")
    parser.add_argument("--pasa-data-root", required=True, help="Path containing AutoScholarQuery and RealScholarQuery.")
    parser.add_argument("--astabench-tasks-root", required=True, help="Path containing AstaBench task folders.")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "data" / "processed"))
    parser.add_argument("--negative-ratio", type=float, default=3.0, help="Target paper:non-paper ratio.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pasa_root = Path(args.pasa_data_root)
    astabench_root = Path(args.astabench_tasks_root)
    output_dir = Path(args.output_dir)

    positives = {
        "train": list(read_pasa_file(pasa_root / "AutoScholarQuery" / "train.jsonl", "AutoScholarQuery/train")),
        "dev": list(read_pasa_file(pasa_root / "AutoScholarQuery" / "dev.jsonl", "AutoScholarQuery/dev")),
        "test": list(read_pasa_file(pasa_root / "AutoScholarQuery" / "test.jsonl", "AutoScholarQuery/test"))
        + list(read_pasa_file(pasa_root / "RealScholarQuery" / "test.jsonl", "RealScholarQuery/test")),
    }

    astabench_negatives = list(read_library_diagnostic(astabench_root / "library_diagnostic"))
    manual_negatives = list(read_manual_negatives())
    negatives = split_negatives(astabench_negatives + manual_negatives, rng)

    gate_splits: dict[str, list[dict[str, str]]] = {}
    intent_splits: dict[str, list[dict[str, str]]] = {}
    metadata: dict[str, object] = {"splits": {}}

    for split in ("train", "dev", "test"):
        paper_records = [
            make_record(item["text"], "paper_search", item["source"], item["id"])
            for item in positives[split]
        ]
        target_negatives = max(1, int(len(paper_records) / args.negative_ratio))
        non_paper_records = [
            make_record(item["text"], "non_paper_search", item["source"], item["id"])
            for item in sample_or_all(negatives[split], target_negatives, rng)
        ]
        gate_records = paper_records + non_paper_records
        rng.shuffle(gate_records)
        gate_splits[split] = gate_records

        intent_records = [
            make_record(item["text"], weak_intent_label(item["text"]), item["source"], item["id"])
            for item in positives[split]
        ]
        rng.shuffle(intent_records)
        intent_splits[split] = intent_records

        metadata["splits"][split] = {
            "gate_count": len(gate_records),
            "gate_labels": dict(Counter(record["label"] for record in gate_records)),
            "intent_count": len(intent_records),
            "intent_labels": dict(Counter(record["label"] for record in intent_records)),
        }

    write_splits(output_dir / "gate", gate_splits)
    write_splits(output_dir / "intent", intent_splits)
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def read_pasa_file(path: Path, source: str) -> Iterable[dict[str, str]]:
    for index, row in enumerate(read_jsonl(path)):
        text = normalize_text(row.get("question") or row.get("query_text"))
        if not text:
            continue
        yield {
            "text": text,
            "source": source,
            "id": normalize_text(row.get("qid")) or f"{source}_{index}",
        }


def read_library_diagnostic(root: Path) -> Iterable[dict[str, str]]:
    for name in ("dev.jsonl", "test.jsonl"):
        path = root / name
        if not path.exists():
            continue
        for index, row in enumerate(read_jsonl(path)):
            text = normalize_text(row.get("question"))
            if text:
                yield {
                    "text": text,
                    "source": f"astabench/library_diagnostic/{name}",
                    "id": normalize_text(row.get("id")) or f"library_diagnostic_{name}_{index}",
                }


def read_manual_negatives() -> Iterable[dict[str, str]]:
    for index, text in enumerate(MANUAL_NEGATIVES):
        yield {"text": text, "source": "manual_negative_templates", "id": f"manual_negative_{index}"}
    for index, text in enumerate(generated_negative_templates()):
        yield {"text": text, "source": "generated_negative_templates", "id": f"generated_negative_{index}"}


def generated_negative_templates() -> Iterable[str]:
    """Generate deterministic non-paper-search requests.

    These stay close to realistic system traffic while avoiding ambiguous
    scientific literature requests from AstaBench SQA/DiscoveryBench.
    """

    programming_subjects = (
        "Python",
        "JavaScript",
        "TypeScript",
        "Go",
        "SQL",
        "FastAPI",
        "React",
        "pandas",
        "PyTorch",
        "Transformers",
        "Docker",
        "Kubernetes",
        "nginx",
        "MySQL",
        "Redis",
        "Elasticsearch",
        "Qdrant",
        "Neo4j",
        "Linux",
        "Git",
    )
    programming_actions = (
        "write a minimal example for",
        "debug an error in",
        "explain how to use",
        "configure",
        "optimize a slow",
        "add unit tests for",
        "refactor this",
        "install",
        "connect to",
        "serialize data with",
        "parse logs from",
        "deploy a service with",
    )
    for action in programming_actions:
        for subject in programming_subjects:
            yield f"Can you {action} {subject}?"
            yield f"Help me {action} {subject} in a local project."

    writing_tasks = (
        "polish this paragraph",
        "write a polite email",
        "draft a meeting summary",
        "translate this text into English",
        "turn these notes into a checklist",
        "make this resume bullet more concise",
        "create a project timeline",
        "summarize this document",
        "rewrite this sentence in a formal tone",
        "prepare an interview self-introduction",
    )
    audiences = (
        "for a professor",
        "for a teammate",
        "for a product manager",
        "for a job application",
        "for a weekly report",
        "for a class presentation",
    )
    for task in writing_tasks:
        for audience in audiences:
            yield f"Please {task} {audience}."

    general_questions = (
        "What is the capital of Canada?",
        "How many grams are in a kilogram?",
        "What is the difference between TCP and UDP?",
        "Explain gradient descent without mentioning papers.",
        "How do I calculate compound interest?",
        "What are common symptoms of dehydration?",
        "How should I plan a two-day trip to Beijing?",
        "What is the difference between RAM and disk storage?",
        "How do I make tomato egg noodles?",
        "Explain recursion with a toy example.",
        "What does HTTP status code 502 mean?",
        "How do I convert a CSV file to Excel?",
        "What is the Unix command to find large files?",
        "How can I clean duplicated rows in a spreadsheet?",
        "Explain OAuth 2.0 at a high level.",
    )
    for question in general_questions:
        yield question

    operations = (
        "restart the web service",
        "check disk usage",
        "open port 8080",
        "rotate application logs",
        "create a Linux user",
        "set an environment variable",
        "download a file with curl",
        "compress this folder",
        "inspect GPU memory",
        "kill a stuck process",
    )
    for operation in operations:
        yield f"How do I {operation} on Ubuntu?"
        yield f"Give me the command to {operation}."


def split_negatives(records: list[dict[str, str]], rng: random.Random) -> dict[str, list[dict[str, str]]]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)

    splits = {"train": [], "dev": [], "test": []}
    for source, source_records in by_source.items():
        shuffled = source_records[:]
        rng.shuffle(shuffled)
        if "/test.jsonl" in source:
            splits["test"].extend(shuffled)
            continue
        train_end = int(len(shuffled) * 0.85)
        dev_end = int(len(shuffled) * 0.925)
        splits["train"].extend(shuffled[:train_end])
        splits["dev"].extend(shuffled[train_end:dev_end])
        splits["test"].extend(shuffled[dev_end:])
    return splits


def sample_or_all(records: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    if len(records) <= count:
        return records[:]
    return rng.sample(records, count)


def make_record(text: str, label: str, source: str, qid: str) -> dict[str, str]:
    return {"text": normalize_text(text), "label": label, "source": source, "id": qid}


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def write_splits(root: Path, splits: dict[str, list[dict[str, str]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        write_jsonl(root / f"{split}.jsonl", records)


def write_jsonl(path: Path, records: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
