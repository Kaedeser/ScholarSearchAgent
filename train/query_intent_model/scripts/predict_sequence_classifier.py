from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one or more query intent labels.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    encoded = tokenizer(args.text, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)

    id2label = model.config.id2label
    results = []
    for text, probs in zip(args.text, probabilities):
        label_id = int(torch.argmax(probs).item())
        label = id2label.get(label_id, id2label.get(str(label_id), str(label_id)))
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

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
