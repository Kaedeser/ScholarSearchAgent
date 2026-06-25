# Query Intent Model Training

This training project uses the open-source Hugging Face Transformers stack as
the first-stage implementation recommended by
`training_plans/01_query_intent_model.md`.

Chosen base:

- Project: Hugging Face Transformers
- License: Apache-2.0
- Upstream: https://github.com/huggingface/transformers
- Training API: `AutoModelForSequenceClassification` + `Trainer`
- Remote primary model on `cu05`:
  `/home/model_train/query_intent_model/local_models/biobert-base-cased-v1.1`
- Remote fallback model on `cu05`: `/data/k8s/model/cu04-bge-m3/bge-m3`

Why this route:

- The current data is mostly English academic search queries.
- The first production milestone needs a reliable gate and intent classifier,
  not a fragile JSON generator.
- Encoder classifiers train quickly on P100-class GPUs and keep the existing
  rule-based `QueryParser` as a fallback for constraints and sub-queries.

## Directory Layout

```text
configs/
  gate_deberta_v3_base.json
  intent_deberta_v3_base.json
  gate_biobert_remote.json
  intent_biobert_remote.json
  gate_bge_m3_remote.json
  intent_bge_m3_remote.json
scripts/
  build_datasets.py
  train_sequence_classifier.py
  predict_sequence_classifier.py
  run_train.sh
  run_train_remote_biobert.sh
  run_train_remote_bge_m3.sh
src/query_intent/
  labeling.py
data/processed/
  gate/{train,dev,test}.jsonl
  intent/{train,dev,test}.jsonl
```

## Build Datasets

From this directory:

```bash
export DATASET_ROOT=/path/to/datasets
python scripts/build_datasets.py \
  --pasa-data-root "$DATASET_ROOT/pasa/data" \
  --astabench-tasks-root "$DATASET_ROOT/asta-bench-dataset/tasks" \
  --output-dir data/processed
```

Outputs:

- `data/processed/gate/*.jsonl`: binary `paper_search` /
  `non_paper_search` samples.
- `data/processed/intent/*.jsonl`: positive paper-search samples with weak
  intent labels.
- `data/processed/metadata.json`: counts and label distributions.

## Train

Install dependencies in the target environment:

```bash
pip install -r requirements.txt
```

Train both models:

```bash
bash scripts/run_train.sh
```

On `cu05`, use the prepared `py-train` environment and local BioBERT
checkpoint:

```bash
PYTHON_BIN=/data/k8s/anaconda3/envs/py-train/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_train_remote_biobert.sh
```

The `bge-m3` encoder is also available as a heavier fallback:

```bash
PYTHON_BIN=/data/k8s/anaconda3/envs/py-train/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_train_remote_bge_m3.sh
```

Or train one task manually:

```bash
python scripts/train_sequence_classifier.py \
  --train-file data/processed/gate/train.jsonl \
  --validation-file data/processed/gate/dev.jsonl \
  --test-file data/processed/gate/test.jsonl \
  --model-name microsoft/deberta-v3-base \
  --output-dir outputs/query_gate_deberta_v3_base \
  --max-length 256 \
  --learning-rate 2e-5 \
  --num-train-epochs 4 \
  --per-device-train-batch-size 16 \
  --fp16
```

## Predict

```bash
python scripts/predict_sequence_classifier.py \
  --model-dir outputs/query_gate_biobert \
  --text "Could you recommend papers about image retrieval?"
```

## Notes

- `RealScholarQuery/test.jsonl` is used only for the test split.
- Negative samples default to AstaBench `library_diagnostic` plus manual
  templates. This keeps the first gate conservative without mixing ambiguous
  scientific QA into `non_paper_search`.
- The intent labels are weak labels produced by deterministic rules. They are
  good enough for a first model pass, but a 300-500 sample manual validation
  set should be added before using metrics as a release gate.
