# Query Intent Model Training Result and Deployment

## Artifact Locations

Local artifacts:

```text
Current directory:
  outputs\query_gate_biobert
  outputs\intent_biobert
  artifacts\remote_model_manifest.json
  logs\train_gate_biobert_20260615_145106_retry.log
  logs\train_intent_biobert_20260615_145106_retry.log
```

Original remote training location on `cu05`:

```text
/home/model_train/query_intent_model
```

The model files were downloaded from `cu05` and verified with SHA256. See:

```text
artifacts\remote_model_manifest.json
```

## Environment

Remote training environment:

```bash
/data/k8s/anaconda3/envs/py-train/bin/python
```

Local validation environment used after download:

```text
torch 2.9.1+cpu
transformers 4.57.3
```

The training implementation is based on Hugging Face Transformers
`AutoModelForSequenceClassification` and `Trainer`.

## Dataset Summary

Gate model:

| Split | Samples | Labels |
| --- | ---: | --- |
| train | 35,083 | paper_search 33,551; non_paper_search 1,532 |
| dev | 1,135 | paper_search 1,000; non_paper_search 135 |
| test | 1,400 | paper_search 1,050; non_paper_search 350 |

Intent model:

| Split | Samples |
| --- | ---: |
| train | 33,551 |
| dev | 1,000 |
| test | 1,050 |

Intent labels:

```text
application_search
citation_trace
comparison_search
dataset_search
mechanism_search
method_search
metric_search
survey_search
```

## Training Results

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 | Train runtime |
| --- | ---: | ---: | ---: | ---: |
| query_gate_biobert | 1.000000 | 0.997143 | 0.996205 | 482.6s |
| intent_biobert | 0.998243 | 0.997143 | 0.923703 | 364.5s |

Detailed metrics:

```text
outputs\query_gate_biobert\metrics.json
outputs\intent_biobert\metrics.json
```

## Local Prediction

Run from:

```powershell
cd <this query_intent_model directory>
```

Gate model:

```powershell
python scripts\predict_sequence_classifier.py `
  --model-dir outputs\query_gate_biobert `
  --text "Could you recommend recent papers about neural information retrieval?" `
  --text "Write a Python function to sort a list."
```

Expected labels:

```text
paper_search
non_paper_search
```

Intent model:

```powershell
python scripts\predict_sequence_classifier.py `
  --model-dir outputs\intent_biobert `
  --text "What works are related to dense passage retrieval?" `
  --text "papers that proposed hierarchical neural ranking models"
```

Expected labels:

```text
survey_search
method_search
```

## Remote Training Commands

The completed training used BioBERT from a patched local checkpoint on `cu05`.
If retraining is needed after uploading the project again:

```bash
cd /home/model_train/query_intent_model
PY=/data/k8s/anaconda3/envs/py-train/bin/python

CUDA_VISIBLE_DEVICES=2 "$PY" scripts/train_sequence_classifier.py \
  --config configs/gate_biobert_remote.json

CUDA_VISIBLE_DEVICES=3 "$PY" scripts/train_sequence_classifier.py \
  --config configs/intent_biobert_remote.json
```

## Notes and Risks

- The gate model is intended as the first classifier: paper-search query or
  non-paper-search query.
- The intent model is trained from deterministic weak labels. It is suitable
  as a first training artifact, but a manually checked validation set is still
  recommended before using the metrics as a release gate.
- Remote `/home` space was limited during training. Intermediate bge and smoke
  checkpoints were removed, and the final important artifacts are preserved
  locally under `outputs`.
