# Crawler Strategy Model Tuning Analysis

## Current Baseline

Current completed run:

```text
model: Qwen/Qwen2.5-3B-Instruct
framework: LLaMA-Factory 0.9.5
method: LoRA SFT
lora_rank: 8
lora_alpha: 16
cutoff_len: 1024
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
GPUs: 7 x V100 16GB, CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
effective_batch_size: 112
```

Metrics:

```text
train_loss: 0.3578266545578285
eval_loss final: 0.17109628021717072
eval_loss checkpoint-50: 0.17330560088157654
eval_loss checkpoint-100: 0.17636631429195404
eval_loss checkpoint-150: 0.17080816626548767
```

The eval curve is already flat after about 50 optimizer steps. The best observed eval loss is checkpoint-150, only slightly better than final.

## Data And Length Diagnostics

Prepared data:

```text
raw lines: 12989
kept: 9494
train: 8994
eval: 500
skipped_missing_sections: 3011
skipped_too_many_expands: 484
```

Token length distribution with the trained tokenizer:

```text
train mean: 509 tokens
train p95: 787
train p99: 988
train max: 1980
train >1024: 72 samples, 0.80%

eval mean: 511 tokens
eval p95: 800
eval p99: 1100
eval max: 1513
eval >1024: 8 samples, 1.60%
```

Conclusion: `cutoff_len=1024` is not the main bottleneck. Increasing to 1536 would recover only a small number of samples while increasing memory pressure.

Label distribution:

```text
0 expands: 4317
1 expand: 808
2 expands: 1280
3 expands: 1189
4 expands: 942
5 expands: 600
6 expands: 358
```

The dataset has many stop-only samples. This is useful for stop behavior, but action-level metrics are needed to know whether the model is learning section choice or mostly learning easy stop patterns.

## Can Hyperparameters Improve It?

Yes, but more epochs alone are unlikely to help much.

The current run has three likely limits:

1. Effective batch is large.
   With 7 GPUs and `gradient_accumulation_steps=16`, the effective batch is 112, producing only 162 optimizer steps. This is stable but coarse for a small SFT dataset.

2. LoRA rank is conservative.
   `rank=8` is fine for a first pass, but crawler strategy is a structured generation task with many section-name variants. `rank=16` may improve adapter capacity at low extra memory cost.

3. Eval loss is insufficient.
   The task needs exact/action metrics: parse success, stop accuracy, section precision/recall/F1. A lower token loss may not map cleanly to better section decisions.

## Recommended Tuning Order

### Round 1: Better 3B LoRA, Low Risk

Use this first.

```text
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 5e-5
num_train_epochs: 3
gradient_accumulation_steps: 8
cutoff_len: 1024
save_steps: 50
eval_steps: 50
```

Expected impact:

- More update steps: about 483 instead of 162.
- More adapter capacity.
- Lower risk of overfitting than keeping `1e-4` for 3 epochs.
- Runtime roughly 3x current run, around 60-70 minutes on the same 7 V100 GPUs.

### Round 2: Slightly Longer Context Ablation

Only run this if Round 1 improves action metrics.

```text
cutoff_len: 1536
gradient_accumulation_steps: 8
lora_rank: 16
learning_rate: 5e-5
```

Expected impact:

- Small gain at best, because less than 2% of examples exceed 1024 tokens.
- Higher OOM risk. Keep GPU 0 excluded and use only idle cards.

### Round 3: Qwen2.5-7B QLoRA

Use only if 3B action metrics are clearly insufficient.

```text
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
quantization_bit: 4
lora_rank: 8 or 16
learning_rate: 5e-5
cutoff_len: 1024
```

Expected impact:

- Best chance of real quality improvement.
- More dependency risk because `bitsandbytes` must work in the cu05 environment.
- Higher inference cost for a crawler service.

## Evaluation Recommendation

Before accepting a tuned model, generate predictions on `crawler_sft_eval.jsonl` and compute:

```text
exact_match
parse_success_rate
stop_accuracy
section_precision
section_recall
section_f1
avg_predicted_sections
```

Model selection should prioritize section F1 and stop accuracy, not eval loss alone.

## Practical Recommendation

The best next experiment is:

```text
Qwen2.5-3B-Instruct + LoRA rank 16 + alpha 32 + lr 5e-5 + 3 epochs + grad_accum 8 + cutoff_len 1024
```

Do not increase `cutoff_len` first. Do not simply increase epochs at the current `1e-4` learning rate. Do not move to 7B before action-level evaluation proves 3B is inadequate.
