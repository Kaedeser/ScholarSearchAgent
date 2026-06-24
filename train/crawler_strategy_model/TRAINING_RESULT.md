# Crawler Strategy Model Training Result

## Selection

Chosen open-source project: LLaMA-Factory 0.9.5.

Rationale:

- It matches the training plan's first choice: Qwen Instruct + LoRA SFT.
- It supports OpenAI `messages` style data directly, so the PaSa crawler SFT records can stay close to their source format.
- It is practical on the available V100 16GB GPUs with fp16 LoRA and vanilla attention.

## Local Package

Local training package:

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\crawler_strategy_model\crawler_strategy_project
```

Important files:

```text
crawler_strategy_project/configs/crawler_qwen2p5_3b_lora.yaml
crawler_strategy_project/data/crawler_sft_train.jsonl
crawler_strategy_project/data/crawler_sft_eval.jsonl
crawler_strategy_project/data/dataset_info.json
crawler_strategy_project/scripts/train_cu05.sh
crawler_strategy_project/scripts/deploy_to_cu05.py
crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora
```

## Remote Location

Remote training root on cu05:

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project
```

Final LoRA adapter on cu05:

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora
```

The final adapter has also been pulled back to:

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\crawler_strategy_model\crawler_strategy_project\outputs\qwen2p5-3b-crawler-lora
```

## Data

Source:

```text
数据集/pasa/data/sft_crawler/train.jsonl
```

Preparation summary:

```text
total_lines: 12989
kept: 9494
train_size: 8994
eval_size: 500
skipped_missing_sections: 3011
skipped_too_many_expands: 484
max_expands: 6
```

## Training

Configuration:

```text
base_model: Qwen/Qwen2.5-3B-Instruct
framework: LLaMA-Factory 0.9.5
finetuning_type: LoRA
lora_rank: 8
lora_alpha: 16
cutoff_len: 1024
precision: fp16
bf16: false
flash_attn: disabled
GPUs used: cu05 GPU 1-7
```

GPU 0 was avoided because other services were already occupying several GB of memory on that card.

Final metrics:

```text
epoch: 2.0
train_loss: 0.3578266545578285
eval_loss: 0.17109628021717072
train_runtime: 1223.6366 seconds
train_samples_per_second: 14.7
train_steps_per_second: 0.132
```

Checkpoints:

```text
checkpoint-50
checkpoint-100
checkpoint-150
checkpoint-162
final adapter at output root
```

## Notes

- The first full launch using GPU 0-7 hit OOM because GPU 0 already hosted other processes.
- The stable run automatically selected GPUs with less than 1GB used memory and completed on GPU 1-7.
- The current output is a LoRA adapter, not a merged standalone model.
- Existing services in the cluster are exposed from the `default` namespace with NodePort, reachable through `10.99.24.182:<nodePort>`. A crawler inference service can follow the same deployment pattern after adding a lightweight model-serving wrapper.
