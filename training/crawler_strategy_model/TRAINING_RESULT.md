# 爬虫策略模型训练结果

## 项目选择

本任务选择的开源训练框架为：

```text
LLaMA-Factory 0.9.5
```

选择原因：

- 与训练计划中的路线一致，适合 `Qwen Instruct + LoRA SFT`。
- 支持 OpenAI `messages` 格式，PaSa 爬虫策略 SFT 数据可以尽量保持原格式。
- 能在 cu05 的 V100 16GB GPU 上使用 fp16 LoRA 稳定训练。

## 本地保存位置

训练代码、配置、数据、模型输出和部署文件均保存在本地：

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\crawler_strategy_model\crawler_strategy_project
```

关键文件：

```text
crawler_strategy_project/configs/crawler_qwen2p5_3b_lora.yaml
crawler_strategy_project/configs/crawler_qwen2p5_3b_lora_r16_e3.yaml
crawler_strategy_project/data/crawler_sft_train.jsonl
crawler_strategy_project/data/crawler_sft_eval.jsonl
crawler_strategy_project/data/dataset_info.json
crawler_strategy_project/scripts/train_cu05.sh
crawler_strategy_project/scripts/deploy_to_cu05.py
crawler_strategy_project/scripts/run_action_eval.sh
crawler_strategy_project/service/serve_crawler_strategy.py
crawler_strategy_project/k8s/crawler-strategy-service.yaml
crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3
```

当前推荐模型文件：

```text
crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3/adapter_model.safetensors
```

## 远端训练位置

cu05 上的训练根目录：

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project
```

baseline LoRA 输出：

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora
```

优化后推荐 LoRA 输出：

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3
```

## 数据处理

数据来源：

```text
数据集/pasa/data/sft_crawler/train.jsonl
```

处理统计：

```text
total_lines: 12989
kept: 9494
train_size: 8994
eval_size: 500
skipped_missing_sections: 3011
skipped_too_many_expands: 484
max_expands: 6
```

## Baseline 训练结果

baseline 配置：

```text
base_model: Qwen/Qwen2.5-3B-Instruct
framework: LLaMA-Factory 0.9.5
finetuning_type: LoRA
lora_rank: 8
lora_alpha: 16
cutoff_len: 1024
num_train_epochs: 2
gradient_accumulation_steps: 16
precision: fp16
bf16: false
flash_attn: disabled
GPUs used: cu05 GPU 1-7
```

baseline 指标：

```text
epoch: 2.0
train_loss: 0.3578266545578285
eval_loss: 0.17109628021717072
train_runtime: 1223.6366 seconds
train_samples_per_second: 14.7
train_steps_per_second: 0.132
```

## 参数优化结果

优化配置：

```text
config: crawler_strategy_project/configs/crawler_qwen2p5_3b_lora_r16_e3.yaml
output: crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3
lora_rank: 16
lora_alpha: 32
learning_rate: 5e-5
num_train_epochs: 3
gradient_accumulation_steps: 8
effective_batch_size: 56
total_optimization_steps: 483
```

优化后指标：

```text
epoch: 3.0
train_loss: 0.2634584511288945
eval_loss: 0.1593523472547531
train_runtime: 1916.7011 seconds
```

相对 baseline：

```text
eval_loss: 0.171096 -> 0.159352，下降约 6.86%
section_f1: 0.2920 -> 0.3007，提升约 3.0%
parse_success_rate: 0.924 -> 0.986
exact_match: 0.124 -> 0.160
```

完整优化对比见：

```text
OPTIMIZATION_RESULT.md
```

## K8s 部署结果

模型已经部署在 K8s 集群的 `default` 命名空间。

```text
Deployment: crawler-strategy-service
Service: crawler-strategy-service
Service type: NodePort
containerPort: 8000
nodePort: 32183
访问地址: http://10.99.24.182:32183
```

健康检查已通过：

```bash
curl http://10.99.24.182:32183/health
```

返回：

```json
{"status": "ok", "model_loaded": true}
```

预测接口已通过：

```text
POST http://10.99.24.182:32183/predict
```

实测返回合法动作串：

```json
{
  "prediction": "[StopExpand]",
  "parse_success": true,
  "sections": [],
  "latency_ms": 1508.68
}
```
