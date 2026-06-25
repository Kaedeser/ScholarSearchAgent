# 爬虫策略模型优化与部署结果

## 结论

本轮已完成参数优化、动作级评估、本地结果保存和 K8s 部署。当前推荐使用优化后的最终 LoRA 适配器：

```text
crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3
```

该模型已经在 cu05 上通过 K8s `default` 命名空间部署，并使用 NodePort 暴露：

```text
http://10.99.24.182:32183
```

可用接口：

```text
GET  /health
POST /predict
POST /generate
```

## 本地文件保存确认

训练代码、配置、数据切分结果、模型文件和评估结果均已保存在本地目录：

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\crawler_strategy_model\crawler_strategy_project
```

关键本地文件如下：

```text
crawler_strategy_project/configs/crawler_qwen2p5_3b_lora_r16_e3.yaml
crawler_strategy_project/scripts/train_cu05.sh
crawler_strategy_project/scripts/deploy_to_cu05.py
crawler_strategy_project/scripts/generate_eval_predictions.py
crawler_strategy_project/scripts/run_action_eval.sh
crawler_strategy_project/service/serve_crawler_strategy.py
crawler_strategy_project/k8s/crawler-strategy-service.yaml
crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3/adapter_model.safetensors
crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3/action_metrics.json
crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3/all_results.json
```

本地优化后模型适配器文件大小：

```text
adapter_model.safetensors: 119801528 bytes
```

## 远端文件位置

cu05 上的训练和部署目录：

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project
```

推荐模型远端位置：

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3
```

基础模型缓存位置：

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project/.cache/modelscope/models/Qwen/Qwen2___5-3B-Instruct
```

## 优化训练配置

本轮优化使用的配置文件：

```text
crawler_strategy_project/configs/crawler_qwen2p5_3b_lora_r16_e3.yaml
```

主要参数：

```text
base_model: Qwen/Qwen2.5-3B-Instruct
finetuning_type: LoRA
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 5e-5
num_train_epochs: 3
gradient_accumulation_steps: 8
cutoff_len: 1024
effective_batch_size: 56
total_optimization_steps: 483
```

训练已在 cu05 上完成，使用 GPU 1-7，避开了已有服务占用的 GPU 0。

## 训练指标对比

| 模型 | eval_loss | train_loss | epoch | 训练耗时 |
| --- | ---: | ---: | ---: | ---: |
| baseline rank8/e2 | 0.171096 | 0.357827 | 2.0 | 1223.6s |
| tuned rank16/e3 final | 0.159352 | 0.263458 | 3.0 | 1916.7s |
| tuned rank16/e3 step350 | 0.156593 | - | 2.17 | - |

优化后最终模型相对 baseline 的 `eval_loss` 降低约 `6.86%`。

## 动作级评估对比

评估集为同一份 `crawler_sft_eval.jsonl`，共 500 条样本。评估方式为贪心生成，然后解析 `[Expand]... [StopExpand]` 动作串。

| 模型 | exact_match | parse_success | stop_accuracy | section_precision | section_recall | section_f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline rank8/e2 final | 0.124 | 0.924 | 0.600 | 0.2321 | 0.3936 | 0.2920 |
| tuned rank16/e3 final | 0.160 | 0.986 | 0.628 | 0.2438 | 0.3924 | 0.3007 |
| tuned rank16/e3 step350 | 0.190 | 0.988 | 0.634 | 0.2437 | 0.3764 | 0.2959 |

虽然 step350 的 token loss 最低，但它的 `section_f1` 低于最终模型。因此当前按爬虫策略动作效果选择：

```text
outputs/qwen2p5-3b-crawler-lora-r16-e3
```

相对 baseline，推荐模型的主要提升：

```text
exact_match: 0.124 -> 0.160
parse_success_rate: 0.924 -> 0.986
stop_accuracy: 0.600 -> 0.628
section_precision: 0.2321 -> 0.2438
section_f1: 0.2920 -> 0.3007
```

## K8s 部署信息

部署命名空间：

```text
default
```

Deployment：

```text
crawler-strategy-service
```

Service：

```text
crawler-strategy-service
```

Service 类型：

```text
NodePort
```

容器端口：

```text
8000
```

NodePort：

```text
32183
```

访问地址：

```text
http://10.99.24.182:32183
```

K8s 部署文件：

```text
crawler_strategy_project/k8s/crawler-strategy-service.yaml
```

部署脚本：

```text
crawler_strategy_project/scripts/deploy_k8s_service.sh
```

当前 K8s 状态：

```text
service/crawler-strategy-service: NodePort 8000:32183/TCP
pod/crawler-strategy-service-74df5965f8-zvzjc: Running, Ready 1/1
node: 11.11.11.5
pod ip: 172.21.252.241
```

## 接口测试结果

健康检查：

```bash
curl http://10.99.24.182:32183/health
```

返回：

```json
{"status": "ok", "model_loaded": true}
```

预测接口示例：

```bash
curl -H "Content-Type: application/json" \
  --data '{
    "query": "What studies have further simplified the GNNs using the technique proposed by ChebNet?",
    "title": "An Overview on the Application of Graph Neural Networks in Wireless Networks",
    "abstract": "This overview introduces graph neural networks and their applications in wireless networks.",
    "sections": [
      "I INTRODUCTION",
      "III Paradigms of GNNs III-A Graph Convolutional Neural Networks",
      "III Paradigms of GNNs III-B Graph Attention Networks",
      "IV Applications in Wireless Networks IV-A Resource Allocation"
    ]
  }' \
  http://10.99.24.182:32183/predict
```

实测返回：

```json
{
  "prediction": "[StopExpand]",
  "parse_success": true,
  "sections": [],
  "latency_ms": 1508.68,
  "model": {
    "base_model": "/app/.cache/modelscope/models/Qwen/Qwen2___5-3B-Instruct",
    "adapter_dir": "/app/outputs/qwen2p5-3b-crawler-lora-r16-e3"
  }
}
```

## 后续建议

当前模型已经完成可访问部署。下一步如果继续提高效果，建议优先做数据和推理策略优化，而不是单纯继续增加 epoch：

```text
1. 针对 StopExpand / Expand 的输出分布做数据均衡。
2. 增加更贴近真实 ScholarSearch 流程的端到端检索评估。
3. 对输出动作增加业务规则校验，例如过滤不存在的 section、限制最大展开数量。
4. 如果 3B 模型仍不足，再考虑 Qwen2.5-7B QLoRA。
```
