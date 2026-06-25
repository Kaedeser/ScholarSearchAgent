# 爬虫策略模型训练与部署项目

本目录用于训练和部署 ScholarSearchAgent 的 crawler strategy model。模型负责根据用户研究问题、论文标题、摘要和 section 列表，预测下一步需要展开阅读的 section，输出格式为：

```text
[Expand]section title[Expand]section title[StopExpand]
```

如果不需要继续展开，则输出：

```text
[StopExpand]
```

## 技术路线

```text
训练框架: LLaMA-Factory 0.9.5
基础模型: Qwen/Qwen2.5-3B-Instruct
微调方式: LoRA SFT
数据格式: OpenAI messages
部署方式: K8s default 命名空间 + NodePort
```

## 目录说明

```text
configs/                         训练配置
data/                            处理后的训练集和评估集
data/raw/                        原始数据
packages/                        远端虚拟环境使用的本地 wheel
scripts/                         数据处理、训练、评估、部署脚本
service/                         HTTP 推理服务代码
k8s/                             K8s Deployment 和 Service YAML
outputs/qwen2p5-3b-crawler-lora  baseline 模型输出
outputs/qwen2p5-3b-crawler-lora-r16-e3  当前推荐模型输出
```

## 当前推荐模型

```text
outputs/qwen2p5-3b-crawler-lora-r16-e3
```

关键指标：

```text
eval_loss: 0.159352
exact_match: 0.160
parse_success_rate: 0.986
stop_accuracy: 0.628
section_f1: 0.3007
```

## 本地训练代码和模型保存

本地完整目录：

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\crawler_strategy_model\crawler_strategy_project
```

关键文件：

```text
configs/crawler_qwen2p5_3b_lora_r16_e3.yaml
scripts/train_cu05.sh
scripts/deploy_to_cu05.py
scripts/run_action_eval.sh
service/serve_crawler_strategy.py
k8s/crawler-strategy-service.yaml
outputs/qwen2p5-3b-crawler-lora-r16-e3/adapter_model.safetensors
```

## cu05 远端位置

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project
```

推荐模型：

```text
/data/model_train/crawler_strategy_model/crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3
```

## 训练命令

在 cu05 上启动 baseline：

```bash
bash scripts/train_cu05.sh
```

在 cu05 上启动优化版：

```bash
CONFIG_PATH=configs/crawler_qwen2p5_3b_lora_r16_e3.yaml bash scripts/start_training_background.sh
```

动作级评估：

```bash
ADAPTER_DIR=outputs/qwen2p5-3b-crawler-lora-r16-e3 bash scripts/run_action_eval.sh
```

## K8s 部署

部署文件：

```text
k8s/crawler-strategy-service.yaml
```

部署命令：

```bash
bash scripts/deploy_k8s_service.sh
```

当前服务信息：

```text
namespace: default
deployment: crawler-strategy-service
service: crawler-strategy-service
type: NodePort
nodePort: 32183
访问地址: http://10.99.24.182:32183
```

健康检查：

```bash
curl http://10.99.24.182:32183/health
```

预测请求：

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
