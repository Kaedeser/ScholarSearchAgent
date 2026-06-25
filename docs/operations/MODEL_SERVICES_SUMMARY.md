# ScholarSearchAgent 三个模型用途与访问地址汇总

整理时间：2026-06-24
范围：`ScholarSearchAgent/train` 下的 `query_intent_model`、`selector_reranker_model`、`crawler_strategy_model`。

> 说明：以下访问地址均来自各模型目录内现有部署文档，属于 cu05/K8s `default` 命名空间暴露的 NodePort 服务。若服务不可达，优先检查 cu05 节点、K8s Pod/Service 状态以及当前网络是否能访问 `10.99.24.182`。

## 1. 总览

| 模型目录 | 模型名称 | 主要用途 | 当前访问地址 | 核心接口 |
| --- | --- | --- | --- | --- |
| `query_intent_model` | Query Intent Service | 判断用户输入是否为论文检索需求，并进一步识别论文检索意图。 | `http://10.99.24.182:22436` | `GET /health`、`POST /predict` |
| `selector_reranker_model` | Selector Reranker Service | 对召回得到的候选论文进行相关性重排，输出论文相关性分数和排序结果。 | `http://10.99.24.182:32082` | `GET /health`、`POST /rerank` |
| `crawler_strategy_model` | Crawler Strategy Service | 根据问题、论文标题/摘要和章节列表，预测下一步应展开阅读哪些 section。 | `http://10.99.24.182:32183` | `GET /health`、`POST /predict`、`POST /generate` |

健康检查结果（2026-06-24 实测）：

| 服务 | `/health` 状态 | 返回摘要 |
| --- | --- | --- |
| Query Intent Service | 200 OK | `{"status": "ok", "service": "query-intent-service"}` |
| Selector Reranker Service | 200 OK | `{"status": "ok", "model_loaded": true, "device": "cuda"}` |
| Crawler Strategy Service | 200 OK | `{"status": "ok", "model_loaded": true}` |

## 2. Query Intent Model

### 用处

该模型是 ScholarSearchAgent 的查询入口分类器，包含两个子模型：

- `query_gate_biobert`：二分类 gate，判断输入是否为论文检索请求。
  - 输出标签：`paper_search`、`non_paper_search`
- `intent_biobert`：论文检索意图分类器，仅对 `paper_search` 输入继续分类。
  - 输出标签：`application_search`、`citation_trace`、`comparison_search`、`dataset_search`、`mechanism_search`、`method_search`、`metric_search`、`survey_search`

服务的 `auto` 模式会先运行 gate；只有 gate 结果为 `paper_search` 时，才继续运行 intent 模型。非论文检索请求会返回 `intent: null`。

### 访问地址与接口

```text
Base URL: http://10.99.24.182:22436
Health:   GET  /health
Predict:  POST /predict
```

示例：

```bash
curl -X POST http://10.99.24.182:22436/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "auto",
    "texts": [
      "Could you recommend recent papers about neural information retrieval?",
      "Write a Python function to sort a list."
    ]
  }'
```

`mode` 支持：

- `auto`：先 gate，再按需 intent。
- `gate`：只判断是否为论文检索。
- `intent`：只做论文检索意图分类。

### 部署与模型信息

```text
Kubernetes namespace: default
Deployment: query-intent-service
Service: query-intent-service
Service type: NodePort
Container port: 8080
NodePort: 22436
Remote app root: /data/csp/query-intent-service
```

本地模型产物：

```text
query_intent_model/outputs/query_gate_biobert
query_intent_model/outputs/intent_biobert
query_intent_model/artifacts/remote_model_manifest.json
```

关键指标：

| 子模型 | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| `query_gate_biobert` | 0.997143 | 0.996205 |
| `intent_biobert` | 0.997143 | 0.923703 |

## 3. Selector Reranker Model

### 用处

该模型用于论文召回后的 topK 重排。输入为：

- `query`：用户检索问题。
- `document`：候选论文文本，主要由 `Title + Abstract` 组成。

模型输出相关性分数，可用于：

- 按 `rerank_score` 从高到低重排候选论文。
- 结合阈值判断候选论文是否相关。

当前推荐阈值：

```text
0.991881787776947
```

如果只用于排序，可以不使用阈值，直接按分数降序排列。

### 访问地址与接口

```text
Base URL: http://10.99.24.182:32082
Health:   GET  /health
Rerank:   POST /rerank
```

示例：

```bash
curl -X POST http://10.99.24.182:32082/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Which papers discuss video captioning?",
    "top_k": 2,
    "documents": [
      {
        "id": "p1",
        "title": "Video Captioning with Transformers",
        "abstract": "A method for generating natural language descriptions for videos."
      },
      {
        "id": "p2",
        "title": "Database Indexing",
        "abstract": "An index structure for transaction processing."
      }
    ]
  }'
```

### 部署与模型信息

```text
Kubernetes namespace: default
Deployment: selector-reranker
Service: selector-reranker
Service type: NodePort
Container port: 8000
NodePort: 32082
Remote deploy root: /data/csp/selector-reranker
Remote model mount: /data/csp/selector-reranker/model
```

模型路线：

```text
Framework: sentence-transformers CrossEncoder
Final base: BAAI/bge-reranker-large
Local final model: selector_reranker_model/outputs/best_selector_reranker_model
Remote final training model: /home/model_train/selector_reranker_model/outputs/bge-large-continue-mean-ep1-lr2e-6-bs4/final
```

最终交付模型关键指标：

| Accuracy | F1 | Precision | Recall | Average Precision |
| ---: | ---: | ---: | ---: | ---: |
| 80.50% | 80.00% | 80.41% | 79.59% | 82.51% |

## 4. Crawler Strategy Model

### 用处

该模型用于控制 ScholarSearchAgent 的论文内容展开策略。它会根据：

- 用户研究问题 `query`
- 当前论文标题 `title`
- 当前论文摘要 `abstract`
- 可选章节列表 `sections`

预测下一步是否继续展开阅读 section，并给出要展开的 section 名称。

输出动作格式：

```text
[Expand]section title[Expand]section title[StopExpand]
```

如果不需要继续展开，则输出：

```text
[StopExpand]
```

### 访问地址与接口

```text
Base URL: http://10.99.24.182:32183
Health:   GET  /health
Predict:  POST /predict
Generate: POST /generate
```

示例：

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

实测返回样例：

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

### 部署与模型信息

```text
Kubernetes namespace: default
Deployment: crawler-strategy-service
Service: crawler-strategy-service
Service type: NodePort
Container port: 8000
NodePort: 32183
Remote project root: /data/model_train/crawler_strategy_model/crawler_strategy_project
```

模型路线：

```text
Framework: LLaMA-Factory 0.9.5
Base model: Qwen/Qwen2.5-3B-Instruct
Fine-tuning: LoRA SFT
Recommended adapter: crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3
```

关键指标：

| eval_loss | exact_match | parse_success | stop_accuracy | section_f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.159352 | 0.160 | 0.986 | 0.628 | 0.3007 |

相对 baseline 的主要提升：

```text
eval_loss: 0.171096 -> 0.159352
exact_match: 0.124 -> 0.160
parse_success_rate: 0.924 -> 0.986
stop_accuracy: 0.600 -> 0.628
section_f1: 0.2920 -> 0.3007
```

## 5. 推荐调用顺序

在 ScholarSearchAgent 主流程中，三个模型可以按如下顺序协作：

1. `query_intent_model`：先判断用户输入是否属于论文检索；若是，再识别检索意图。
2. 检索/召回模块：根据意图执行关键词、向量、引用或其他召回策略。
3. `selector_reranker_model`：对召回候选论文进行二次重排，得到更相关的 topK 论文。
4. `crawler_strategy_model`：在需要继续阅读论文正文时，决定是否展开 section 以及展开哪些 section。

## 6. 信息来源

本汇总整理自以下已有文档和配置：

- `query_intent_model/README.md`
- `query_intent_model/TRAINING_RESULT_AND_DEPLOYMENT.md`
- `query_intent_model/DEPLOYMENT_NODEPORT.md`
- `query_intent_model/deploy/query-intent-service.yaml`
- `selector_reranker_model/TRAINING_REPORT.md`
- `selector_reranker_model/deploy/README_DEPLOY.md`
- `selector_reranker_model/deploy/k8s/selector-reranker.yaml`
- `crawler_strategy_model/crawler_strategy_project/README.md`
- `crawler_strategy_model/TRAINING_RESULT.md`
- `crawler_strategy_model/OPTIMIZATION_RESULT.md`
- `crawler_strategy_model/crawler_strategy_project/k8s/crawler-strategy-service.yaml`
