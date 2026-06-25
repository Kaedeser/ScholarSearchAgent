# Selector Reranker 模型训练报告

本文档记录 `selector_reranker_model` 的框架选型、数据构造、远端训练流程、调参结果、最终模型位置以及后续启动/使用方式。

## 1. 任务目标

根据 `training_plans/02_selector_reranker_model.md` 的要求，为 ScholarSearchAgent 构建一个论文检索结果重排模型。模型输入为：

- `query`：用户检索问题。
- `document`：候选论文文本，主要由 `Title + Abstract` 组成。

模型输出为一个相关性分数，用于对召回阶段得到的候选论文进行二次排序，或按阈值判断是否相关。

## 2. 框架与模型选型

最终采用开源项目 `sentence-transformers` 中的 `CrossEncoder` 训练方案。

本地源码位置：

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\selector_reranker_model\framework\sentence-transformers
```

选择 Cross-Encoder 的原因：

- Query 和候选论文文本联合编码，适合重排任务。
- 相比双塔向量模型，Cross-Encoder 通常在 topK 重排阶段准确率更高。
- `sentence-transformers` 提供成熟的训练、评估和模型保存能力，便于快速接入。

主要尝试过的基座模型：

| 模型 | 说明 |
| --- | --- |
| `BAAI/bge-reranker-base` | 中文/英文检索重排常用基座，作为初始主方案 |
| `BAAI/bge-reranker-large` | 更大规模 reranker，最终主力模型 |
| `BAAI/bge-reranker-v2-m3` | 多语言 reranker，对比实验使用 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 轻量英文 reranker，对比实验使用 |

## 3. 数据构造

原始数据来源：

```text
F:\中国研究生人工智能大赛\数据集\pasa\data\sft_selector
```

远端训练数据位置：

```text
/home/model_train/selector_reranker_model/data/processed
```

训练脚本将原始 PaSa selector 数据转换为 Cross-Encoder 所需 JSONL 格式：

```json
{
  "query": "Which papers discuss about Video Captioning?",
  "document": "Title: ...\nAbstract: ...",
  "label": 1
}
```

最终训练/验证数据规模：

| 数据集 | 样本数 | 正样本 | 负样本 |
| --- | ---: | ---: | ---: |
| train | 19,826 | 9,913 | 9,913 |
| dev | 200 | 98 | 102 |

本地重新构造数据命令：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent\train\selector_reranker_model
$env:PYTHONPATH="$PWD\src;$PWD\framework\sentence-transformers"
python -m selector_reranker.data_builder `
  --pasa-data-dir ..\..\..\数据集\pasa\data `
  --output-dir data\processed `
  --max-abstract-chars 3500
```

## 4. 训练环境

本地项目目录：

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\selector_reranker_model
```

远端训练目录：

```text
/home/model_train/selector_reranker_model
```

远端虚拟环境：

```text
/home/model_train/py-train2
```

说明：

- `/home/model_train/py-train` 当时正在训练其他模型，因此本任务使用 `/home/model_train/py-train2`。
- `/home` 空间不足时，将训练输出迁移到 `/data/model_train/selector_reranker_model/outputs`，并在 `/home/model_train/selector_reranker_model/outputs` 建立软链接。
- 训练使用 V100 GPU，启用 fp16。

安装依赖：

```bash
cd /home/model_train/selector_reranker_model
/home/model_train/py-train2/bin/python -m pip install -r requirements.txt
```

远端训练前需要设置：

```bash
export PYTHONPATH=/home/model_train/selector_reranker_model/src:/home/model_train/selector_reranker_model/framework/sentence-transformers:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
```

## 5. 训练脚本调整

核心训练入口：

```text
src/selector_reranker/train_cross_encoder.py
```

主要调整：

- 使用 `CrossEncoder.old_fit`，避免 `datasets/pyarrow` 依赖问题。
- 强制 `DataLoader(num_workers=0)`，避免 CUDA fork 报错。
- 支持按不同验证指标保存最佳 checkpoint：
  - `average_precision`
  - `f1`
  - `accuracy`
  - `min_accuracy_f1`
  - `mean_accuracy_f1`
- 训练结束后自动保存：
  - `metrics.json`
  - `final/metrics.json`
  - `final/` 可直接加载的 CrossEncoder 模型目录。

最终为了同时提升 Accuracy 和 F1，新增了 `mean_accuracy_f1` 作为 checkpoint 选择指标。

## 6. 训练过程与调参结果

### 6.1 初始 baseline

初始模型：

```text
BAAI/bge-reranker-base
```

结果：

| Run | Accuracy | F1 | Average Precision |
| --- | ---: | ---: | ---: |
| `bge-reranker-base-pasa` | 74.50% | 76.50% | 80.44% |

### 6.2 第一轮调参

| Run | Accuracy | F1 | Average Precision |
| --- | ---: | ---: | ---: |
| `bge-base-fresh-ep4-lr1e-5-bs8` | 75.50% | 77.68% | 81.03% |
| `bge-base-continue-ep3-lr5e-6-bs8` | 75.00% | 76.52% | 82.05% |
| `minilm-msmarco-ep4-lr2e-5-bs32` | 73.00% | 74.89% | 80.23% |
| `bge-base-f1select-continue-ep2-lr3e-6-bs8` | 75.00% | 77.39% | 80.78% |

结论：`bge-reranker-base` 能稳定提升，但很难把 Accuracy/F1 同时推到 80 以上。

### 6.3 大模型实验

使用：

```text
BAAI/bge-reranker-large
```

关键配置：

```bash
--max-length 512
--epochs 3
--batch-size 4
--learning-rate 7e-6
--selection-metric f1
--fp16 --no-bf16
```

结果：

| Run | Accuracy | F1 | Average Precision |
| --- | ---: | ---: | ---: |
| `bge-large-f1select-ep3-lr7e-6-bs4` | 78.00% | 80.36% | 85.92% |

该模型 F1 首次超过 80%，但 Accuracy 只有 78%。

### 6.4 其他对比实验

| Run | Accuracy | F1 | Average Precision |
| --- | ---: | ---: | ---: |
| `bge-v2-m3-ep2-lr1e-5-bs2` | 79.50% | 79.00% | 86.26% |

该模型 AP 较高，但 F1 未超过 `bge-reranker-large`。

### 6.5 最终均衡精调

为了让 Accuracy 和 F1 同时达到 80 附近，使用已经完成的 `bge-large-f1select` 模型继续训练 1 epoch，并将 checkpoint 选择指标改为 `mean_accuracy_f1`。

最终训练命令：

```bash
cd /home/model_train/selector_reranker_model
export PYTHONPATH=/home/model_train/selector_reranker_model/src:/home/model_train/selector_reranker_model/framework/sentence-transformers:${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=2 /home/model_train/py-train2/bin/python -m selector_reranker.train_cross_encoder \
  --train-file data/processed/train.jsonl \
  --dev-file data/processed/dev.jsonl \
  --output-dir outputs/bge-large-continue-mean-ep1-lr2e-6-bs4 \
  --model-name outputs/bge-large-f1select-ep3-lr7e-6-bs4/final \
  --max-length 512 \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-6 \
  --warmup-ratio 0.05 \
  --eval-steps 500 \
  --save-steps 500 \
  --logging-steps 100 \
  --dataloader-num-workers 0 \
  --selection-metric mean_accuracy_f1 \
  --fp16 --no-bf16
```

最终结果：

| Run | Accuracy | F1 | Precision | Recall | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bge-large-continue-mean-ep1-lr2e-6-bs4` | 80.50% | 80.00% | 80.41% | 79.59% | 82.51% |

最终决策：

- 如果只追求 F1，`bge-large-f1select-ep3-lr7e-6-bs4` 更高，F1 为 80.36%。
- 如果要求 Accuracy 和 F1 都到 80 左右，选择 `bge-large-continue-mean-ep1-lr2e-6-bs4`。
- 本次交付模型采用均衡版，即 Accuracy 80.50%、F1 80.00% 的模型。

## 7. 最终模型位置

本地最终模型目录：

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\selector_reranker_model\outputs\best_selector_reranker_model
```

目录内关键文件：

```text
config.json
config_sentence_transformers.json
metrics.json
model.safetensors
modules.json
README.md
sentence_bert_config.json
special_tokens_map.json
tokenizer_config.json
tokenizer.json
```

远端最终模型目录：

```text
/home/model_train/selector_reranker_model/outputs/bge-large-continue-mean-ep1-lr2e-6-bs4/final
```

训练记录与对比结果已拉回本地：

```text
F:\中国研究生人工智能大赛\ScholarSearchAgent\train\selector_reranker_model\outputs\remote_training_artifacts
```

其中：

| 文件 | 说明 |
| --- | --- |
| `best_model_manifest.json` | 最终模型选择说明和指标 |
| `balanced_eval.csv` | 最终均衡版模型验证过程 |
| `high_f1_eval.csv` | 最高 F1 版模型验证过程 |
| `tuning_summary.json` | 第一轮调参结果 |
| `ensemble_search_results.json` | 模型分数融合搜索结果 |

## 8. 本地评估方式

进入模型目录：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent\train\selector_reranker_model
$env:PYTHONPATH="$PWD\src;$PWD\framework\sentence-transformers"
```

评估最终模型：

```powershell
python -m selector_reranker.evaluate_cross_encoder `
  --model-dir outputs\best_selector_reranker_model `
  --eval-file data\processed\dev.jsonl `
  --batch-size 16
```

也可以直接查看最终指标：

```powershell
Get-Content outputs\best_selector_reranker_model\metrics.json
```

## 9. 推理与接入方式

### 9.1 Python 单条预测

```python
from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder(
    r"F:\中国研究生人工智能大赛\ScholarSearchAgent\train\selector_reranker_model\outputs\best_selector_reranker_model",
    max_length=512,
)

query = "Which papers discuss video captioning?"
document = "Title: Video Captioning with Transformer...\nAbstract: ..."

score = float(model.predict([(query, document)], batch_size=1)[0])
print(score)
```

### 9.2 批量重排

```python
from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder(
    r"F:\中国研究生人工智能大赛\ScholarSearchAgent\train\selector_reranker_model\outputs\best_selector_reranker_model",
    max_length=512,
)

def rerank(query: str, papers: list[dict], top_k: int = 10) -> list[dict]:
    pairs = [
        (
            query,
            f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}",
        )
        for paper in papers
    ]
    scores = model.predict(pairs, batch_size=16)
    ranked = []
    for paper, score in zip(papers, scores):
        item = dict(paper)
        item["rerank_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]
```

### 9.3 阈值判断

最终均衡版模型在 dev 集上的推荐阈值为：

```text
0.991881787776947
```

示例：

```python
threshold = 0.991881787776947
is_relevant = score >= threshold
```

注意：

- 阈值来自当前 dev 集，dev 集只有 200 条，实际线上使用时建议结合业务数据重新校准。
- 若只用于排序，可以不使用阈值，直接按 `rerank_score` 从高到低排序。

## 10. 远端重新训练方式

基础训练可以使用：

```bash
cd /home/model_train/selector_reranker_model
bash scripts/train_cu05.sh /home/model_train/selector_reranker_model /home/model_train/pasa/data
```

如果需要复现最终均衡模型，建议先训练或准备：

```text
outputs/bge-large-f1select-ep3-lr7e-6-bs4/final
```

然后运行第 6.5 节的继续训练命令。

训练完成后，将模型目录拉回本地即可：

```text
/home/model_train/selector_reranker_model/outputs/<run-name>/final
```

## 11. 当前结论与后续建议

当前交付模型已经达到：

- Accuracy：80.50%
- F1：80.00%

后续如果还要继续冲更高指标，建议优先尝试：

1. 扩大 dev/test 集，避免 200 条验证集带来的指标波动。
2. 针对当前误判样本做 hard negative mining。
3. 增加 query-document 的结构化输入字段，例如 venue、year、keywords、citation context。
4. 保留 `bge-large-f1select` 和最终均衡版两个模型，根据比赛指标偏好选择：
   - 偏 F1：使用 `bge-large-f1select-ep3-lr7e-6-bs4`。
   - Accuracy/F1 均衡：使用 `best_selector_reranker_model`。

