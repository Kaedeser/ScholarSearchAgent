# Selector Reranker Model

本目录放置根据 `training_plans/02_selector_reranker_model.md` 选择并调整好的开源训练方案。

## 选型

- 开源框架：`sentence-transformers`
- 本地源码：`framework/sentence-transformers`
- 训练形态：Cross-Encoder 二分类 reranker
- 默认基座：`BAAI/bge-reranker-base`
- 轻量烟测基座：`cross-encoder/ms-marco-MiniLM-L-6-v2`

Cross-Encoder 会联合编码 `query` 和 `Title + Abstract`，输出一个相关性分数。它不能预计算论文向量，但非常适合作为 BM25/dense/citation 召回后的 topK 重排器。

## 目录

```text
framework/sentence-transformers/  # 选定开源项目源码
src/selector_reranker/            # PaSa/ScholarSearchAgent 数据适配与训练入口
scripts/                          # 本地/远端运行脚本
requirements.txt                  # 训练依赖
```

## 本地构造数据

从项目根目录运行：

```powershell
cd ScholarSearchAgent/train/selector_reranker_model
python -m selector_reranker.data_builder `
  --pasa-data-dir ..\..\..\数据集\pasa\data `
  --output-dir data\processed
```

如果只是快速验证：

```powershell
python -m selector_reranker.data_builder `
  --pasa-data-dir ..\..\..\数据集\pasa\data `
  --output-dir data\processed_smoke `
  --max-train 200 `
  --max-dev 40
```

## 远端训练命令

在 `cu05` 上放到 `/home/model_train/selector_reranker_model` 后：

```bash
cd /home/model_train/selector_reranker_model
bash scripts/train_cu05.sh /home/model_train/selector_reranker_model /home/model_train/pasa/data
```

脚本会：

1. 创建或复用 `/home/model_train/py-train2` 虚拟环境。
2. 在 `py-train2` 中安装本地 `framework/sentence-transformers` 源码和训练依赖。
2. 从 `sft_selector/train.jsonl` 与 `sft_selector/test.jsonl` 构造 cross-encoder 训练集。
3. 使用 `BAAI/bge-reranker-base` 在 P100 友好的 fp16、小 batch、梯度累积配置下训练。
4. 将模型保存到 `outputs/bge-reranker-base-pasa/final`。

## 数据格式

中间数据为 JSONL：

```json
{
  "query": "Which papers discuss about Video Captioning?",
  "document": "Title: ...\nAbstract: ...",
  "label": 0,
  "source": "sft_selector",
  "title": "..."
}
```

训练时只使用 `query`、`document`、`label` 三列。
