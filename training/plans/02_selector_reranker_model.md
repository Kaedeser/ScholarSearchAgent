# Selector / Reranker 相关性识别模型训练路线

## 1. 模型定位

Selector / Reranker 是当前三条训练路线中最值得优先投入的一条。它负责判断：

```text
给定用户学术查询 query 和候选论文 title/abstract/chunk，这篇论文是否满足查询需求。
```

它对应 PaSa 系统中的 `Selector Agent`，也对应当前项目中的：

```text
relevance_ranking/ranking.py
```

当前 `CandidateRanker` 使用规则约束覆盖、检索分数、来源数量、引用量、年份等手工特征融合。训练后的 selector 应提供更强的语义相关性分数，作为最终排序主信号。

## 2. 训练目标

### 2.1 主目标

训练一个 query-paper pair 模型，输入：

```text
query
paper_title
paper_abstract
optional: evidence chunks
```

输出：

```text
relevance_score: 0-1
decision: True / False
reason: 可选
```

在线排序中主要使用 `relevance_score`，不要依赖长篇 reason。

### 2.2 目标能力

| 能力 | 说明 |
| --- | --- |
| 语义匹配 | query 词面和论文标题/摘要不完全重合时仍能识别 |
| 细粒度约束匹配 | 判断论文是否同时满足方法、任务、数据集、指标等约束 |
| 排除误召回 | 过滤只命中单个关键词但主题不符的论文 |
| 候选重排 | 对 BM25、dense、citation expansion 候选统一打分 |
| 可解释性 | 可选输出 reason，帮助结果解释和错误分析 |

## 3. 数据来源

### 3.1 已有 SFT 数据

核心数据：

```text
数据集/pasa/data/sft_selector/train.jsonl
数据集/pasa/data/sft_selector/test.jsonl
```

样本格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are an elite researcher ...\n\nSearched Paper:\nTitle: ...\nAbstract: ...\n\nUser Query: ...\n\nOutput format: Decision: True/False\nReason:... \nDecision:"
    },
    {
      "role": "assistant",
      "content": "False\nReason: ..."
    }
  ]
}
```

规模：

```text
sft_selector/train.jsonl: 约 19826 条
sft_selector/test.jsonl: 约 200 条
```

### 3.2 Gold 查询数据

用于构造 pairwise reranker 样本：

```text
数据集/pasa/data/AutoScholarQuery/train.jsonl
数据集/pasa/data/AutoScholarQuery/dev.jsonl
数据集/pasa/data/AutoScholarQuery/test.jsonl
数据集/pasa/data/RealScholarQuery/test.jsonl
ScholarSearchAgent/data_ingestion_indexing/data_processed/gold_labels.jsonl
ScholarSearchAgent/data_ingestion_indexing/data_processed/papers.jsonl
```

`gold_labels.jsonl` 中的 `qid -> paper_id` 是正样本来源。

### 3.3 候选负样本

负样本质量决定 reranker 上限。不要只随机采负样本，应混合三类：

| 负样本类型 | 来源 | 价值 |
| --- | --- | --- |
| random negative | 全论文库随机采样 | 学会基本区分主题 |
| BM25 hard negative | BM25 topK 但非 gold | 学会过滤词面误召回 |
| dense hard negative | dense topK 但非 gold | 学会过滤语义相近但不满足约束的论文 |
| citation negative | 引用扩展得到但非 gold | 学会过滤同领域但不相关论文 |

推荐比例：

```text
positive : hard_negative : random_negative = 1 : 3 : 1
```

如果训练 cross-encoder 二分类，每个 query 至少构造：

```text
正样本: 所有 gold papers
负样本: top100 召回中非 gold 论文采 5-20 篇
```

## 4. 训练样本格式

### 4.1 Cross-Encoder 二分类格式

适合 Sentence Transformers 或 Hugging Face。

```json
{
  "query": "Which papers discuss about Video Captioning?",
  "document": "Title: Deep Residual Learning for Image Recognition\nAbstract: ...",
  "label": 0
}
```

或：

```json
{
  "sentence1": "Which papers discuss about Video Captioning?",
  "sentence2": "Title: ... Abstract: ...",
  "score": 0.0
}
```

### 4.2 Pairwise / listwise 排序格式

适合后续更强 reranker。

```json
{
  "query": "Could you provide me some works employs image patches and superpixels in region-based methods for semantic segmentation?",
  "positive": "Title: CEREALS ... Abstract: ...",
  "negative": "Title: Mask R-CNN ... Abstract: ..."
}
```

### 4.3 LLM Selector SFT 格式

适合 Qwen LoRA 复现 PaSa selector。

```json
{
  "messages": [
    {
      "role": "user",
      "content": "User Query: ...\n\nSearched Paper:\nTitle: ...\nAbstract: ...\n\nOutput format: Decision: True/False\nReason:"
    },
    {
      "role": "assistant",
      "content": "True\nReason: ..."
    }
  ]
}
```

## 5. 模型与框架选型

### 5.1 首选：Cross-Encoder reranker

| 项 | 建议 |
| --- | --- |
| 框架 | Sentence Transformers CrossEncoder / Hugging Face Trainer |
| 基座 | `BAAI/bge-reranker-v2-m3`、`BAAI/bge-reranker-base`、`cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 输入长度 | 512 或 1024 |
| 输出 | relevance score |
| 优点 | 排序效果强，训练和部署简单 |
| 缺点 | 不能预计算论文向量，只适合 topK 重排 |

Cross-Encoder 会联合编码 query 和 document，适合 reranking；它不能像 bi-encoder 一样提前为每篇论文预计算 embedding，但通常排序更准。

### 5.2 次选：BGE / Qwen reranker 微调

| 项 | 建议 |
| --- | --- |
| 框架 | FlagEmbedding |
| 基座 | `BAAI/bge-reranker-v2-m3` |
| 任务 | reranker fine-tuning |
| 优点 | 与检索生态适配好 |
| 缺点 | 训练数据格式需要按 FlagEmbedding 规范转换 |

### 5.3 备选：Qwen Selector LLM

| 项 | 建议 |
| --- | --- |
| 框架 | LLaMA-Factory / TRL |
| 基座 | `Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`、`Qwen3-4B` |
| 训练 | LoRA / QLoRA SFT |
| 输出 | `Decision + Reason` |
| 优点 | 可解释能力强，能处理复杂 query |
| 缺点 | 推理慢，成本高，不适合作为所有候选的第一重排器 |

本项目建议：

```text
在线主 reranker: Cross-Encoder / BGE reranker
解释型 selector: Qwen LoRA 可选，仅对 top10 生成 reason
```

## 6. 训练逻辑

### 6.1 推荐 pipeline

```text
1. 用现有 BM25/dense 检索为每个 train query 召回 top100
2. 用 gold_labels 标注正负
3. 采 hard negatives
4. 合并 sft_selector 中已有 True/False 样本
5. 训练 cross-encoder reranker
6. 在 dev 上选择阈值和融合权重
7. 在 AutoScholarQuery test + RealScholarQuery test 上评估
8. 接入 CandidateRanker
```

### 6.2 标签构造

正样本：

```text
qid 对应 gold_labels 中的 paper_id
```

负样本：

```text
同一 qid 下召回 topK 但不在 gold paper_id 集合中的候选
```

注意：

- `gold_labels` 可以用于训练和评估标签。
- 不能把 `answer_arxiv_id` 或 gold 信息写入检索索引。
- test split 的 gold 只能用于离线评估，不参与训练。

### 6.3 输入文本拼接

第一版：

```text
Title: {title}
Abstract: {abstract}
```

第二版可加入 evidence chunks：

```text
Title: {title}
Abstract: {abstract}
Relevant Chunks:
1. {chunk_1}
2. {chunk_2}
```

如果 max_length 为 512，优先保留 title + abstract 前 350-450 tokens。复杂 query 可使用 1024，但训练成本更高。

## 7. 训练超参数建议

### 7.1 Cross-Encoder

```text
max_length: 512
learning_rate: 2e-5
epochs: 2-3
batch_size: 8-32
loss: BinaryCrossEntropy 或 CrossEntropy
warmup_ratio: 0.1
evaluation: 每 1000-2000 steps
early_stopping: dev NDCG@20 或 MRR
```

P100 16GB 上建议：

```text
batch_size: 4-8
gradient_accumulation_steps: 4
fp16: true
bf16: false
```

### 7.2 LLM Selector LoRA

```text
model: Qwen2.5-3B-Instruct 或 Qwen2.5-7B-Instruct
method: LoRA / QLoRA
max_seq_length: 1024
learning_rate: 1e-5 到 2e-5
epochs: 1-2
lora_rank: 8 或 16
precision: fp16
```

如果用 7B，P100 x2 只能考虑 QLoRA/LoRA + 小 batch，且要接受训练速度较慢。

## 8. 评估指标

### 8.1 模型级

| 指标 | 用途 |
| --- | --- |
| AUC | 判断二分类区分度 |
| Accuracy | 简单可读，但受正负比例影响大 |
| Precision / Recall / F1 | 选择 selector 阈值 |
| Calibration | 分数是否可用于融合 |

### 8.2 排序级

更重要的是排序指标：

```text
MRR@10
NDCG@10 / NDCG@20
Recall@20 / Recall@50 / Recall@100
Precision@10
```

### 8.3 端到端指标

接入后对比：

```text
baseline: CandidateRanker 当前规则分数
experiment: CandidateRanker + selector_score
```

关注：

- `Recall@50` 是否提升。
- `MRR` 是否提升。
- top10 是否更少出现明显误召回。
- 延迟是否可接受。

## 9. 接入方式

建议将 selector 分数加入 `CandidateRanker._score`。

融合公式第一版：

```text
final_score =
  0.55 * selector_score
  + 0.20 * retrieval_signal
  + 0.10 * keyword_overlap
  + 0.05 * source_confidence
  + 0.05 * citation_authority
  + 0.05 * recency_score
```

若 selector 分数质量足够高，可继续提高权重。

新增模块建议：

```text
relevance_ranking/selector_model.py
relevance_ranking/train_selector.py
relevance_ranking/data_builder.py
```

在线流程：

```text
BM25/dense/citation 召回 top200
  -> candidate normalization
  -> selector rerank top200
  -> final fusion
  -> result composition
```

为控制延迟，可分两级：

```text
cheap ranker top300 -> selector top80 -> final top20
```

## 10. 错误分析重点

每轮训练后必须抽样检查：

| 错误类型 | 说明 |
| --- | --- |
| 关键词误召回 | 只命中 video/image/KD 等词，但论文主题不符 |
| 约束缺失 | 满足任务但不满足方法或数据集 |
| 过窄判断 | 与 query 高度相关但表述不同，被判 False |
| 综述 query 覆盖不足 | survey_search 需要宽召回，不应过度精排 |
| 时间泄漏 | 查询时间之后的论文被误纳入 |

## 11. 推荐里程碑

```text
M1: 从 sft_selector 抽取 query-paper-label 三元组
M2: 用现有召回系统构造 hard negatives
M3: 训练 bge-reranker 或 CrossEncoder baseline
M4: 接入 relevance_ranking，做消融
M5: 可选训练 Qwen selector reason 模型，用于 top10 解释
```

## 12. 参考框架

- Sentence Transformers Cross-Encoder: https://www.sbert.net/docs/package_reference/cross_encoder/model.html
- Sentence Transformers reranking examples: https://www.sbert.net/examples/cross_encoder/applications/README.html
- FlagEmbedding: https://github.com/FlagOpen/FlagEmbedding
- BGE reranker tutorial: https://bge-model.com/tutorial/5_Reranking/5.2.html
- BAAI/bge-reranker-v2-m3: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Hugging Face Transformers fine-tuning: https://huggingface.co/docs/transformers/en/training
