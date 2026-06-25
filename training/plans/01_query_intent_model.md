# 查询意图识别与查询理解模型训练路线

## 1. 模型定位

查询意图识别模型负责把用户自然语言问题转换成可检索、可规划、可评估的结构化查询意图。它不是最终排序模型，而是检索流水线的入口控制层。

在当前 `ScholarSearchAgent` 中，它应替换或增强：

```text
query_understanding_decomposition/query.py
```

当前实现是规则版 `QueryParser`，适合作为 baseline 和兜底。训练后的模型应输出更稳定的：

- 是否属于学术论文检索问题。
- 检索意图类型。
- 研究领域、任务、方法、数据集、指标、时间、会议等约束。
- 多路召回子查询。
- 排除条件和软约束。

## 2. 训练目标

### 2.1 主目标

训练一个轻量查询理解模型，将输入 query 映射到结构化 JSON。

推荐输出格式：

```json
{
  "is_paper_search": true,
  "intent_type": "method_search",
  "research_fields": ["computer vision", "semantic segmentation"],
  "must_have_constraints": ["image patches", "superpixels", "region-based methods"],
  "soft_constraints": ["active learning", "annotation cost"],
  "excluded_meanings": [],
  "time_range": {"start_year": null, "end_year": null},
  "venues": [],
  "sub_queries": [
    "image patches superpixels region-based semantic segmentation",
    "region based active learning semantic segmentation"
  ]
}
```

### 2.2 可拆分子任务

| 子任务 | 类型 | 是否必做 | 说明 |
| --- | --- | --- | --- |
| 检索问题识别 | 二分类 | P0 | 区分论文检索问题与非检索问题 |
| 检索意图分类 | 单标签/多标签 | P0 | `survey_search`、`method_search`、`dataset_search` 等 |
| 约束抽取 | 序列标注或生成式抽取 | P0 | 抽取任务、方法、数据集、指标、年份、会议 |
| 子查询生成 | 生成式 | P1 | 为 BM25、dense retrieval、KG expansion 生成不同召回 query |
| 排除条件识别 | 生成式/规则增强 | P1 | 处理 `not about`、`exclude`、`except` 等 |

## 3. 标签体系建议

### 3.1 一级标签

```text
paper_search
non_paper_search
```

### 3.2 检索意图类型

建议先用 8 类，避免早期标签过细导致样本稀疏。

| 标签 | 含义 | 示例 |
| --- | --- | --- |
| `survey_search` | 找综述/领域相关工作 | What works are related to image retrieval? |
| `method_search` | 找提出某类方法的论文 | papers that proposed hierarchical neural models |
| `dataset_search` | 找使用/提出数据集的论文 | works using COCO for video captioning |
| `metric_search` | 找关注指标/性能比较的论文 | works reporting recall improvement |
| `mechanism_search` | 找机制、原理、解释分析 | understanding the working mechanism of KD |
| `citation_trace` | 找某思想、算法、术语来源 | Who proposed OAC? |
| `comparison_search` | 找对比某些方法的论文 | compare dense and sparse retrieval |
| `application_search` | 找应用场景相关论文 | papers applying GNN to anomaly detection |

## 4. 数据来源与构造

### 4.1 正样本

主要使用 PaSa 查询数据：

```text
数据集/pasa/data/AutoScholarQuery/train.jsonl
数据集/pasa/data/AutoScholarQuery/dev.jsonl
数据集/pasa/data/AutoScholarQuery/test.jsonl
数据集/pasa/data/RealScholarQuery/test.jsonl
ScholarSearchAgent/data_ingestion_indexing/data_processed/queries.jsonl
```

这些样本字段：

```json
{
  "question": "Could you provide me some works employs image patches and superpixels in region-based methods for semantic segmentation?",
  "answer": ["..."],
  "answer_arxiv_id": ["..."],
  "source_meta": {"published_time": "20230917"},
  "qid": "AutoScholarQuery_train_1"
}
```

其中 `question` 可作为输入，`answer_count`、`published_time` 可作为辅助特征，但 `answer_arxiv_id` 只能用于检索评估，不能泄漏进在线查询理解输入。

### 4.2 负样本

PaSa 主要是正向学术检索 query，因此必须补负样本。否则模型会把几乎所有输入都判断为论文检索。

负样本来源建议：

| 来源 | 构造方式 |
| --- | --- |
| AstaBench `library_diagnostic` | 编程/库使用问题，标记为 `non_paper_search` |
| AstaBench `discoverybench` | 数据分析/假设生成问题，部分标为 `non_paper_search` 或 `application_search` |
| AstaBench `sqa` | 科学问答问题，标为 `non_paper_search` 或 `paper_evidence_qa` |
| 人工模板 | 闲聊、系统操作、代码问题、普通百科问答 |
| 当前系统日志 | 用户真实输入中的非检索请求 |

建议正负比例控制在：

```text
paper_search : non_paper_search = 3 : 1 或 2 : 1
```

过多负样本会使模型过于保守，复杂学术查询可能被误拒。

### 4.3 弱标注生成

可以先用规则生成初版标签，再人工抽检。

示例规则：

| 规则 | 标签 |
| --- | --- |
| 包含 `papers`, `works`, `studies`, `research`, `proposed` | `paper_search` |
| 包含 `survey`, `overview`, `related to` | `survey_search` |
| 包含 `proposed`, `method`, `algorithm`, `model` | `method_search` |
| 包含 `dataset`, `benchmark`, `corpus` | `dataset_search` |
| 包含 `metric`, `accuracy`, `recall`, `F1` | `metric_search` |
| 包含 `who proposed`, `inspired by`, `origin` | `citation_trace` |
| 包含 `mechanism`, `why`, `understanding` | `mechanism_search` |

生成后抽样 300-500 条人工校验，优先修正：

- `survey_search` 与 `method_search` 混淆。
- `mechanism_search` 与普通 `method_search` 混淆。
- 问答型 query 是否需要论文检索支撑。

## 5. 训练样本格式

### 5.1 分类器格式

适合 Hugging Face `AutoModelForSequenceClassification`。

```json
{
  "text": "What works are related to the field of image retrieval?",
  "label": "survey_search"
}
```

若同时做是否检索和意图类型，可使用两个 head，或拆成两个模型：

```text
query_gate: paper_search / non_paper_search
intent_classifier: survey_search / method_search / ...
```

### 5.2 生成式 JSON 格式

适合 Qwen Instruct + LoRA SFT。

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Analyze the academic search intent of this query and output JSON only:\nCould you provide me some studies that proposed hierarchical neural models to capture spatio-temporal features in sign videos?"
    },
    {
      "role": "assistant",
      "content": "{\"is_paper_search\": true, \"intent_type\": \"method_search\", \"research_fields\": [\"sign language recognition\", \"video understanding\"], \"must_have_constraints\": [\"hierarchical neural models\", \"spatio-temporal features\", \"sign videos\"], \"soft_constraints\": [\"sign language translation\"], \"excluded_meanings\": [], \"time_range\": null, \"venues\": [], \"sub_queries\": [\"hierarchical neural models spatio temporal sign videos\", \"sign language translation hierarchical spatio temporal graph neural network\"]}"
    }
  ]
}
```

## 6. 模型与框架选型

### 6.1 推荐路线 A：轻量分类器 + 规则抽取

适合快速上线。

| 项 | 建议 |
| --- | --- |
| 框架 | Hugging Face Transformers |
| 模型 | `microsoft/deberta-v3-base`、`BAAI/bge-base-en-v1.5`、`intfloat/e5-base-v2` |
| 训练任务 | query gate + intent classifier |
| 优点 | 快、稳定、资源需求低 |
| 缺点 | 结构化约束抽取仍依赖规则 |

当前数据主要是英文 query，因此优先使用英文/多语种 encoder。若后续用户中文输入增多，可引入 `BAAI/bge-m3` 或中文 RoBERTa 类模型。

### 6.2 推荐路线 B：Qwen 小模型生成结构化 JSON

适合需要统一输出复杂结构。

| 项 | 建议 |
| --- | --- |
| 框架 | LLaMA-Factory 或 TRL SFTTrainer |
| 模型 | `Qwen2.5-1.5B-Instruct`、`Qwen2.5-3B-Instruct`、`Qwen3-1.7B`、`Qwen3-4B` |
| 微调方式 | LoRA / QLoRA |
| 优点 | 能直接输出约束、子查询、排除条件 |
| 缺点 | 需要 JSON 合法性校验和兜底 |

### 6.3 本项目推荐

第一阶段推荐：

```text
query_gate + intent_classifier: encoder 分类器
constraint/sub_query: 规则版 QueryParser + 少量 LLM fallback
```

第二阶段再训练：

```text
Qwen 小模型 JSON SFT
```

原因是查询理解错误会影响全链路，先用可控分类器把入口稳定住，再引入生成式结构化输出更安全。

## 7. 训练逻辑

### 7.1 数据切分

建议固定切分：

```text
train: AutoScholarQuery/train + 构造负样本
dev: AutoScholarQuery/dev + 构造负样本
test: AutoScholarQuery/test + RealScholarQuery/test + 人工负样本
```

`RealScholarQuery/test` 样本少但真实，应只用于最终压力测试，不参与训练。

### 7.2 训练流程

```text
1. 收集 query
2. 构造正负样本
3. 规则弱标注 intent_type
4. 人工抽检和修正小规模 dev/test
5. 训练 query_gate
6. 训练 intent_classifier 或 JSON SFT
7. 在检索流水线中替换 QueryParser
8. 用 Recall@K / MRR 验证最终检索收益
```

### 7.3 关键超参数

Encoder 分类器：

```text
max_length: 256
learning_rate: 2e-5 或 3e-5
epochs: 3-5
batch_size: 16-64
metric: macro F1
early_stopping: dev macro F1
```

Qwen LoRA：

```text
max_length: 1024
learning_rate: 1e-4 到 2e-4
epochs: 2-3
lora_rank: 8 或 16
lora_alpha: 16 或 32
target_modules: q_proj,k_proj,v_proj,o_proj
precision: fp16
```

当前 `cu02` 是 Tesla P100 16GB x2。P100 不支持 bf16，训练配置应使用 fp16；7B 全参训练不现实，建议小模型 LoRA。

## 8. 评估指标

### 8.1 模型级指标

| 任务 | 指标 |
| --- | --- |
| 是否检索 | Accuracy、Precision、Recall、F1 |
| 意图分类 | Macro F1、per-class F1 |
| 约束抽取 | Exact Match、token-level F1、人工抽检 |
| JSON 输出 | JSON valid rate、schema pass rate |

### 8.2 系统级指标

最终必须看检索收益：

```text
Recall@20 / Recall@50 / Recall@100
MRR
Precision@K
query latency
fallback rate
```

不要只看分类准确率。查询理解模型的价值在于提高下游召回和排序。

## 9. 接入方式

建议保留当前规则解析器作为兜底：

```text
用户 query
  -> query_gate
    -> non_paper_search: 返回澄清/普通回答/不进入论文检索
    -> paper_search:
      -> intent_classifier 或 JSON SFT
      -> schema validator
      -> QueryIntent
      -> search_strategy_planning
```

工程上可新增：

```text
query_understanding_decomposition/model_parser.py
query_understanding_decomposition/schema.py
query_understanding_decomposition/train/
```

## 10. 风险与规避

| 风险 | 规避 |
| --- | --- |
| 缺少负样本 | 从 AstaBench、人工模板、真实日志补充 |
| 弱标注噪声 | 先抽检 dev/test，训练集允许一定噪声 |
| JSON 生成不合法 | 加 schema validator，失败时回退规则 parser |
| 意图标签过细 | 第一版控制在 8 类以内 |
| 英文训练、中文输入 | 中文 query 先翻译/改写，或补中文增强样本 |
| 离线指标好但召回无提升 | 用检索 Recall@K 作为最终准入标准 |

## 11. 推荐里程碑

```text
M1: 构造 4-5 万 query gate / intent 样本
M2: 训练 encoder 分类器并接入 QueryParser 前置
M3: 构造 300-500 条人工校验集
M4: 评估分类指标和端到端 Recall@50
M5: 再考虑 Qwen JSON SFT 替代规则抽取
```

## 12. 参考框架

- Hugging Face Transformers text classification: https://huggingface.co/docs/transformers/en/tasks/sequence_classification
- Hugging Face Transformers fine-tuning: https://huggingface.co/docs/transformers/en/training
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Qwen LLaMA-Factory SFT 文档: https://qwen.readthedocs.io/en/v2.5/training/SFT/llama_factory.html
