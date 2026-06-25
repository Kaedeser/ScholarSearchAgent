# Crawler 检索策略与章节展开模型训练路线

## 1. 模型定位

Crawler 模型负责在检索过程中决定下一步动作：

```text
当前用户 query + 当前论文 title/abstract + 可展开章节列表
  -> 应该展开哪些章节 / 是否继续搜索 / 是否停止
```

它不是普通意图分类模型，也不是最终相关性排序模型，而是检索 Agent 的策略模型。它的作用是提高召回覆盖率，尤其是在复杂学术查询中，通过引用网络、章节引用、相关工作部分继续发现目标论文。

在当前项目中，它应增强：

```text
citation_network_expansion/
coverage_iteration/
search_strategy_planning/
```

## 2. 训练目标

### 2.1 主目标

训练一个策略识别模型，根据 query 和候选论文结构，输出哪些章节值得展开。

PaSa 数据中的典型输出：

```text
[Expand]1 Introduction[StopExpand]
```

或：

```text
[Expand]3 Closed-Form Policy Improvement 3.4 Theoretical guarantees for CFPI operators[Expand]4 Related Work[StopExpand]
```

### 2.2 目标能力

| 能力 | 说明 |
| --- | --- |
| 章节选择 | 从章节列表中选择最可能包含相关引用的章节 |
| 停止判断 | 当前论文不值得继续展开时停止 |
| 引用扩展策略 | 优先展开 introduction、related work、method、experiment 等不同章节 |
| 查询约束感知 | 根据 query 关注来源、方法、机制、实验还是应用 |
| 成本控制 | 避免无意义展开过多章节 |

## 3. 数据来源

核心数据：

```text
数据集/pasa/data/sft_crawler/train.jsonl
```

样本格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are conducting research on `Who worked on maximizing Chow's excess risk?`. You need to predict which sections to look at for getting more relevant papers. Title: Selective Nonparametric Regression via Testing\nAbstract: ...\nSections: [\"1 Introduction\", \"2 Regression with Abstention\", ...]"
    },
    {
      "role": "assistant",
      "content": "[Expand]1 Introduction[StopExpand]"
    }
  ]
}
```

规模：

```text
sft_crawler/train.jsonl: 约 12989 条
```

辅助数据：

```text
数据集/pasa/data/AutoScholarQuery/*.jsonl
数据集/pasa/data/paper_database/cs_paper_2nd.zip
ScholarSearchAgent/data_ingestion_indexing/data_processed/paper_chunks.jsonl
```

## 4. 输出动作设计

### 4.1 PaSa 原生格式

保持兼容：

```text
[Expand]{section_title}[Expand]{section_title}[StopExpand]
```

优点：

- 可直接复用 `sft_crawler`。
- 与 PaSa 论文和代码一致。
- 简单、短输出、易训练。

缺点：

- 不包含置信度。
- 不包含动作原因。
- 不利于后续结构化策略分析。

### 4.2 推荐结构化格式

为了工程可控，建议训练时仍保留 PaSa 格式，推理后解析成结构化对象：

```json
{
  "action": "expand",
  "sections": [
    "1 Introduction",
    "4 Related Work"
  ],
  "stop_after_expand": true
}
```

第二阶段可直接训练 JSON：

```json
{
  "selected_sections": ["1 Introduction", "4 Related Work"],
  "should_expand": true,
  "reason": "The query asks for origin and related work; introduction and related work are most likely to contain citations to prior methods."
}
```

## 5. 模型与框架选型

### 5.1 首选：Qwen Instruct + LoRA SFT

| 项 | 建议 |
| --- | --- |
| 框架 | LLaMA-Factory |
| 基座 | `Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`、`Qwen3-4B` |
| 训练方式 | LoRA / QLoRA |
| 输出 | `[Expand]... [StopExpand]` |
| 优点 | 与 PaSa 原训练方式接近，能处理复杂 prompt |
| 缺点 | 推理成本高于分类器 |

当前硬件 `Tesla P100 16GB x2`：

- 推荐先用 `Qwen2.5-3B-Instruct` LoRA。
- 如果必须用 7B，使用 QLoRA/LoRA，小 batch，fp16。
- 不要使用 bf16。
- flash-attention 需要确认环境支持；P100 上优先用普通 attention 保守起步。

### 5.2 备选：多标签章节分类器

如果只做章节选择，也可以把每个 section 变成 pair 分类：

```text
input: query + title + abstract + section_title
label: expand / not_expand
```

| 项 | 建议 |
| --- | --- |
| 框架 | Hugging Face Transformers |
| 模型 | DeBERTa / BGE encoder |
| 优点 | 快、便宜、易部署 |
| 缺点 | 不能自然输出多动作序列，难表达停止策略 |

这个方案适合作为轻量 fallback，但不如 Qwen SFT 贴合 PaSa 数据。

### 5.3 本项目推荐

第一版：

```text
Qwen2.5-3B-Instruct + LoRA SFT + PaSa 原生 Expand 格式
```

上线策略：

```text
只对 topN seed papers 调用 crawler
N 建议 10-30
每篇最多展开 1-3 个章节
```

## 6. 训练逻辑

### 6.1 数据清洗

从 `sft_crawler/train.jsonl` 抽取：

```text
query
title
abstract
sections
target_output
```

清洗规则：

- 丢弃缺少 `Sections:` 的样本。
- 丢弃 assistant 输出无法解析 `[Expand]` / `[StopExpand]` 的样本。
- 保留 section title 原文，避免训练和推理时标题不一致。
- 统计每条样本展开 section 数量，限制异常样本。

### 6.2 样本增强

建议不要大规模改写原 prompt，先保持 PaSa prompt 分布一致。

可做轻量增强：

| 增强 | 说明 |
| --- | --- |
| section 顺序保持 | 不要打乱，章节顺序有语义 |
| query 同义改写 | 小比例，防止过拟合模板 |
| stop-only 样本 | 补充不应展开的样本 |
| hard cases | query 关注理论、实验、相关工作时分别强化 |

stop-only 样本可以从 selector 判定低相关论文中构造：

```text
query + 不相关 paper + sections -> [StopExpand]
```

### 6.3 Prompt 模板

训练和推理保持一致：

```text
You are conducting research on `{query}`.
You need to predict which sections to look at for getting more relevant papers.

Title: {title}
Abstract: {abstract}
Sections: {sections_json}

Output only actions in this format:
[Expand]section title[Expand]section title[StopExpand]
```

如果沿用 PaSa 原数据，可以不改 prompt；如果要部署更稳，建议逐步统一模板。

## 7. 训练超参数建议

### 7.1 Qwen 3B LoRA

```text
model_name_or_path: Qwen2.5-3B-Instruct
stage: sft
finetuning_type: lora
template: qwen
cutoff_len: 1024 或 1536
learning_rate: 1e-4
num_train_epochs: 2
per_device_train_batch_size: 1-2
gradient_accumulation_steps: 8-16
lora_rank: 8 或 16
lora_alpha: 16 或 32
fp16: true
bf16: false
```

### 7.2 Qwen 7B LoRA / QLoRA

```text
model_name_or_path: Qwen2.5-7B-Instruct
finetuning_type: lora
quantization_bit: 4
cutoff_len: 1024
learning_rate: 5e-5 到 1e-4
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
fp16: true
bf16: false
```

7B 方案更接近 PaSa 原文路线，但在当前 P100 环境上成本更高。建议先用 3B 跑通训练、评估和接入。

## 8. 评估指标

### 8.1 动作级指标

| 指标 | 说明 |
| --- | --- |
| exact match | 输出动作串完全一致 |
| section precision | 预测展开章节中有多少正确 |
| section recall | 标准展开章节中有多少被找回 |
| section F1 | 综合章节选择能力 |
| stop accuracy | 是否正确停止 |
| parse success rate | 输出能否被解析 |

不要只看 exact match，因为多个章节可能都合理。

### 8.2 检索级指标

Crawler 的最终价值是提升召回：

```text
crawler_seed_recall@K
expanded_candidate_hit_rate
final Recall@20 / Recall@50 / Recall@100
```

建议做消融：

```text
baseline: 不做 citation expansion
rule: 当前 citation_network_expansion 规则
model: crawler SFT 选择章节
oracle: 使用 gold 可达路径上限估计
```

### 8.3 成本指标

必须记录：

```text
每个 query 展开论文数
每个 query 展开章节数
新增候选数
新增 gold 命中数
平均延迟
```

如果召回提升来自无限展开，工程上不可接受。Crawler 模型必须和预算控制一起评估。

## 9. 接入方式

建议新增：

```text
citation_network_expansion/crawler_model.py
citation_network_expansion/action_parser.py
citation_network_expansion/train_crawler/
```

在线流程：

```text
初始召回 topK
  -> selector/ranker 得到 seed papers
  -> crawler 对 topN seed papers 选择 sections
  -> 从 KG / paper zip / citation edges 展开候选
  -> candidate_normalization 去重
  -> selector rerank
```

预算建议：

```text
topN_seed_papers: 10-30
max_sections_per_paper: 2
max_expanded_candidates_per_query: 100-300
```

## 10. 与知识图谱的关系

Crawler 不直接替代 KG，它决定从 KG 中走哪条边。

```text
Crawler 输出 section
  -> Section 节点
  -> MENTIONS_REFERENCE / CITES 边
  -> expanded papers
```

如果 KG 尚未完整构建，可从 `cs_paper_2nd.zip` 按需读取章节引用；如果 KG 已构建，则直接查：

```text
Paper -> HAS_SECTION -> Section -> MENTIONS_REFERENCE -> RESOLVES_TO -> Paper
```

## 11. 风险与规避

| 风险 | 规避 |
| --- | --- |
| 模型倾向总是展开 Introduction / Related Work | 加强不同 query 类型样本，统计 section 分布 |
| 输出章节名不在原列表中 | 推理后做 fuzzy match，只允许选择输入 section |
| 展开过多导致延迟高 | 加 max section / max candidate 预算 |
| Crawler 召回噪声放大 | 扩展候选必须经过 selector rerank |
| 训练集只有 expand 样本，stop 能力弱 | 从低相关论文构造 stop-only 样本 |
| 复杂 query 下策略不稳定 | 使用多次采样 ensemble，但限制成本 |

## 12. 推荐里程碑

```text
M1: 解析 sft_crawler 为标准 instruction 数据
M2: Qwen2.5-3B LoRA 训练第一版
M3: 实现 action parser 和 section fuzzy matcher
M4: 接入 citation_network_expansion topN seed
M5: 用 AutoScholarQuery dev 评估扩展召回收益
M6: 再尝试 7B 或 PPO/偏好优化
```

## 13. 参考框架

- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Qwen LLaMA-Factory SFT 文档: https://qwen.readthedocs.io/en/v2.5/training/SFT/llama_factory.html
- Hugging Face TRL SFTTrainer: https://huggingface.co/docs/trl/en/sft_trainer
- Hugging Face Transformers fine-tuning: https://huggingface.co/docs/transformers/en/training
- PaSa README: `数据集/pasa/README.md`
