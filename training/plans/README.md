# ScholarSearchAgent 模型训练路线文档

本目录整理三条模型训练路线，用于把当前规则版论文检索 demo 升级成可训练、可评估、可迭代的学术搜索 Agent。

## 推荐阅读顺序

| 顺序 | 文档 | 模型 | 优先级 | 主要收益 |
| ---: | --- | --- | --- | --- |
| 1 | `02_selector_reranker_model.md` | Selector / Reranker 相关性识别 | P0 | 最直接提升排序质量和 Recall/MRR |
| 2 | `01_query_intent_model.md` | 查询意图识别与查询理解 | P0 | 稳定入口、查询分解、减少误检索 |
| 3 | `03_crawler_strategy_model.md` | Crawler 检索策略与章节展开 | P1 | 通过引用/章节扩展提高复杂查询召回 |

## 三个模型的分工

```text
用户 query
  -> Query Intent Model
       判断是否是论文检索问题
       抽取意图、约束、子查询
  -> BM25 / Dense / KG / Citation Retrieval
       多源召回候选论文
  -> Selector / Reranker Model
       判断 query-paper 是否相关
       对候选论文重排
  -> Crawler Strategy Model
       对 top seed papers 选择值得展开的章节
       扩展引用网络候选
  -> Selector / Reranker Model
       对扩展候选再次重排
```

## 当前最建议先做的路线

先训练 `Selector / Reranker`。

原因：

- 已有 `sft_selector/train.jsonl`，数据最贴近监督训练。
- 当前 `relevance_ranking/ranking.py` 还是手工打分，提升空间明显。
- reranker 可以直接接到现有检索结果后面，不需要大改系统。
- 对比赛指标 `Recall@K`、`MRR`、`Precision@K` 的影响最直接。

## 数据边界

可用于训练：

```text
数据集/pasa/data/AutoScholarQuery/train.jsonl
数据集/pasa/data/AutoScholarQuery/dev.jsonl
数据集/pasa/data/sft_selector/train.jsonl
数据集/pasa/data/sft_crawler/train.jsonl
```

建议只用于最终评估：

```text
数据集/pasa/data/AutoScholarQuery/test.jsonl
数据集/pasa/data/RealScholarQuery/test.jsonl
数据集/pasa/data/sft_selector/test.jsonl
```

注意：

- `answer_arxiv_id`、`gold_labels`、`eval_sets.gold_paper_ids` 只能作为训练/评估标签。
- 它们不能进入检索索引，也不能在测试集在线检索阶段泄漏给模型。

## 硬件建议

当前 `cu02` 记录为：

```text
GPU: Tesla P100-PCIE-16GB x2
MEM: 125GB
CUDA: 12.4 driver
```

建议：

- Cross-Encoder / BGE reranker 可优先尝试。
- Qwen 训练优先使用 1.5B/3B LoRA。
- 7B 只建议 LoRA/QLoRA，小 batch 慢速训练。
- P100 不支持 bf16，训练配置使用 fp16。
- flash-attention 先不要作为硬依赖。

## 文档清单

- `01_query_intent_model.md`: 查询意图识别、检索问题识别、结构化查询理解。
- `02_selector_reranker_model.md`: query-paper 相关性判断、reranker 训练、hard negative 构造。
- `03_crawler_strategy_model.md`: 检索策略、章节展开、引用网络扩展策略模型。
