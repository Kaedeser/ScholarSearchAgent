# ScholarSearchAgent 最新版项目功能架构与模块实现逻辑

生成日期：2026-07-05
文档状态：最终架构文档
依据材料：`reports` 下第一阶段到第七阶段优化报告，以及当前 `apps`、`packages`、`scripts`、`tests` 代码结构。

## 1. 最新优化基线

当前项目已经从早期的规则检索流水线演进为多源召回、dense paper 语义召回、GPUStack query rewrite、规则预筛、selector rerank、覆盖诊断和远端分片评测组成的学术论文搜索 Agent。

第七阶段固化的主检索链路是：

```text
用户查询
  -> QueryParser 规则解析
  -> gated Query Rewrite
  -> Query Type Routing
  -> SearchPlanner 生成多源召回动作
  -> ES / Qdrant / Neo4j / MySQL 多源召回与回表
  -> CandidateNormalizer 合并候选
  -> CandidateRanker 初排
  -> CandidatePreselector: 500 -> 120
  -> Selector Reranker 精排
  -> diversity / source rank backfill
  -> CoverageAnalyzer 诊断和二轮补检
  -> Top 50 结果与诊断信息
```

第七阶段全量评测固定配置：

```env
SELECTOR_RERANKER_ENABLED=true
SELECTOR_RERANKER_POOL_LIMIT=500
SELECTOR_RERANKER_CANDIDATE_LIMIT=120
SELECTOR_RERANKER_PROTECTED_HEAD=0
QUERY_REWRITE_MODEL=Qwen3-30B-A3B-Instruct-2507
QUERY_INTENT_ENABLED=false
CRAWLER_STRATEGY_ENABLED=false
QDRANT_DENSE_PAPER_ENABLED=true
QDRANT_SPARSE_PAPER_ENABLED=true
```

最终全量评测结果，来自 `reports/preselector_full_cl120_ph0_20260705_180154/merged/db_eval_report.md`：

| 指标 | 数值 |
| --- | ---: |
| Requested queries | 1050 |
| Evaluated queries | 1050 |
| Failed queries | 0 |
| Gold labels | 3194 |
| Avg latency/query | 39.216807s |
| Precision@50 | 0.014990 |
| Recall@50 | 0.315106 |
| F1@50 | 0.026365 |
| MRR@50 | 0.108689 |
| HitRate@50 | 0.468571 |
| Hits@50 | 787 |

分数据集结果：

| Dataset | Queries | Recall@50 | F1@50 | MRR@50 | HitRate@50 | Hits@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AutoScholarQuery/test | 1000 | 0.311902 | 0.021737 | 0.092401 | 0.442000 | 575 |
| RealScholarQuery/test | 50 | 0.379195 | 0.118920 | 0.434446 | 1.000000 | 212 |

与第七阶段前的 `500 -> 50 -> rerank` 相比：

| 指标 @50 | 500 -> 50 -> rerank | 500 -> 120 -> rerank | 变化 |
| --- | ---: | ---: | ---: |
| Precision@50 | 0.013771 | 0.014990 | +0.001219 |
| Recall@50 | 0.290871 | 0.315106 | +0.024235 |
| F1@50 | 0.024205 | 0.026365 | +0.002160 |
| MRR@50 | 0.131683 | 0.108689 | -0.022994 |
| HitRate@50 | 0.437143 | 0.468571 | +0.031428 |
| Hits@50 | 723 | 787 | +64 |

结论：当前默认目标更偏 Recall/F1，故固定 `500 -> 120 -> rerank`。如果后续更重视首位结果，可以重新评估 `protected_head=8` 或加入原始初排分数保护。

## 2. 七轮优化沉淀

| 轮次 | 核心优化 | 关键沉淀 |
| --- | --- | --- |
| 第一阶段 | 查询解析、SearchPlanner、CoverageAnalyzer、Ranker 基础增强 | 增加专业短语、约束覆盖、二轮补检和回归测试，确认主要瓶颈在候选池与覆盖而非单纯 reranker |
| 第二阶段 | ES 查询结构优化 | 保留 broad BM25 主召回，把 phrase/title/clean query 放入 rescore window，避免主查询过窄损伤多答案召回 |
| 第三阶段 | Qdrant 全量 sparse 接入 | 完成 sparse chunk 查询向量清洗、chunk 正文回填、Qdrant 原始分数 log 压缩、失败样本 alias 闭环 |
| 第四阶段 | Neo4j 与结构化增强路线 | 引入 concept/alias/graph 辅助召回，明确 dense semantic retrieval、query rewrite、query type routing 是下一阶段重点 |
| 第五阶段 | WeightedQuery、source-aware 融合和分片评测工程 | 增加 query type 诊断、source gold hits、pool recall@100/200/500、4 shard 远端评测与合并 |
| 第六阶段 | GPUStack dense paper semantic retrieval | 上线 `saiti3_papers_dense_v1`，dense used rate 从 0 到 1，pool recall@500 从 0.415653 提升到 0.508708 |
| 第七阶段 | 500 -> 120 -> selector rerank 预筛 | 新增规则预筛模块，保留更多 gold 候选进入 selector，Recall@50 提升到 0.315106 |

## 3. 当前目录分层

```text
ScholarSearchAgent/
  apps/
    backend/                  # HTTP/CLI 后端应用入口
    frontend/                 # React/Vite 前端应用

  packages/
    scholar_core/             # 领域核心：解析、规划、召回协议、排序、覆盖、流水线
    scholar_infra/            # 基础设施：配置、DB/ES/Qdrant/Neo4j、模型服务、embedding
    scholar_ingest/           # 数据导入、索引构建、Qdrant sparse/dense 点生成
    scholar_eval/             # 离线评测能力

  configs/                    # 推荐配置模板
  config/                     # 兼容旧配置目录
  scripts/                    # 启动、部署、远端评测、报告合并脚本
  tests/                      # 回归测试
  training/                   # 三个模型训练工程与训练计划
  reports/                    # 七轮优化报告和评测产物
  docs/architecture/          # 最终架构文档，仅保留本文件
  docs/operations/            # 启动、运维和模型服务说明
```

当前新开发以 `apps` 和 `packages` 为准。旧的根级业务目录如 `scholar_app`、`scholar_common`、`scholar_frontend` 和若干同名业务模块仍作为历史资产或兼容入口存在，不再作为主架构入口。

## 4. 功能架构

### 4.1 应用层

`apps/backend/scholar_api` 是后端入口，当前仍使用轻量 Python HTTP server 形态，职责是接收 HTTP/CLI 请求、装配 `SearchPipeline`、调用领域核心并返回 JSON。API 层不直接写检索算法，也不直接封装数据库细节。

`apps/frontend` 是 React/Vite 前端，职责是搜索界面、结果列表、详情面板、覆盖/trace 信息展示和 API 调用封装。前端只依赖 `/health` 与搜索 API 响应，不感知 MySQL、ES、Qdrant、Neo4j 或模型服务地址。

### 4.2 领域核心层

`packages/scholar_core` 承载在线检索的业务逻辑。它不直接读取环境变量，不直接创建数据库连接，外部能力通过 `retrieval.ports` 和 `model_services.ports` 注入。

核心模块：

| 模块 | 主要文件 | 职责 |
| --- | --- | --- |
| 领域模型 | `models.py` | 定义 `QueryIntent`、`SearchAction`、`SearchPlan`、`Candidate`、`CoverageReport`、`SearchResponse` |
| 查询理解 | `query_understanding/parser.py` | 抽取研究领域、硬约束、软约束、时间范围、venue、sub queries |
| 检索前加权 | `retrieval/weighted_query.py` | 生成 clean query、weighted terms、phrase boost、dense query text、sparse feature map |
| 策略规划 | `planning/planner.py` | 根据 query profile 生成 ES/Qdrant/Neo4j 多路召回动作和预算 |
| 流水线编排 | `pipeline.py` | 串联解析、rewrite、召回、归一、排序、预筛、rerank、覆盖诊断、二轮补检 |
| 候选归一 | `normalization/normalizer.py` | 按 canonical id 合并多源候选、来源和分数 |
| 规则排序 | `ranking/ranker.py` | 计算约束覆盖、source rank、RRF、dense/sparse synergy、alias bonus、missing penalty |
| 预筛 | `ranking/preselector.py` | 从最多 500 个初排候选中按多路 lane 选 120 个进入 selector |
| 覆盖分析 | `coverage/analyzer.py` | 判断 matched/missing constraints，生成二轮补检 query |
| 引用规划 | `citation/planner.py` | 选择可用于后续 citation/section 展开的种子论文 |
| 模型端口 | `model_services/ports.py` | 定义 query intent、query rewrite、selector reranker、crawler strategy 的协议 |

### 4.3 基础设施层

`packages/scholar_infra` 承载外部系统适配：

| 模块 | 主要文件 | 职责 |
| --- | --- | --- |
| 配置 | `config.py` | 读取 model services、query rewrite、selector reranker、Neo4j 等运行配置 |
| 多源召回适配 | `retrieval_backends/retrieval.py` | `LocalCorpus` 与 `DatabaseCorpus`，把检索动作转换成候选 |
| MySQL | `persistence/mysql.py` | paper/chunk/gold 回表和评测数据读取 |
| Elasticsearch | `persistence/elasticsearch.py` | paper/chunk BM25、字段权重、phrase/rescore 查询 |
| Qdrant | `persistence/qdrant.py` | sparse chunk、sparse paper、dense paper 检索 |
| Neo4j | `persistence/neo4j.py` | concept、alias、graph neighbor 检索 |
| Embedding | `embeddings.py` | sentence-transformers 与 OpenAI-compatible/GPUStack embedding 客户端 |
| 模型服务 | `model_services/client.py` | query rewrite、query intent、selector reranker、crawler strategy HTTP 客户端 |
| IO | `io/jsonl.py` | 本地 JSONL corpus 加载 |

### 4.4 数据导入和索引层

`packages/scholar_ingest` 负责离线导入和索引构建，在线后端不直接依赖训练逻辑。

关键职责：

- 从 PaSa/数据库源构建 paper、chunk、eval set。
- 写入 MySQL、Elasticsearch、Qdrant。
- 构建 `saiti3_paper_chunks_v1` sparse chunk collection。
- 构建 `saiti3_papers_dense_v1` dense paper collection。
- 生成 sparse paper / dense paper Qdrant point。
- 支持 dense paper 分片导入，避免单进程长时间构建。

第六阶段 dense paper 正式 collection：

| 项目 | 值 |
| --- | --- |
| Qdrant collection | `saiti3_papers_dense_v1` |
| Embedding backend | GPUStack / OpenAI-compatible |
| Embedding model | `qwen3-embedding-4b` |
| Vector size | 2560 |
| Points | 569265 |
| Text format | `title + metadata + abstract[:2200]` |

### 4.5 评测与运维层

`scripts/evaluate_db_agent.py` 是当前最重要的离线评测入口，支持：

- AutoScholarQuery / RealScholarQuery 全量或抽样评测。
- `--max-total-queries` 与 `--sample-profile head|proportional|balanced`。
- `--shard-count` / `--shard-index` 远端分片评测。
- Precision / Recall / F1 / MRR / HitRate。
- pool recall@100/200/500。
- source gold hits、dense/alias contribution。
- query type / feature usage。
- selector preselection 诊断。
- primary reason 与 priority stage 分桶。

`scripts/merge_db_eval_reports.py` 合并多分片 JSON/Markdown 报告。
`scripts/launch_stage6_full_4shard_remote.py` 封装远端 start/status/merge/download 流程。

## 5. 核心检索链路实现逻辑

### 5.1 QueryParser

`QueryParser` 是规则解析器，输入自然语言 query，输出 `QueryIntent`。

主要规则表：

- `FIELD_HINTS`：把 token 映射到研究领域，例如 `llm -> large language models`。
- `SYNONYMS`：维护高价值同义词和失败样本沉淀的桥接词，例如 `semantic tokens -> generative spoken language modeling`。
- `BRIDGE_ALIASES`：把 query 词面桥接到更常见的论文标题/方法表达。
- `KEY_PHRASE_PATTERNS`：用正则识别 HuBERT codes、mask classification、RLHF、NeRF、object navigation 等关键短语。

输出字段：

- `main_intent`：搜索意图摘要。
- `research_field`：研究领域。
- `must_have_constraints`：必须覆盖的硬约束。
- `soft_constraints`：同义词、别名、任务线索等软约束。
- `excluded_meanings`：排除语义。
- `time_range` / `venues`：年份和会议限制。
- `sub_queries`：用于多路召回的查询改写。
- `query_tokens`：供排序和 snippet 使用。

### 5.2 WeightedQuery

`retrieval/weighted_query.py` 在检索前统一处理词项权重，服务于 ES、Qdrant sparse、BM25 和 dense query 生成。

实现要点：

- 过滤低价值词和停用词。
- 提升 acronym、高价值学术词、带连字符术语、领域后缀词。
- 提取 2-4 gram 学术短语。
- 注入 `_ALIAS_GROUPS` 中的同义表达。
- 生成 `clean_query`、`expanded_query`、`weighted_token_map`、`dense_retrieval_query_text`、`weighted_sparse_query_feature_map`、`sparse_document_feature_map`。

Dense paper retrieval 使用 `dense_retrieval_query_text()`，避免直接把模板化自然语言送入 embedding。

### 5.3 SearchPlanner

`SearchPlanner` 先通过 `query_profile_kind()` 对查询分型，再根据 `profile_retrieval_budget()` 生成召回动作。

当前 profile：

```text
auto_locator
real_multi_answer
survey_or_list
dataset_or_benchmark
foundational_or_origin
method_or_dataset
comparison_or_claim
```

生成的主要 `SearchAction`：

| Action | 后端含义 |
| --- | --- |
| `local_title_bm25` | ES paper title/abstract BM25 |
| `local_chunk_bm25` | ES chunk text/section BM25 |
| `local_tfidf` | Qdrant sparse chunk lexical retrieval |
| `qdrant_dense_paper` | Qdrant dense paper semantic retrieval |
| `qdrant_sparse_paper` | Qdrant sparse paper title/abstract retrieval |
| `neo4j_concept` | Neo4j concept graph retrieval |
| `neo4j_alias` | Neo4j alias graph retrieval |

不同 profile 使用不同预算。例如 `auto_locator` 提升 title、dense paper、sparse paper 和 alias；`real_multi_answer` 提升 chunk、sparse、dense 和 diversity；`dataset_or_benchmark` 提升 dataset/benchmark 相关 dense、sparse 和 alias。

### 5.4 DatabaseCorpus

`DatabaseCorpus` 是真实数据库检索适配器，把 `SearchAction` 转成候选列表。

主要数据源：

| 数据源 | 当前角色 |
| --- | --- |
| MySQL | 事实库、回表库、评测 gold 来源 |
| Elasticsearch papers | paper-level title/abstract BM25 主召回 |
| Elasticsearch chunks | chunk-level 正文证据召回 |
| Qdrant sparse chunks | lexical sparse chunk 辅助召回 |
| Qdrant sparse papers | title/abstract paper-level sparse 补充 |
| Qdrant dense papers | paper-level semantic recall，当前第二强 gold source |
| Neo4j | concept/alias/graph neighbor 辅助召回 |

实现细节：

- ES paper hit 直接转成 `Candidate`。
- ES chunk hit 先按 `paper_id` 回表 paper，再生成 chunk snippet。
- Qdrant sparse chunk hit 根据 `chunk_id` 用 ES `_mget` 回填 chunk text，并对 raw score 做 `log1p` 压缩。
- Qdrant dense paper hit 使用 dense query embedding，命中后回表 paper，标记 `dense_used=True`。
- Qdrant sparse paper hit 以 title/abstract 为 paper-level sparse 召回，分数同样 log 压缩。
- Neo4j concept/alias hit 回表 MySQL，并写入 graph support、relations、aliases 等 metadata。
- `_paper_cache`、`_chunk_cache`、`_dense_query_cache` 降低重复回表和重复 embedding 调用。

### 5.5 SearchPipeline

`SearchPipeline.search()` 是在线检索主编排。

执行顺序：

1. `QueryParser.parse()` 生成初始 `QueryIntent`。
2. 可选 `QueryIntentServiceClient` 修正或拦截非论文搜索。
3. `_should_use_query_rewrite()` 对高风险 query 启用 GPUStack query rewrite。
4. `SearchPlanner.plan()` 生成第一轮多源召回动作。
5. `_run_actions()` 执行 ES/Qdrant/Neo4j 检索并标注 source rank 与 RRF。
6. `_rank()` 合并候选、初排、预筛、selector rerank。
7. `CoverageAnalyzer.analyze()` 判断约束覆盖。
8. 条件触发 Neo4j graph expansion。
9. 如果 coverage 仍不足，执行第二轮 title/chunk/sparse/dense 补检。
10. 可选 `CrawlerStrategyServiceClient` 给 TopN 结果补充章节展开策略。
11. 返回 `SearchResponse`，包含 papers、plan、coverage、cost 和 diagnostic pool。

Pipeline 的 cost 字段是评测诊断的关键：

- `query_type`
- `rewrite_used`
- `dense_used`
- `sparse_paper_used`
- `alias_used`
- `diagnostic_pool_candidates`
- `model_services`
- 后端 stats

### 5.6 CandidateRanker

`CandidateRanker` 是 selector 前的规则初排。

主要计算：

- `_constraint_coverage()`：判断 candidate 是否覆盖 must-have constraints。
- `_label()`：根据覆盖率和来源数标记 highly/partially/weakly relevant。
- `_score()`：综合各类信号得到 `final_score`。

当前 score 主要信号：

| 信号 | 作用 |
| --- | --- |
| selector_relevance | must-have constraint 覆盖率 |
| source_rank_signal | 各来源 rank 的加权位置分 |
| rrf_signal | 多源 reciprocal rank fusion 分 |
| paper_sparse_synergy | sparse paper 与 ES title/chunk 的协同 |
| keyword_match | query token overlap |
| soft_alias_bonus | soft constraints 命中加分 |
| strong_alias_bonus | 高置信长短语/方法名命中加分 |
| graph_alias_bonus | Neo4j alias 支持加分 |
| citation_authority | citation count log 加分 |
| recency_score | 年份新近性 |
| exact_phrase_bonus | 硬约束精确短语命中 |
| missing_penalty | 缺失硬约束惩罚 |

这层的职责不是最终精排，而是为 selector 提供尽量完整且质量较高的候选池。

### 5.7 CandidatePreselector

`CandidatePreselector` 是第七阶段新增并固化的核心模块。

目标：

```text
初排最多 500 个候选 -> 规则预筛 120 个候选 -> selector rerank
```

预筛不是简单截断 top120，而是按 lane 组合：

| Lane | 选择逻辑 |
| --- | --- |
| `protected_rule_head` | 保留一部分原始初排高置信头部 |
| `constraint_coverage` | 优先保留约束覆盖候选 |
| `multi_source_support` | 多来源同时命中优先 |
| `title_anchor` | title BM25 靠前优先 |
| `dense_lexical_bridge` | dense 命中且有 lexical/alias 支持 |
| `sparse_lexical_bridge` | sparse paper/chunk 与 title/chunk 协同 |
| `alias_or_graph` | Neo4j alias/concept 作为补充 |
| `score_fill` | 用预筛分数补满目标数量 |

第七阶段全量诊断：

| 指标 | 数值 |
| --- | ---: |
| Avg input candidates | 496.319510 |
| Avg selected candidates | 120.000000 |
| Avg compression ratio | 0.242157 |
| Selector/model errors | 0 |

### 5.8 Selector Reranker

`SelectorRerankerServiceClient` 调用远端 `/rerank` 服务，对预筛后的候选精排。

融合逻辑：

```text
heuristic_weight = min(0.75, 0.25 + 0.35 * soft_alias_bonus + 0.35 * strong_alias_bonus)
selector_weight = 1.0 - heuristic_weight
final_score = selector_weight * selector_score + heuristic_weight * heuristic_score
```

含义：

- 普通候选更信任 selector。
- 强 alias / 强约束候选保留更多规则分，避免模型压低关键 gold。
- `_promote_strong_aliases()` 会把强 alias 候选保护到 rerank 结果头部之后。

第七阶段默认 `SELECTOR_RERANKER_PROTECTED_HEAD=0`，表示不额外保护原始初排头部，优先追求 Recall/F1。

### 5.9 Diversity 与 source rank backfill

selector 后还有两步保守修正：

1. `_diversify_ranked()`：减少 topK 中高度相似 title 堆叠，特别服务 Real / survey 多答案查询。
2. `_source_rank_backfill()`：对 source rank 靠前但融合分偏低的候选进行回填，避免 gold 已在某个可靠来源靠前却被融合分压出 top50。

这两步都限制在 Top50 输出目标附近，不改变候选池全局结构。

### 5.10 CoverageAnalyzer 与二轮补检

`CoverageAnalyzer` 分析 top results 是否覆盖 must-have constraints，并生成 `next_queries`。

触发条件：

- top 结果缺失关键约束。
- high confidence 候选不足。
- coverage report 判定 `should_continue=True`。

二轮补检动作：

- `local_title_bm25`
- `local_chunk_bm25`
- `local_tfidf`
- `qdrant_dense_paper`

二轮 dense topK 按 profile 调整，Real / survey / dataset 查询给更高 dense 预算。

## 6. 外部模型服务

当前支持四类模型服务，其中第七阶段全量评测只启用了 query rewrite 和 selector reranker。

| 服务 | 当前角色 | 第七阶段状态 |
| --- | --- | --- |
| Query Intent | 判断是否论文搜索、粗粒度意图分类 | disabled |
| Query Rewrite | 高风险 query 的学术检索改写 | enabled，GPUStack |
| Selector Reranker | 对预筛候选做最终精排 | enabled |
| Crawler Strategy | 对 Top 论文做章节展开策略 | disabled |

Query rewrite 使用 OpenAI-compatible chat/completions 接口，要求严格 JSON 输出：

```json
{
  "rewrites": ["3-5 diverse search queries"],
  "concepts": ["important concepts"],
  "possible_answer_terms": ["title-like or method-like terms"]
}
```

安全约束：

- 不允许输出 paper ID、arXiv ID 或 citation。
- 不使用已知 gold labels。
- 使用 cache key 缓存 query + model + parsed context，降低重复调用成本。

## 7. 配置体系

推荐配置目录是 `configs`，旧 `config` 目录仍兼容。

主要配置文件：

```text
configs/app.env.example
configs/database.env.example
configs/model-services.env.example
```

核心运行配置：

| 配置 | 默认/当前含义 |
| --- | --- |
| `MODEL_SERVICES_ENABLED` | 是否启用模型服务总开关 |
| `QUERY_REWRITE_ENABLED` | 是否启用 GPUStack query rewrite |
| `QUERY_REWRITE_CACHE_PATH` | query rewrite 缓存位置 |
| `SELECTOR_RERANKER_ENABLED` | 是否启用 selector rerank |
| `SELECTOR_RERANKER_POOL_LIMIT` | 预筛前候选池，默认 500 |
| `SELECTOR_RERANKER_CANDIDATE_LIMIT` | 送入 selector 的候选数，默认 120 |
| `SELECTOR_RERANKER_PROTECTED_HEAD` | rerank 后保护原始头部数量，默认 0 |
| `QDRANT_DENSE_PAPER_ENABLED` | 是否启用 dense paper retrieval |
| `QDRANT_DENSE_PAPER_COLLECTION` | `saiti3_papers_dense_v1` |
| `DENSE_EMBEDDING_BACKEND` | `sentence_transformers` 或 `gpustack` |
| `DENSE_EMBEDDING_MODEL` | 第六阶段为 `qwen3-embedding-4b` |
| `NEO4J_RETRIEVAL_ENABLED` | 是否启用 Neo4j concept/alias |

`packages/scholar_infra/config.py` 管模型服务和 Neo4j 配置。
`packages/scholar_ingest/config.py` 管数据库、ES、Qdrant、dense ingest/retrieval 配置。

## 8. API 与前端边界

后端 API 层只负责：

- 启动服务。
- 解析 HTTP 参数。
- 调用 `SearchPipeline`。
- 返回结构化 JSON。
- 暴露健康检查。

前端只负责：

- 搜索输入和状态管理。
- 调用后端 API。
- 展示结果、证据、trace、coverage、cost。

前端不得读取或硬编码：

- MySQL / ES / Qdrant / Neo4j 地址。
- 模型服务地址。
- 后端内部 pipeline 类名。
- 数据库密码或 API key。

## 9. 当前主要瓶颈

第七阶段最终诊断：

| Primary reason | Queries |
| --- | ---: |
| `gold_not_found_in_diagnostic_pool` | 321 |
| `top_results_miss_required_constraints` | 291 |
| `gold_retrieved_but_ranked_below_cutoff` | 237 |
| `acceptable_for_current_cutoff` | 130 |
| `partial_recall_needs_better_ranking_or_expansion` | 71 |

Pool recall：

| Pool cutoff | Recall |
| --- | ---: |
| @100 | 0.360501 |
| @200 | 0.429285 |
| @500 | 0.503575 |

判断：

1. 召回池上限仍是第一瓶颈，`pool_recall@500` 只有 0.503575。
2. 规则预筛扩大到 120 后已经释放一部分 `gold_retrieved_but_ranked_below_cutoff` 收益。
3. `top_results_miss_required_constraints` 仍多，说明 dense 语义相似不等于约束完整。
4. Neo4j alias 当前贡献低且 false positive 高，只能作为弱补充。
5. Query rewrite 使用率接近 46%，后续需要消融确认净收益和成本。

## 10. 后续优化方向

推荐顺序：

1. 继续优化召回池：SearchPlanner 子查询扩展、ES 字段权重、同义词/别名、chunk 召回、Qdrant dense/sparse 覆盖。
2. 做 query rewrite ablation：比较 dense only、dense + rewrite、dense + rewrite + coverage rerank。
3. 导出 top500 candidate feature table，为轻量排序校准准备数据。
4. 做 constraint coverage rerank，降低主题相似但缺少关键约束的候选。
5. 细分 query type routing，减少 `real_multi_answer` 过大问题。
6. 建立 alias graph 高置信过滤，弱 alias 只作为 support，不直接前推。
7. 如果允许训练，再用 hard negative 校准 selector reranker 或训练 profile-specific ranker。
8. 控制延迟：query embedding cache、rewrite 更严格 gate、profile-specific dense topK、动态 selector candidate limit。

下一阶段务实目标：

| 指标 | 当前 full | 下一阶段目标 |
| --- | ---: | ---: |
| Overall Recall@50 | 0.315106 | >= 0.33 |
| Auto Recall@50 | 0.311902 | >= 0.33 |
| Real Recall@50 | 0.379195 | 不退化 |
| Pool Recall@500 | 0.503575 | >= 0.55 |
| Failed queries | 0 | 0 |

## 11. 架构维护规则

1. `apps/frontend` 不读取后端 Python 代码和任何数据库/模型服务配置。
2. `apps/backend` 只做 HTTP/CLI 入口和依赖装配，不写检索算法。
3. `packages/scholar_core` 不读取环境变量，不创建外部连接。
4. `packages/scholar_infra` 负责外部系统适配，不向 core 反向依赖应用层。
5. `packages/scholar_ingest` 是离线导入层，不应被在线后端直接作为业务依赖。
6. 新增召回源必须写入 source ranks、RRF 或诊断字段，否则无法评测贡献。
7. 修改 selector candidate 数、query rewrite gate、dense topK、alias 权重时必须跑 100 proportional + 100 balanced gate，再决定是否跑全量。
8. API 响应字段变更必须同步前端和 contract 测试。
9. 训练工程与在线服务隔离，在线侧只通过 HTTP client 或 embedding client 使用模型能力。
10. 架构文档只保留本文件作为最终版，启动和运维细节放在 `docs/operations`。
