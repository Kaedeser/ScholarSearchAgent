# ScholarSearch-Agent 项目结构

本目录现在作为赛题三 ScholarSearch-Agent 的主项目根目录。每个系统模块独立成一个文件夹，5.1 数据接入与索引模块继续保留在 `data_ingestion_indexing`，其余在线检索 demo 模块按系统设计目标拆分到根目录。

## 模块划分

| 模块 | 目录 | 当前职责 |
| --- | --- | --- |
| 共享层 | `scholar_common` | 数据结构、JSONL IO、文本规范化等公共能力 |
| 5.1 数据接入与索引 | `data_ingestion_indexing` | PaSa 数据转换、MySQL/Elasticsearch/Qdrant 初始化、导入和验证 |
| 5.2 查询理解与分解 | `query_understanding_decomposition` | 规则版查询解析、约束抽取、子查询生成 |
| 5.3 搜索策略规划 | `search_strategy_planning` | 多路召回动作规划和预算设置 |
| 5.4 多源召回 | `multi_source_retrieval` | JSONL 本地 BM25/TF-IDF 召回，或真实 ES + Qdrant + MySQL 后端 |
| 5.5 引用网络扩展 | `citation_network_expansion` | 从已排序候选中选择后续引用图扩展 seed |
| 5.6 候选归一与去重 | `candidate_normalization` | arXiv/title 规范化、别名合并、分数合并 |
| 5.7 相关性排序 | `relevance_ranking` | 约束覆盖、检索信号、年份/引用等融合排序 |
| 5.8 覆盖迭代 | `coverage_iteration` | 判断约束覆盖缺口并生成下一轮查询 |
| 5.9 结果组织 | `result_composition` | JSON、Markdown、BibTeX、Mermaid 输出 |
| 5.10 成本控制与编排 | `cost_control_cache` | 串联完整检索流水线并记录 latency、候选量、后端状态 |
| 5.11 离线评测 | `offline_evaluation` | Precision/Recall/F1/MRR 评测 |
| Demo 入口 | `scholar_app` | CLI/Web 入口 |
| 统一配置 | `config` | `database.env` 与 `database.env.example` |

## 配置入口

数据库、索引、数据路径配置统一放在：

```text
config/database.env
```

模板文件是：

```text
config/database.env.example
```

读取优先级为：环境变量 > `SCHOLAR_SEARCH_CONFIG` 指向的 env 文件 > `config/database.env` > 代码默认值。5.1 模块和 demo 的 `auto/database` 后端都会读取同一份配置。

## 常用命令

验证 5.1 数据库真实接入：

```bash
cd data_ingestion_indexing
python -m scholar_ingest.cli doctor --check-mysql --check-es --check-qdrant
python -m scholar_ingest.cli verify-all
```

运行 demo 检索：

```bash
python -m scholar_app.cli --backend auto search --query "image retrieval" --top-k 5
```

强制真实数据库后端：

```bash
python -m scholar_app.cli --backend database search --query "image retrieval" --top-k 3
```

启动浏览器 demo：

```bash
python -m scholar_app.cli --backend auto serve --port 8765
```

运行测试：

```bash
python -m pytest -q data_ingestion_indexing tests
```

## 开发约定

新的业务代码优先放入对应根级模块目录。`scholar_app` 只放 CLI/Web 入口，检索、排序、评测等逻辑仍放在各自模块中。
