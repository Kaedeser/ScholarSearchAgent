# ScholarSearch-Agent 配置说明

本目录统一管理 ScholarSearch-Agent 的数据库、索引库和图数据库连接配置。

默认配置文件：

```text
database.env
```

模板文件：

```text
database.env.example
```

虚拟环境使用

```
cu02 的
/home/model_train/py-train
```

以下入口会自动读取该配置文件：

- `data_ingestion_indexing/scholar_ingest`
- `data_ingestion_indexing/scripts/server_ingest.py`
- `data_ingestion_indexing/scripts/build_neo4j_paper_kg.py`
- `scholar_app --backend auto|database`

## 读取优先级

配置读取优先级从高到低：

1. 当前 shell 中的环境变量。
2. `SCHOLAR_SEARCH_CONFIG` 指向的 env 文件。
3. 本目录下的 `database.env`。
4. 代码内默认值。

## Neo4j 配置

当前 Neo4j 服务只暴露默认数据库 `neo4j`，不支持创建独立数据库 `paper`。论文知识图谱实际存放在 `neo4j` 数据库中，并通过 `NEO4J_GRAPH_NAME=paper`、`(:Graph {name: "paper"})` 和节点/关系属性 `graph_name="paper"` 标识。

```text
NEO4J_HTTP_URL=http://10.99.24.182:30474
NEO4J_BOLT_URL=bolt://10.99.24.182:30687
NEO4J_URI=bolt://10.99.24.182:30687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j
NEO4J_GRAPH_NAME=paper
```

`build_neo4j_paper_kg.py` 使用 `NEO4J_HTTP_URL` 写入和维护图谱；如果后续应用侧使用官方 Neo4j driver，优先使用 `NEO4J_URI` 或 `NEO4J_BOLT_URL`。

## 路径规则

配置文件中的相对路径按配置文件所在目录解析。例如：

```text
PASA_DATA_ROOT=../../数据集/pasa/data
PROCESSED_DIR=../data_ingestion_indexing/data_processed
LOG_DIR=../data_ingestion_indexing/logs
```

无论从 `data_ingestion_indexing` 还是项目根目录运行命令，都会解析到同一位置。

## 常用检查

```bash
cd ../data_ingestion_indexing
python -m scholar_ingest.cli doctor --check-mysql --check-es --check-qdrant
python -m scholar_ingest.cli verify-all
```

Neo4j 图谱脚本检查：

```bash
python scripts/build_neo4j_paper_kg.py doctor
```

远端导入脚本检查：

```bash
python scripts/server_ingest.py status
```

Demo 使用数据库后端：

```bash
cd ..
python -m scholar_app.cli --backend database search --query "image retrieval" --top-k 3
```

## 更换环境

临时使用另一份配置：

```bash
set SCHOLAR_SEARCH_CONFIG=F:\path\to\database.env
```

临时覆盖单项：

```bash
set MYSQL_PASSWORD=...
```

注意：`database.env` 含数据库密码和 API key，已在本目录 `.gitignore` 中忽略；需要共享配置结构时使用 `database.env.example`。
