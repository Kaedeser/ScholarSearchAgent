# ScholarSearchAgent 项目结构

本目录是赛题三 ScholarSearchAgent 的主项目根目录。当前已按“前端应用、后端应用、领域核心、基础设施、数据导入、评测、训练资产、文档与配置”拆分，旧的根级模块保留为兼容入口。

## 当前目录

| 层级 | 目录 | 职责 |
| --- | --- | --- |
| 前端应用 | `apps/frontend` | React/Vite 论文检索工作台，只调用后端公开 API |
| 后端应用 | `apps/backend/scholar_api` | 标准库 HTTP API、CLI、路由、参数解析和依赖装配 |
| 领域核心 | `packages/scholar_core` | 领域模型、查询理解、规划、排序、归一、覆盖分析、引用扩展、结果组织、检索流水线 |
| 基础设施 | `packages/scholar_infra` | JSONL IO、配置读取、数据库/索引/向量库客户端、远端模型服务客户端、召回后端 |
| 数据导入 | `packages/scholar_ingest` | PaSa 转换、MySQL/ES/Qdrant 导入、Neo4j 图谱构建脚本 |
| 离线评测 | `packages/scholar_eval` | Precision、Recall、F1、MRR 等离线评测 |
| 配置 | `configs` | 数据库、模型服务和应用配置模板；本地密钥文件不提交 |
| 文档 | `docs/architecture`、`docs/operations` | 架构、启动、模型服务和运维说明 |
| 训练资产 | `training` | 三个模型训练工程与训练计划，和在线后端隔离 |
| 兼容入口 | `scholar_app`、`scholar_common`、根级业务目录 | 只转发到 `apps` / `packages` 新实现，便于旧命令过渡 |

## 后端框架

后端仍使用 Python 标准库：

```text
http.server.ThreadingHTTPServer
```

目前没有引入 Flask、FastAPI 或 Django。后续如果需要 OpenAPI、Pydantic schema 和生产中间件，可以在保持 `/health`、`/api/search` 合约不变的前提下迁移到 FastAPI。

## 配置入口

优先使用：

```text
configs/database.env
```

模板文件：

```text
configs/database.env.example
```

读取优先级为：环境变量 > `SCHOLAR_SEARCH_CONFIG` 指向的 env 文件 > `configs/database.env` > 旧兼容路径 `config/database.env` > 代码默认值。

三个本地训练并部署的远端模型服务默认启用，可通过同一份配置覆盖：

- Query Intent：`QUERY_INTENT_SERVICE_URL`
- Selector Reranker：`SELECTOR_RERANKER_SERVICE_URL`
- Crawler Strategy：`CRAWLER_STRATEGY_SERVICE_URL`

外部学术搜索 API 可选启用：

- Semantic Scholar：`ACADEMIC_SEARCH_ENABLED=true`、`ACADEMIC_SEARCH_PROVIDER=semantic_scholar`
- `semantic_scholar` 后端只依赖公开 API；`database` 后端会把 API 结果与 ES、Qdrant、Neo4j 结果融合粗排
- `auto` 会优先使用数据库，数据库初始化失败且 Semantic Scholar 已启用时直接回退到 API-only 后端

Query Rewrite 和 Dense Embedding 依赖单独的 GPUStack/OpenAI-compatible 凭据，仍按各自开关显式启用。离线调试可在命令中添加 `--disable-model-services`，一次性关闭三个训练模型服务。

## 常用命令

启动后端 API：

```powershell
python -m apps.backend.scholar_api.cli --backend auto serve --host 127.0.0.1 --port 8765
```

启动前端页面：

```powershell
cd apps/frontend
npm install
npm run dev
```

构建前端页面：

```powershell
cd apps/frontend
npm run build
```

命令行检索：

```powershell
python -m apps.backend.scholar_api.cli --backend auto search --query "image retrieval" --top-k 5
```

强制真实数据库后端：

```powershell
python -m apps.backend.scholar_api.cli --backend database serve --host 127.0.0.1 --port 8765
```

仅使用 Semantic Scholar（无需配置 ES、Qdrant、Neo4j）：

```powershell
python -m apps.backend.scholar_api.cli --backend semantic_scholar serve --host 127.0.0.1 --port 8765
```

离线 JSONL 后端：

```powershell
python -m apps.backend.scholar_api.cli --backend jsonl --disable-model-services serve --host 127.0.0.1 --port 8765
```

验证数据导入配置：

```powershell
python -m packages.scholar_ingest.cli doctor --check-mysql --check-es --check-qdrant
python -m packages.scholar_ingest.cli verify-all
```

运行测试：

```powershell
python -m pytest -q
```

## 文档

- 前后端启动：[docs/operations/FRONTEND_BACKEND_STARTUP.md](docs/operations/FRONTEND_BACKEND_STARTUP.md)
- 架构方案：[docs/architecture/PROJECT_ARCHITECTURE.md](docs/architecture/PROJECT_ARCHITECTURE.md)
- 模型服务：[docs/operations/MODEL_SERVICES_SUMMARY.md](docs/operations/MODEL_SERVICES_SUMMARY.md)

## 开发约定

新代码优先放入 `apps` 或 `packages` 对应层级。旧根级模块只用于兼容，不再承载新的业务实现。
