# ScholarSearchAgent 完整项目前后端分层架构设计

## 0. 已落地状态

本项目已经按本文目标结构完成第一轮目录调整：

- 后端主入口：`apps/backend/scholar_api`
- 前端主入口：`apps/frontend`
- 领域核心：`packages/scholar_core`
- 基础设施：`packages/scholar_infra`
- 数据导入：`packages/scholar_ingest`
- 离线评测：`packages/scholar_eval`
- 配置模板：`configs`
- 架构/运维文档：`docs`
- 三个模型训练工程：`training`

旧目录 `scholar_app`、`scholar_common`、各根级业务模块和 `scholar_frontend` 仅作为兼容入口或历史资产保留，新开发以 `apps` 和 `packages` 为准。
## 1. 目标

当前项目已经具备检索流水线、数据库接入、三个远端模型服务调用、后端 API 和独立静态前端。下一步如果要做成一个层级清晰、可维护、可部署的完整项目，建议按“前端应用、后端应用、领域核心、基础设施、训练与运维”拆开。

设计目标：

- 前后端物理目录独立，能分别启动、测试、部署。
- 后端 API 层只处理 HTTP，不承载检索业务细节。
- 检索业务模块按领域能力分层，不让 Web、数据库、模型 HTTP 调用相互穿透。
- 配置、数据导入、模型服务、评测、训练产物有清晰边界。
- 保留当前代码资产，迁移时尽量移动和重命名，少做大规模重写。

## 2. 当前后端框架判断

当前后端不是 Flask、FastAPI、Django，而是 Python 标准库：

```text
http.server.ThreadingHTTPServer
```

优点是零依赖、启动简单；缺点是路由、参数校验、OpenAPI、错误处理、中间件、日志和生产部署能力较弱。

建议演进路线：

1. 短期保留标准库 HTTP server，先完成前后端目录拆分和业务层分层。
2. 中期迁移到 FastAPI，补齐 OpenAPI、Pydantic 请求/响应模型、健康检查、CORS、中间件和统一异常处理。
3. 生产部署时用 `uvicorn`/`gunicorn` 承载 FastAPI；前端用 Nginx 或静态资源服务。

## 3. 推荐目标目录结构

建议把项目根目录整理成下面的结构：

```text
ScholarSearchAgent/
  apps/
    backend/
      scholar_api/
        __init__.py
        main.py
        api/
          __init__.py
          routes/
            __init__.py
            health.py
            search.py
            evaluation.py
          schemas/
            __init__.py
            search.py
            common.py
        bootstrap/
          __init__.py
          container.py
          settings.py
        cli.py
      tests/
        test_health_api.py
        test_search_api.py

    frontend/
      public/
      src/
        api/
          client.js
          searchApi.js
        components/
          ApiStatus.js
          SearchBox.js
          ResultTable.js
          TracePanel.js
        pages/
          SearchPage.js
        styles/
          base.css
          layout.css
        app.js
        main.js
      index.html
      README.md

  packages/
    scholar_core/
      __init__.py
      models.py
      text.py
      pipeline.py
      query_understanding/
        parser.py
      planning/
        planner.py
      retrieval/
        ports.py
        service.py
      ranking/
        ranker.py
      normalization/
        normalizer.py
      coverage/
        analyzer.py
      citation/
        planner.py
      composition/
        composer.py

    scholar_infra/
      __init__.py
      config.py
      persistence/
        mysql.py
        elasticsearch.py
        qdrant.py
      retrieval_backends/
        local_jsonl.py
        database.py
      model_services/
        client.py
        query_intent.py
        selector_reranker.py
        crawler_strategy.py
      io/
        jsonl.py

    scholar_ingest/
      __init__.py
      cli.py
      pasa.py
      ids.py
      jobs/
        build_indices.py
        build_neo4j_graph.py
      sql/
        schema.sql

    scholar_eval/
      __init__.py
      evaluation.py
      metrics.py

  configs/
    app.env.example
    model-services.env.example
    database.env.example

  docs/
    architecture/
      PROJECT_ARCHITECTURE_PLAN.md
      API_CONTRACT.md
      DEPLOYMENT.md
      MODULE_BOUNDARIES.md
    operations/
      FRONTEND_BACKEND_STARTUP.md
      MODEL_SERVICES_SUMMARY.md

  training/
    plans/
    query_intent_model/
    selector_reranker_model/
    crawler_strategy_model/

  scripts/
    dev_backend.ps1
    dev_frontend.ps1
    test_all.ps1

  tests/
    integration/
      test_pipeline_with_fake_services.py
    contract/
      test_search_response_contract.py

  pyproject.toml
  README.md
```

前端已升级为独立 React/Vite 工程，`apps/frontend` 只负责页面交互和 HTTP API 调用，构建产物输出到 `apps/frontend/dist`。

## 4. 分层职责

### 4.1 `apps/backend`

后端应用层，只负责 HTTP/CLI 入口。

应该包含：

- 路由定义。
- 请求参数解析和校验。
- 响应格式转换。
- CORS、日志、错误处理。
- 应用启动和依赖装配。

不应该包含：

- 检索排序算法。
- 数据库查询细节。
- 模型服务 HTTP 细节。
- PaSa 数据转换逻辑。

当前对应代码：

| 当前目录 | 目标位置 |
| --- | --- |
| `scholar_app/web.py` | `apps/backend/scholar_api/main.py` + `api/routes/search.py` |
| `scholar_app/cli.py` | `apps/backend/scholar_api/cli.py` |

### 4.2 `apps/frontend`

前端应用层，只负责用户界面和调用后端 API。

应该包含：

- 搜索框、结果表格、Trace 面板、模型状态展示。
- API client 封装。
- 前端样式和页面状态管理。

不应该包含：

- 后端检索规则。
- 数据库地址、密码。
- 模型服务真实地址。

当前对应代码：

| 当前目录 | 目标位置 |
| --- | --- |
| `scholar_frontend/index.html` | `apps/frontend/index.html` |
| `scholar_frontend/app.js` | `apps/frontend/src/app.js` 或拆到 `src/api`、`src/components` |
| `scholar_frontend/styles.css` | `apps/frontend/src/styles/base.css` |

### 4.3 `packages/scholar_core`

领域核心层，负责 ScholarSearchAgent 的业务能力。

核心原则：

- 不直接读取环境变量。
- 不直接访问 HTTP。
- 不直接访问数据库驱动。
- 通过接口/端口调用外部能力。

应该包含：

- `SearchPipeline`
- `QueryIntent`、`Candidate`、`SearchPlan` 等领域模型。
- 查询理解、搜索规划、候选归一、排序、覆盖分析、结果组织。

当前对应代码：

| 当前目录 | 目标位置 |
| --- | --- |
| `scholar_common/models.py` | `packages/scholar_core/models.py` |
| `scholar_common/text.py` | `packages/scholar_core/text.py` |
| `cost_control_cache/pipeline.py` | `packages/scholar_core/pipeline.py` |
| `query_understanding_decomposition/query.py` | `packages/scholar_core/query_understanding/parser.py` |
| `search_strategy_planning/planner.py` | `packages/scholar_core/planning/planner.py` |
| `candidate_normalization/normalizer.py` | `packages/scholar_core/normalization/normalizer.py` |
| `relevance_ranking/ranking.py` | `packages/scholar_core/ranking/ranker.py` |
| `coverage_iteration/coverage.py` | `packages/scholar_core/coverage/analyzer.py` |
| `citation_network_expansion/citation.py` | `packages/scholar_core/citation/planner.py` |
| `result_composition/composer.py` | `packages/scholar_core/composition/composer.py` |

### 4.4 `packages/scholar_infra`

基础设施层，负责外部系统接入。

应该包含：

- MySQL、Elasticsearch、Qdrant client。
- 本地 JSONL 后端。
- 数据库检索后端。
- 三个远端模型服务 client。
- 配置读取。

当前对应代码：

| 当前目录 | 目标位置 |
| --- | --- |
| `scholar_common/config.py` | `packages/scholar_infra/config.py` |
| `scholar_common/model_services.py` | `packages/scholar_infra/model_services/client.py` |
| `multi_source_retrieval/retrieval.py` | 拆分到 `retrieval_backends/local_jsonl.py`、`retrieval_backends/database.py` |
| `data_ingestion_indexing/scholar_ingest/mysql.py` | `packages/scholar_infra/persistence/mysql.py` |
| `data_ingestion_indexing/scholar_ingest/es.py` | `packages/scholar_infra/persistence/elasticsearch.py` |
| `data_ingestion_indexing/scholar_ingest/qdrant.py` | `packages/scholar_infra/persistence/qdrant.py` |

### 4.5 `packages/scholar_ingest`

数据接入与索引构建层，负责离线导入任务。

应该包含：

- PaSa 数据转换。
- MySQL/ES/Qdrant 初始化和导入。
- Neo4j 图谱构建脚本。
- 导入校验。

不应该被在线后端直接依赖。在线后端最多复用 `scholar_infra.persistence` 中的 client。

当前对应代码：

| 当前目录 | 目标位置 |
| --- | --- |
| `data_ingestion_indexing/scholar_ingest/*` | `packages/scholar_ingest/*` |
| `data_ingestion_indexing/scripts/server_ingest.py` | `packages/scholar_ingest/jobs/server_ingest.py` |
| `data_ingestion_indexing/scripts/build_neo4j_paper_kg.py` | `packages/scholar_ingest/jobs/build_neo4j_graph.py` |
| `data_ingestion_indexing/sql/schema.sql` | `packages/scholar_ingest/sql/schema.sql` |

### 4.6 `packages/scholar_eval`

评测层，负责离线指标和回归验证。

当前对应代码：

| 当前目录 | 目标位置 |
| --- | --- |
| `offline_evaluation/evaluation.py` | `packages/scholar_eval/evaluation.py` |

### 4.7 `training`

训练与模型产物层，和在线应用隔离。

建议将当前：

```text
train/
training_plans/
```

整理为：

```text
training/
  plans/
  query_intent_model/
  selector_reranker_model/
  crawler_strategy_model/
```

在线后端只通过 HTTP 调用模型服务，不直接 import 训练代码。

## 5. 后端内部推荐分层

后端推荐依赖方向：

```text
api/routes
  -> bootstrap/container
    -> scholar_core services
      -> scholar_core ports/interfaces
        <- scholar_infra adapters
```

不要出现反向依赖：

```text
scholar_core -> apps.backend
scholar_core -> scholar_infra.config
scholar_core -> HTTP request object
```

建议后端结构：

```text
apps/backend/scholar_api/
  main.py                 # 创建 HTTP/FastAPI app
  cli.py                  # 命令行入口
  api/routes/search.py    # /api/search
  api/routes/health.py    # /health
  api/schemas/search.py   # 请求/响应 DTO
  bootstrap/settings.py   # 应用配置
  bootstrap/container.py  # 装配 pipeline、retriever、model clients
```

## 6. 前端内部分层

```text
apps/frontend/src/
  api/
    client.js       # fetch 封装、baseUrl、错误处理
    searchApi.js    # search(query, topK)
  components/
    ApiStatus.jsx
    SearchControls.jsx
    ResultList.jsx
    PaperDetail.jsx
    CoveragePanel.jsx
    TracePanel.jsx
    InsightRail.jsx
  pages/
    SearchPage.jsx
  styles/
    base.css
  utils/
    formatters.js
  App.jsx
  main.jsx
```

前端依赖方向：

```text
pages -> components -> api
```

前端不应知道：

- MySQL/ES/Qdrant 配置。
- 模型服务端口。
- 后端内部 pipeline 模块名。

前端只应知道后端公开 API，例如：

```text
GET /health
GET /api/search
```

## 7. API 合约建议

### 7.1 健康检查

```http
GET /health
```

响应：

```json
{
  "status": "ok",
  "service": "scholar-search-api",
  "version": "0.1.0"
}
```

### 7.2 检索接口

```http
GET /api/search?q=image%20retrieval&top_k=10
```

建议未来改为同时支持 POST：

```http
POST /api/search
Content-Type: application/json

{
  "query": "image retrieval",
  "top_k": 10,
  "options": {
    "backend": "auto",
    "use_model_services": true
  }
}
```

响应顶层字段保持稳定：

```json
{
  "query": "...",
  "parsed_query": {},
  "plan": {},
  "coverage": {},
  "cost": {},
  "papers": []
}
```

前端只依赖这些稳定字段，不依赖后端内部类名。

## 8. 配置设计

建议拆成三类配置：

```text
configs/
  app.env.example             # API host/port、日志级别、运行模式
  database.env.example        # MySQL/ES/Qdrant/Neo4j
  model-services.env.example  # 三个模型服务
```

开发环境优先使用 `configs/database.env`，也可以继续兼容旧的 `config/database.env`，但代码中应抽象为：

```text
AppSettings
DatabaseSettings
ModelServiceSettings
```

配置读取只发生在：

```text
apps/backend/scholar_api/bootstrap/settings.py
packages/scholar_infra/config.py
```

业务核心层不读取 env。

## 9. 测试分层

建议按层级拆测试：

```text
tests/
  unit/
    test_query_parser.py
    test_ranker.py
    test_normalizer.py
  integration/
    test_pipeline_with_local_jsonl.py
    test_pipeline_with_fake_model_services.py
  contract/
    test_search_response_contract.py

apps/backend/tests/
  test_health_api.py
  test_search_api.py

apps/frontend/
  tests/
    searchApi.test.js
```

测试原则：

- 核心业务单测不依赖数据库和远端模型。
- pipeline 集成测试用 fake model service。
- API contract 测试锁定响应字段，避免前端被破坏。
- 真实数据库/真实模型测试单独标记为 smoke，不放进默认快速测试。

## 10. 部署形态

开发环境：

```text
backend:  python -m apps.backend.scholar_api.cli --backend auto serve --port 8765
frontend: cd apps/frontend && npm run dev
```

目标环境：

```text
frontend static server / nginx
  -> calls
backend API service
  -> MySQL / Elasticsearch / Qdrant / Neo4j
  -> Query Intent Service
  -> Selector Reranker Service
  -> Crawler Strategy Service
```

推荐端口：

| 服务 | 开发端口 | 说明 |
| --- | ---: | --- |
| Frontend | 5174 | React/Vite 页面 |
| Backend API | 8765 | ScholarSearch API |
| Query Intent | 22436 | 已部署模型服务 |
| Selector Reranker | 32082 | 已部署模型服务 |
| Crawler Strategy | 32183 | 已部署模型服务 |

## 11. 迁移步骤建议

### 阶段一：文档和边界冻结

- 保留当前代码可运行。
- 补齐 API 合约文档。
- 明确前端只调 `/health` 和 `/api/search`。
- 明确后端 API 不再返回 HTML。

### 阶段二：移动前端

- 将 `scholar_frontend` 移到 `apps/frontend`。
- 将 JS 拆成 `api/client.js`、`components/*`、`main.js`。
- 保持静态启动方式不变。

### 阶段三：移动后端入口

- 将 `scholar_app` 移到 `apps/backend/scholar_api`。
- 保留兼容入口或在 README 中更新命令。
- API 层只保留路由和响应转换。

### 阶段四：整理核心包

- 将多个根级业务目录迁移到 `packages/scholar_core`。
- 修正 import。
- 保证核心层不依赖 env、HTTP、数据库驱动。

### 阶段五：整理基础设施包

- 将数据库 client、模型服务 client、本地 JSONL backend 放入 `packages/scholar_infra`。
- 通过接口注入到 pipeline。

### 阶段六：可选迁移 FastAPI

- 用 FastAPI 替换标准库 HTTP server。
- 增加 Pydantic schema、OpenAPI 文档和统一异常处理。
- 保持 `/health`、`/api/search` 合约不变。

## 12. 模块边界规则

建议写入团队开发约定：

1. `apps/frontend` 不允许 import 或读取后端 Python 代码。
2. `apps/backend` 可以依赖 `scholar_core` 和 `scholar_infra`，但不写业务算法。
3. `scholar_core` 不允许依赖 `scholar_infra`、`apps`、环境变量、HTTP 框架。
4. `scholar_infra` 不允许依赖 `apps/frontend` 或 API route。
5. `training` 不允许被在线后端直接 import。
6. API 响应字段变更必须同步 contract tests 和前端。
7. 数据库密码和模型服务地址只放配置，不硬编码在业务模块。

## 13. 当前项目到目标结构的最小映射

如果只做最小整理，不立刻大迁移，可以先按以下方式理解边界：

| 当前目录 | 角色 | 后续目标 |
| --- | --- | --- |
| `scholar_frontend` | 独立前端 | `apps/frontend` |
| `scholar_app` | 后端 API/CLI | `apps/backend/scholar_api` |
| `cost_control_cache` | Pipeline 编排 | `packages/scholar_core/pipeline.py` |
| `query_understanding_decomposition` | 查询理解 | `packages/scholar_core/query_understanding` |
| `search_strategy_planning` | 搜索规划 | `packages/scholar_core/planning` |
| `multi_source_retrieval` | 召回适配器混合层 | 拆到 `scholar_core/retrieval` + `scholar_infra/retrieval_backends` |
| `relevance_ranking` | 排序 | `packages/scholar_core/ranking` |
| `coverage_iteration` | 覆盖分析 | `packages/scholar_core/coverage` |
| `candidate_normalization` | 候选去重 | `packages/scholar_core/normalization` |
| `citation_network_expansion` | 引用扩展规划 | `packages/scholar_core/citation` |
| `result_composition` | 输出组织 | `packages/scholar_core/composition` |
| `scholar_common` | 公共模型、配置、服务 client 混合 | 拆到 `scholar_core` + `scholar_infra` |
| `data_ingestion_indexing` | 数据导入/索引构建 | `packages/scholar_ingest` |
| `offline_evaluation` | 离线评测 | `packages/scholar_eval` |
| `train`、`training_plans` | 模型训练 | `training` |

## 14. 后续建议

优先级从高到低：

1. API 合约已补到 `docs/architecture/API_CONTRACT.md`，后续接口变更需要同步更新。
2. 前端已迁入 `apps/frontend` 并拆出 `src/api`、`src/components`、`src/pages`、`src/styles`。
3. 把 `multi_source_retrieval/retrieval.py` 拆成 core port 和 infra adapter，降低耦合。
4. 配置读取已迁到 `packages/scholar_infra/config.py`，旧 `scholar_common/config.py` 仅作兼容入口。
5. 如果要长期维护，迁移后端到 FastAPI。

这样整理后，项目会从“多个比赛模块平铺”变成“前端、后端、核心领域、基础设施、训练运维”五条边界清楚的工程结构。
