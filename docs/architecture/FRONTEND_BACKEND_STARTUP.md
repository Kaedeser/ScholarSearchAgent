# ScholarSearchAgent 前后端独立启动文档

## 1. 技术栈

后端当前没有使用 Flask、FastAPI、Django 等第三方 Web 框架，而是基于 Python 标准库：

```text
http.server.ThreadingHTTPServer
```

前后端已经物理拆分：

| 部分 | 主目录 | 端口 | 说明 |
| --- | --- | ---: | --- |
| 后端 API | `apps/backend/scholar_api` | `8765` | 只提供 JSON API 和健康检查 |
| 前端页面 | `apps/frontend` | `5174` | React/Vite 论文检索工作台，通过 fetch 调后端 API |

旧目录 `scholar_app` 和 `scholar_frontend` 仅作为兼容入口，新的开发和启动文档以 `apps/` 为准。

## 2. 启动后端

从项目根目录运行：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent
python -m apps.backend.scholar_api.cli --backend auto serve --host 127.0.0.1 --port 8765
```

健康检查：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -Method Get
```

强制真实数据库后端：

```powershell
python -m apps.backend.scholar_api.cli --backend database serve --host 127.0.0.1 --port 8765
```

离线 JSONL 后端，并关闭远端模型服务：

```powershell
python -m apps.backend.scholar_api.cli --backend jsonl --disable-model-services serve --host 127.0.0.1 --port 8765
```

## 3. 启动前端

另开一个终端，从项目根目录运行：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent
cd apps/frontend
npm install
npm run dev
```

浏览器打开：

```text
http://127.0.0.1:5174
```

前端默认后端地址为：

```text
http://127.0.0.1:8765
```

页面顶部的 API 输入框可以切换到其他后端地址，并会保存到浏览器 localStorage。

也可以使用项目根目录脚本启动：

```powershell
.\scripts\dev_frontend.ps1
```

如果本机端口策略不同，也可以指定端口：

```powershell
.\scripts\dev_frontend.ps1 -Port 5175
```

生产构建：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent\apps\frontend
npm run build
```

本地预览生产构建：

```powershell
npm run preview
```

## 4. API 调用

健康检查：

```http
GET /health
```

检索接口：

```http
GET /api/search?q=<query>&top_k=<number>
```

PowerShell 示例：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8765/api/search?q=image%20retrieval&top_k=3" `
  -Method Get
```

返回 JSON 顶层字段包括：

```text
papers
parsed_query
plan
coverage
cost
```

## 5. 模型服务配置

配置优先放在：

```text
configs/database.env
```

旧路径 `config/database.env` 仍作为兼容回退。远端模型服务配置项：

```text
MODEL_SERVICES_ENABLED=true
MODEL_SERVICE_TIMEOUT_SEC=8
QUERY_INTENT_ENABLED=true
QUERY_INTENT_SERVICE_URL=http://10.99.24.182:22436
QUERY_INTENT_MODE=auto
SELECTOR_RERANKER_ENABLED=true
SELECTOR_RERANKER_SERVICE_URL=http://10.99.24.182:32082
SELECTOR_RERANKER_CANDIDATE_LIMIT=100
CRAWLER_STRATEGY_ENABLED=true
CRAWLER_STRATEGY_SERVICE_URL=http://10.99.24.182:32183
CRAWLER_STRATEGY_TOP_N=3
```

临时关闭模型服务：

```powershell
python -m apps.backend.scholar_api.cli --backend auto --disable-model-services serve --port 8765
```

## 6. 数据库后端检查

```powershell
python -m packages.scholar_ingest.cli doctor --check-mysql --check-es --check-qdrant
python -m packages.scholar_ingest.cli verify-all
```

## 7. 常见问题

### 前端显示 API unavailable

先确认后端已启动：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -Method Get
```

### 后端端口被占用

换一个端口：

```powershell
python -m apps.backend.scholar_api.cli --backend auto serve --port 8766
```

同时把前端页面顶部 API 地址改成：

```text
http://127.0.0.1:8766
```

### 数据库不可达

先用离线模式确认前后端链路：

```powershell
python -m apps.backend.scholar_api.cli --backend jsonl --disable-model-services serve --port 8765
```

再检查 `configs/database.env` 或旧兼容路径 `config/database.env`。
