# scholar_app 后端 API

`scholar_app` 是 ScholarSearchAgent 的后端 API 与 CLI 入口。后端没有使用 Flask、FastAPI、Django 等第三方框架，当前基于 Python 标准库：

```text
http.server.ThreadingHTTPServer
```

检索业务逻辑仍在 `cost_control_cache.SearchPipeline`，本模块负责：

- 启动 JSON API 服务。
- 暴露 `/health` 和 `/api/search`。
- 提供 CLI search/eval 命令。

前端已经独立到：

```text
scholar_frontend
```

## 启动后端

从项目根目录运行：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent
python -m scholar_app.cli --backend auto serve --host 127.0.0.1 --port 8765
```

健康检查：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -Method Get
```

检索 API：

```text
GET http://127.0.0.1:8765/api/search?q=<query>&top_k=<number>
```

## 后端模式

真实数据库优先，失败时回退到本地 JSONL：

```powershell
python -m scholar_app.cli --backend auto serve --port 8765
```

强制真实数据库：

```powershell
python -m scholar_app.cli --backend database serve --port 8765
```

离线 JSONL：

```powershell
python -m scholar_app.cli --backend jsonl --disable-model-services serve --port 8765
```

## CLI 检索

```powershell
python -m scholar_app.cli --backend auto search --query "image retrieval" --top-k 5
```

输出 Markdown：

```powershell
python -m scholar_app.cli --backend auto search --query "image retrieval" --top-k 5 --format markdown
```

保存结果文件：

```powershell
python -m scholar_app.cli --backend auto search `
  --query "image retrieval" `
  --top-k 5 `
  --output-dir .\runs\image_retrieval
```

完整前后端启动说明见：

```text
FRONTEND_BACKEND_STARTUP.md
```
