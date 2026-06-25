# 部署与启动

## 本地开发

后端：

```powershell
python -m apps.backend.scholar_api.cli --backend auto serve --host 127.0.0.1 --port 8765
```

前端：

```powershell
cd apps/frontend
npm install
npm run dev
```

前端生产构建：

```powershell
cd apps/frontend
npm run build
```

## 离线调试

```powershell
python -m apps.backend.scholar_api.cli --backend jsonl --disable-model-services serve --port 8765
```

## 数据库后端

```powershell
python -m apps.backend.scholar_api.cli --backend database serve --port 8765
```

## 配置

默认优先读取：

```text
configs/database.env
```

也可以通过环境变量指定：

```powershell
$env:SCHOLAR_SEARCH_CONFIG = "F:\path\to\database.env"
```

## 生产形态

```text
React/Vite 构建产物 / Nginx
  -> Backend API
    -> MySQL / Elasticsearch / Qdrant / Neo4j
    -> Query Intent Service
    -> Selector Reranker Service
    -> Crawler Strategy Service
```
