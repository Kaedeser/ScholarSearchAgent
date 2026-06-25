# ScholarSearch API 合约

## 健康检查

```http
GET /health
```

响应字段：

```json
{
  "status": "ok",
  "service": "scholar-search-api",
  "endpoints": ["/health", "/api/search"]
}
```

## 检索接口

```http
GET /api/search?q=image%20retrieval&top_k=10
```

请求参数：

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `q` | 是 | 用户检索问题 |
| `top_k` | 否 | 返回论文数量，默认 `10` |

响应顶层字段保持稳定：

```text
query
parsed_query
plan
coverage
cost
papers
```

前端只依赖这些公开字段，不依赖后端内部类名、模块名或数据库结构。
