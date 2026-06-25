# 模块边界约定

## 依赖方向

```text
apps/frontend
  -> 后端公开 HTTP API

apps/backend/scholar_api
  -> packages/scholar_core
  -> packages/scholar_infra
  -> packages/scholar_eval

packages/scholar_core
  -> 仅依赖自身模型、端口和纯业务逻辑

packages/scholar_infra
  -> 外部数据库、索引、向量库、模型 HTTP 服务、JSONL IO

packages/scholar_ingest
  -> 离线数据转换和索引构建

training
  -> 模型训练工程，不被在线后端直接 import
```

## 规则

1. `apps/frontend` 不读取 Python 代码和数据库配置。
2. `apps/backend` 不写排序、归一、覆盖分析等业务算法，只装配和转发。
3. `packages/scholar_core` 不读取 env，不访问 HTTP，不创建数据库客户端。
4. `packages/scholar_infra` 不依赖前端和 API route。
5. `packages/scholar_ingest` 面向离线任务，在线后端只复用 `scholar_infra.persistence` 客户端。
6. 旧根级目录仅作兼容入口，新功能必须进入 `apps` 或 `packages`。
