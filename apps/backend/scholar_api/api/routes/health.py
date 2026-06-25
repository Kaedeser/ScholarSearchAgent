# 中文功能说明：健康检查路由逻辑，负责返回后端服务状态和公开接口列表。

from __future__ import annotations


def health_response() -> dict:
    return {
        "status": "ok",
        "service": "scholar-search-api",
        "endpoints": ["/health", "/api/search"],
    }
