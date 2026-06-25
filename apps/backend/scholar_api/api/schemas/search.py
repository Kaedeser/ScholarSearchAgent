# 中文功能说明：检索接口参数解析模块，负责把 HTTP 查询字符串转换为稳定的请求字段。

from __future__ import annotations

from urllib.parse import parse_qs


def parse_search_query(raw_query: str) -> tuple[str, int]:
    params = parse_qs(raw_query)
    query = (params.get("q") or [""])[0].strip()
    top_k = _safe_int((params.get("top_k") or ["10"])[0], default=10)
    return query, top_k


def _safe_int(value: str, *, default: int) -> int:
    try:
        return max(1, int(value))
    except ValueError:
        return default
