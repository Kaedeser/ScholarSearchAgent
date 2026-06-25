# 中文功能说明：后端应用启动参数模型，集中描述 HTTP 服务运行时配置。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiRuntimeSettings:
    processed_dir: Path
    host: str
    port: int
    paper_limit: int | None
    chunk_limit: int | None
    max_chunks_per_paper: int
    per_query_top_k: int
    backend: str
    model_services_enabled: bool | None
