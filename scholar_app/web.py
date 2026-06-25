# 中文功能说明：旧版后端 API 兼容入口，实际转发到 apps/backend/scholar_api/main.py。

from __future__ import annotations

from apps.backend.scholar_api.main import ApiServer, run_server

__all__ = ["ApiServer", "run_server"]