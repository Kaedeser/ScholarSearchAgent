# 中文功能说明：旧版命令行兼容入口，实际转发到 apps/backend/scholar_api/cli.py。

from __future__ import annotations

from apps.backend.scholar_api.cli import build_parser, build_pipeline, cmd_eval, cmd_search, cmd_serve, main


if __name__ == "__main__":
    main()