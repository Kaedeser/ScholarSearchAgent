# 中文功能说明：pytest 全局配置，确保项目根目录和数据接入模块可被测试导入。

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
INGESTION_ROOT = PROJECT_ROOT / "data_ingestion_indexing"
PACKAGES_ROOT = PROJECT_ROOT / "packages"
BACKEND_ROOT = PROJECT_ROOT / "apps" / "backend"

for path in (PROJECT_ROOT, PACKAGES_ROOT, BACKEND_ROOT, INGESTION_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
