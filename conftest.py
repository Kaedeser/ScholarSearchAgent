from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
INGESTION_ROOT = PROJECT_ROOT / "data_ingestion_indexing"

for path in (PROJECT_ROOT, INGESTION_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
