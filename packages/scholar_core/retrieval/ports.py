# 中文功能说明：召回后端端口定义，隔离核心检索流水线和 JSONL/数据库等具体召回实现。

from __future__ import annotations

from typing import Protocol

from packages.scholar_core.models import Candidate, SearchAction


class CorpusBackend(Protocol):
    backend_name: str

    def run_action(self, action: SearchAction) -> list[Candidate]:
        ...

    def stats(self) -> dict:
        ...
