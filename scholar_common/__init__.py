# 中文功能说明：Python 包初始化文件，声明当前目录为可导入模块并暴露必要对象。

"""Shared data structures and helpers for ScholarSearch-Agent modules."""

from .config import ModelServiceSettings, ScholarSearchSettings
from .models import (
    Candidate,
    CoverageReport,
    Paper,
    QueryIntent,
    SearchAction,
    SearchPlan,
    SearchResponse,
)

__all__ = [
    "Candidate",
    "CoverageReport",
    "ModelServiceSettings",
    "Paper",
    "QueryIntent",
    "ScholarSearchSettings",
    "SearchAction",
    "SearchPlan",
    "SearchResponse",
]
