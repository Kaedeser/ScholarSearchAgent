# 中文功能说明：Python 包初始化文件，声明当前目录为可导入模块并暴露必要对象。

"""5.7 Relevance judgment and ranking module."""

from .ranking import CandidateRanker

__all__ = ["CandidateRanker"]
