# 中文功能说明：Python 包初始化文件，声明当前目录为可导入模块并暴露必要对象。

"""5.6 Candidate deduplication and normalization module."""

from .normalizer import CandidateNormalizer, canonical_id

__all__ = ["CandidateNormalizer", "canonical_id"]
