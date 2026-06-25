# 中文功能说明：旧版离线评测兼容入口，实际实现位于 packages/scholar_eval/evaluation.py。

from __future__ import annotations

from packages.scholar_eval.evaluation import Evaluator, QueryMetrics, score_prediction

__all__ = ["Evaluator", "QueryMetrics", "score_prediction"]