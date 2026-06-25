# 中文功能说明：检索路由逻辑，负责调用检索流水线并转换为前端可消费的 JSON 字典。

from __future__ import annotations

from packages.scholar_core.composition.composer import ResultComposer
from packages.scholar_core.pipeline import SearchPipeline


def search_response(pipeline: SearchPipeline, composer: ResultComposer, query: str, top_k: int) -> dict:
    response = pipeline.search(query, top_k=top_k)
    return composer.to_jsonable(response)
