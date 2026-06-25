// 中文功能说明：论文检索 API 封装，暴露健康检查和论文检索两个后端调用。

import { requestJson } from "./client.js";

export function fetchHealth(apiBase) {
  return requestJson(apiBase, "/health");
}

export function searchPapers(apiBase, query, topK) {
  const params = new URLSearchParams({ q: query, top_k: String(topK) });
  return requestJson(apiBase, `/api/search?${params.toString()}`);
}
