// 中文功能说明：前端格式化工具，负责延迟、相关性、来源等展示文本转换。

export function formatLatency(value) {
  const numeric = Number(value || 0);
  if (!numeric) {
    return "0s";
  }
  return numeric < 1 ? `${Math.round(numeric * 1000)}ms` : `${numeric.toFixed(2)}s`;
}

export function relevanceLabel(value) {
  const labels = {
    highly_relevant: "高度相关",
    partially_relevant: "部分相关",
    weakly_relevant: "弱相关",
    low_relevance: "低相关",
  };
  return labels[value] || value || "-";
}

export function sourceBadgeText(values = []) {
  const items = Array.isArray(values) ? values : [];
  if (!items.length) {
    return "-";
  }
  const labels = items.map(sourceLabel);
  const visible = labels.slice(0, 3).join(" / ");
  return labels.length > 3 ? `${visible} +${labels.length - 3}` : visible;
}

export function topSourceText(papers = []) {
  const counts = new Map();
  papers.forEach((paper) => {
    (paper.sources || []).forEach((source) => counts.set(source, (counts.get(source) || 0) + 1));
  });
  const [source, count] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0] || [];
  return source ? `${sourceLabel(source)} (${count})` : "-";
}

function sourceLabel(value) {
  const labels = {
    elasticsearch: "ES",
    local_chunk_bm25: "Chunk BM25",
    local_title_bm25: "Title BM25",
    local_tfidf: "TF-IDF",
    qdrant_dense_paper: "Qdrant dense",
    qdrant_sparse_paper: "Qdrant sparse",
    neo4j_alias: "Neo4j alias",
    neo4j_concept: "Neo4j concept",
  };
  return labels[value] || String(value || "-").replaceAll("_", " ");
}
