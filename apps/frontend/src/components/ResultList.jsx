// 中文功能说明：论文结果列表组件，显示检索到的论文排名、分数、来源和命中约束。

import { BookOpenCheck, ChevronRight, Loader2 } from "lucide-react";
import { relevanceLabel, sourceBadgeText } from "../utils/formatters.js";
import { EmptyState } from "./EmptyState.jsx";

export function ResultList({ papers, selectedPaperId, isLoading, onSelect }) {
  return (
    <section className="results-panel" aria-label="论文结果列表">
      <div className="panel-head">
        <div>
          <h3>检索结果</h3>
          <p>{isLoading ? "正在排序候选论文" : `${papers.length} 篇候选论文`}</p>
        </div>
        {isLoading ? <Loader2 className="spin" size={18} /> : <BookOpenCheck size={18} />}
      </div>

      {papers.length === 0 ? (
        <EmptyState icon={BookOpenCheck} title="暂无论文" description="检索完成后会在这里显示排序结果。" compact />
      ) : (
        <div className="paper-list">
          {papers.map((paper) => {
            const score = Number(paper.score || 0);
            return (
              <button
                key={`${paper.rank}-${paper.paper_id}`}
                type="button"
                className={`paper-row relevance-${paper.relevance || "unknown"} ${
                  paper.paper_id === selectedPaperId ? "selected" : ""
                }`}
                aria-pressed={paper.paper_id === selectedPaperId}
                onClick={() => onSelect(paper.paper_id)}
              >
                <span className="rank-badge">{paper.rank}</span>
                <span className="paper-row-main">
                  <strong>{paper.title || paper.paper_id}</strong>
                  <span className="paper-row-meta">
                    <span>{paper.year || "未知年份"}</span>
                    <span>{relevanceLabel(paper.relevance)}</span>
                    <span>{sourceBadgeText(paper.sources)}</span>
                  </span>
                  <span className="paper-row-evidence">{paper.evidence?.[0] || "暂无摘要片段"}</span>
                </span>
                <span className="score-pill">
                  <small>score</small>
                  {score.toFixed(3)}
                </span>
                <ChevronRight size={17} className="row-chevron" />
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
