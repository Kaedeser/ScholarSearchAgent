// 中文功能说明：论文详情组件，展示选中论文的标题、证据片段、约束命中、分数和模型元数据。

import { CheckCircle2, Copy, ExternalLink, FileText, XCircle } from "lucide-react";
import { relevanceLabel, sourceBadgeText } from "../utils/formatters.js";

export function PaperDetail({ paper }) {
  if (!paper) {
    return null;
  }

  const arxivId = String(paper.paper_id || "").replace(/^arxiv:/i, "");
  const arxivUrl = arxivId && arxivId !== paper.paper_id ? `https://arxiv.org/abs/${arxivId}` : "";
  const score = Number(paper.score || 0);
  const scorePercent = Math.max(0, Math.min(100, Math.round(score * 100)));

  return (
    <section className="paper-detail" aria-label="论文详情">
      <div className="paper-detail-head">
        <div className="paper-kicker">
          <FileText size={16} />
          <span>{paper.paper_id}</span>
        </div>
        <div className="detail-actions">
          {arxivUrl ? (
            <a className="icon-button ghost" href={arxivUrl} target="_blank" rel="noreferrer" title="打开 arXiv">
              <ExternalLink size={17} />
            </a>
          ) : null}
          <button
            className="icon-button ghost"
            type="button"
            title="复制论文 ID"
            onClick={() => navigator.clipboard?.writeText(paper.paper_id)}
          >
            <Copy size={17} />
          </button>
        </div>
      </div>

      <h3>{paper.title || paper.paper_id}</h3>

      <div className="detail-score" aria-label={`综合分数 ${score.toFixed(3)}`}>
        <div>
          <span>综合分数</span>
          <strong>{score.toFixed(3)}</strong>
        </div>
        <span className="score-track">
          <span style={{ width: `${scorePercent}%` }} />
        </span>
      </div>

      <div className="detail-meta-grid">
        <Meta label="年份" value={paper.year || "-"} />
        <Meta label="相关性" value={relevanceLabel(paper.relevance)} />
        <Meta label="分数" value={Number(paper.score || 0).toFixed(3)} />
        <Meta label="来源" value={sourceBadgeText(paper.sources)} />
      </div>

      <div className="evidence-block">
        <span>证据片段</span>
        {(paper.evidence || []).length ? (
          paper.evidence.map((item, index) => <p key={`${paper.paper_id}-evidence-${index}`}>{item}</p>)
        ) : (
          <p>暂无证据片段</p>
        )}
      </div>

      <div className="constraint-columns">
        <ConstraintList title="已命中" icon={CheckCircle2} values={paper.matched_constraints} state="good" />
        <ConstraintList title="未覆盖" icon={XCircle} values={paper.missing_constraints} state="bad" />
      </div>

      {paper.metadata?.crawler_strategy ? (
        <div className="model-note">
          <strong>爬取策略</strong>
          <span>{paper.metadata.crawler_strategy.prediction || "-"}</span>
        </div>
      ) : null}
    </section>
  );
}

function Meta({ label, value }) {
  return (
    <div className="detail-meta">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ConstraintList({ title, icon: Icon, values = [], state }) {
  return (
    <div className={`constraint-card ${state}`}>
      <div className="constraint-title">
        <Icon size={16} />
        <span>{title}</span>
      </div>
      <div className="constraint-values">
        {values.length ? values.map((item) => <span key={item}>{item}</span>) : <em>-</em>}
      </div>
    </div>
  );
}
