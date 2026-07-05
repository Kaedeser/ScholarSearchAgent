// 中文功能说明：覆盖分析组件，展示查询约束的覆盖状态和下一轮推荐查询。

import { CheckCircle2, CircleDashed, Route, XCircle } from "lucide-react";

export function CoveragePanel({ coverage }) {
  const entries = Object.entries(coverage?.coverage || {}).map(([key, value]) => [key, normalizeCoverageState(value)]);
  const nextQueries = coverage?.next_queries || [];
  const covered = entries.filter(([, value]) => value === "covered").length;
  const weak = entries.filter(([, value]) => value === "weak").length;
  const missing = entries.filter(([, value]) => value === "missing").length;
  const total = entries.length;
  const progress = total ? Math.round(((covered + weak * 0.5) / total) * 100) : 0;

  return (
    <section className="coverage-panel" aria-label="覆盖分析">
      <div className="panel-head compact-head">
        <div>
          <h3>覆盖分析</h3>
          <p>{reasonText(coverage?.reason)}</p>
        </div>
        <Route size={18} />
      </div>
      <div className="coverage-body">
        <div className="coverage-summary">
          <div className="coverage-meter" aria-label={`覆盖进度 ${progress}%`}>
            <span style={{ width: `${progress}%` }} />
          </div>
          <div className="coverage-stats">
            <span>
              <strong>{covered}</strong> 已覆盖
            </span>
            <span>
              <strong>{weak}</strong> 弱覆盖
            </span>
            <span>
              <strong>{missing}</strong> 待补全
            </span>
          </div>
        </div>

        {entries.length ? (
          <div className="coverage-list">
            {entries.map(([key, value]) => (
              <div className={`coverage-row ${value}`} key={key}>
                {coverageIcon(value)}
                <span>{key}</span>
                <strong>{coverageLabel(value)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted-line">暂无约束覆盖信息</p>
        )}

        {nextQueries.length ? (
          <div className="next-query-box">
            <span>下一轮查询</span>
            {nextQueries.map((item) => (
              <code key={item}>{item}</code>
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function normalizeCoverageState(value) {
  if (value === "covered" || value === "matched") {
    return "covered";
  }
  if (value === "weak") {
    return "weak";
  }
  return "missing";
}

function coverageLabel(value) {
  if (value === "covered") {
    return "已覆盖";
  }
  if (value === "weak") {
    return "弱覆盖";
  }
  return "待补全";
}

function coverageIcon(value) {
  if (value === "covered") {
    return <CheckCircle2 size={16} />;
  }
  if (value === "weak") {
    return <CircleDashed size={16} />;
  }
  return <XCircle size={16} />;
}

function reasonText(reason) {
  const labels = {
    "top results still miss required query constraints": "当前结果仍有关键约束未覆盖",
    "enough high relevance candidates found for demo budget": "已找到足够高相关候选",
    "no useful next query generated": "暂无可用的下一轮补充查询",
  };
  return labels[reason] || reason || "暂无覆盖判断";
}
