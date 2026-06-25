// 中文功能说明：覆盖分析组件，展示查询约束的覆盖状态和下一轮推荐查询。

import { CheckCircle2, CircleDashed, Route } from "lucide-react";

export function CoveragePanel({ coverage }) {
  const entries = Object.entries(coverage?.coverage || {});
  const nextQueries = coverage?.next_queries || [];

  return (
    <section className="coverage-panel" aria-label="覆盖分析">
      <div className="panel-head compact-head">
        <div>
          <h3>覆盖分析</h3>
          <p>{coverage?.reason || "暂无覆盖判断"}</p>
        </div>
        <Route size={18} />
      </div>
      <div className="coverage-body">
        {entries.length ? (
          <div className="coverage-list">
            {entries.map(([key, value]) => (
              <div className={`coverage-row ${value}`} key={key}>
                {value === "matched" ? <CheckCircle2 size={16} /> : <CircleDashed size={16} />}
                <span>{key}</span>
                <strong>{value === "matched" ? "已覆盖" : "待补全"}</strong>
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
