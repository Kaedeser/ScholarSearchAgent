// 中文功能说明：检索执行轨迹组件，展示搜索动作、预算和成本摘要。

import { Activity, ListFilter } from "lucide-react";
import { formatLatency } from "../utils/formatters.js";

export function TracePanel({ plan, cost }) {
  const actions = plan?.search_actions || [];
  const budget = plan?.budget || {};

  return (
    <section className="trace-panel" aria-label="执行轨迹">
      <div className="rail-title">
        <Activity size={17} />
        <h3>执行轨迹</h3>
      </div>
      <div className="trace-stats">
        <span>
          <strong>{cost?.actions_executed || 0}</strong>
          动作
        </span>
        <span>
          <strong>{cost?.raw_candidates || 0}</strong>
          原始候选
        </span>
        <span>
          <strong>{formatLatency(cost?.latency_sec)}</strong>
          延迟
        </span>
      </div>
      {actions.length ? (
        <ol className="action-list">
          {actions.map((action, index) => (
            <li key={`${action.source}-${index}`}>
              <ListFilter size={15} />
              <div>
                <strong>{action.source}</strong>
                <span>{action.query}</span>
              </div>
              <em>{action.top_k}</em>
            </li>
          ))}
        </ol>
      ) : (
        <p className="muted-line">暂无动作</p>
      )}
      <div className="budget-grid">
        {Object.entries(budget).map(([key, value]) => (
          <span key={key}>
            <small>{key}</small>
            <strong>{String(value)}</strong>
          </span>
        ))}
      </div>
    </section>
  );
}
