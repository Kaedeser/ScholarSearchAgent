// 中文功能说明：右侧洞察栏通用组件，用于展示解析意图、模型状态和原始响应。

export function InsightRail({ icon: Icon, title, rows, code }) {
  return (
    <section className="insight-card">
      <div className="rail-title">
        <Icon size={17} />
        <h3>{title}</h3>
      </div>
      {rows.length ? (
        <dl className="insight-rows">
          {rows.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {code ? <pre className="raw-json">{code}</pre> : null}
    </section>
  );
}
