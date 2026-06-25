// 中文功能说明：空状态组件，用于检索前和无结果时的占位展示。

export function EmptyState({ icon: Icon, title, description, compact = false }) {
  return (
    <div className={`empty-state ${compact ? "compact" : ""}`}>
      <Icon size={compact ? 22 : 32} />
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  );
}
