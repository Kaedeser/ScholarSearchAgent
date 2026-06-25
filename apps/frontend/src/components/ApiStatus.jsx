// 中文功能说明：后端状态组件，根据健康检查结果显示 API 连接状态。

import { CheckCircle2, Loader2, XCircle } from "lucide-react";

export function ApiStatus({ state, message }) {
  const Icon = state === "ok" ? CheckCircle2 : state === "checking" ? Loader2 : XCircle;
  return (
    <div className={`api-status ${state}`} aria-live="polite">
      <Icon size={17} className={state === "checking" ? "spin" : ""} />
      <span>{message}</span>
    </div>
  );
}
