// 中文功能说明：检索表单组件，负责输入论文检索问题、TopK 参数和示例查询选择。

import { Loader2, Search, SlidersHorizontal } from "lucide-react";

export function SearchControls({
  query,
  topK,
  isSearching,
  sampleQueries,
  onQueryChange,
  onTopKChange,
  onSubmit,
  onSampleSelect,
}) {
  return (
    <form className="search-console" onSubmit={onSubmit}>
      <label className="sr-only" htmlFor="query-input">
        检索问题
      </label>
      <div className="query-input-wrap">
        <Search size={20} aria-hidden="true" />
        <textarea
          id="query-input"
          value={query}
          rows={2}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="输入研究问题或论文主题"
        />
      </div>

      <div className="control-row">
        <div className="stepper">
          <SlidersHorizontal size={17} aria-hidden="true" />
          <label htmlFor="top-k">Top K</label>
          <input
            id="top-k"
            type="number"
            min="1"
            max="50"
            value={topK}
            onChange={(event) => onTopKChange(Number(event.target.value) || 1)}
          />
        </div>
        <button className="primary-action" type="submit" disabled={isSearching}>
          {isSearching ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
          <span>{isSearching ? "检索中" : "开始检索"}</span>
        </button>
      </div>

      <div className="sample-row" aria-label="示例查询">
        {sampleQueries.map((sample) => (
          <button key={sample} type="button" className="sample-chip" onClick={() => onSampleSelect(sample)}>
            {sample}
          </button>
        ))}
      </div>
    </form>
  );
}
