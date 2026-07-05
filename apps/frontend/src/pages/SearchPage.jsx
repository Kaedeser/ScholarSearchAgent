// 中文功能说明：论文检索页面，管理 API 地址、健康状态、检索状态和检索结果工作台布局。

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BookOpenCheck,
  Braces,
  Clock3,
  Database,
  FileSearch,
  Gauge,
  Layers3,
  Network,
  RefreshCw,
  Save,
  Search,
  Server,
  Sparkles,
} from "lucide-react";
import { cleanApiBase, loadApiBase, saveApiBase } from "../api/client.js";
import { fetchHealth, searchPapers } from "../api/searchApi.js";
import { ApiStatus } from "../components/ApiStatus.jsx";
import { CoveragePanel } from "../components/CoveragePanel.jsx";
import { EmptyState } from "../components/EmptyState.jsx";
import { InsightRail } from "../components/InsightRail.jsx";
import { PaperDetail } from "../components/PaperDetail.jsx";
import { ResultList } from "../components/ResultList.jsx";
import { SearchControls } from "../components/SearchControls.jsx";
import { TracePanel } from "../components/TracePanel.jsx";
import { formatLatency, topSourceText } from "../utils/formatters.js";

const SAMPLE_QUERIES = [
  "image retrieval representation learning",
  "large language model retrieval augmented generation survey",
  "graph neural network paper recommendation",
  "semantic segmentation active learning after 2020",
];

export function SearchPage() {
  const [apiBase, setApiBase] = useState(loadApiBase);
  const [draftApiBase, setDraftApiBase] = useState(loadApiBase);
  const [health, setHealth] = useState({ state: "checking", message: "连接中" });
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [topK, setTopK] = useState(10);
  const [response, setResponse] = useState(null);
  const [selectedPaperId, setSelectedPaperId] = useState("");
  const [error, setError] = useState("");
  const [isSearching, setIsSearching] = useState(false);

  const selectedPaper = useMemo(() => {
    const papers = response?.papers || [];
    return papers.find((paper) => paper.paper_id === selectedPaperId) || papers[0] || null;
  }, [response, selectedPaperId]);

  const metrics = useMemo(() => {
    const papers = response?.papers || [];
    const cost = response?.cost || {};
    const coverage = response?.coverage?.coverage || {};
    const coverageValues = Object.values(coverage);
    const covered = coverageValues.filter((value) => ["covered", "matched"].includes(value)).length;
    const weak = coverageValues.filter((value) => value === "weak").length;
    const total = coverageValues.length;
    return [
      { label: "论文", value: papers.length || 0, hint: "候选结果", icon: BookOpenCheck, tone: "papers" },
      { label: "延迟", value: formatLatency(cost.latency_sec), hint: "端到端", icon: Clock3, tone: "latency" },
      {
        label: "覆盖",
        value: total ? `${covered + weak}/${total}` : "0/0",
        hint: weak ? `${covered} 强 / ${weak} 弱` : "约束命中",
        icon: Gauge,
        tone: "coverage",
      },
      { label: "后端", value: cost.backend || "-", hint: "检索源", icon: Database, tone: "backend" },
    ];
  }, [response]);

  useEffect(() => {
    checkHealth(apiBase);
  }, [apiBase]);

  async function checkHealth(base = apiBase) {
    setHealth({ state: "checking", message: "连接中" });
    try {
      const data = await fetchHealth(base);
      setHealth({ state: "ok", message: data.service || "scholar-search-api", data });
    } catch (err) {
      setHealth({ state: "error", message: err.message || "后端不可用" });
    }
  }

  function handleSaveApiBase(event) {
    event.preventDefault();
    const nextBase = saveApiBase(draftApiBase);
    setDraftApiBase(nextBase);
    setApiBase(nextBase);
  }

  async function handleSearch(event) {
    event?.preventDefault();
    const text = query.trim();
    if (!text) {
      setError("请输入检索问题");
      return;
    }
    setIsSearching(true);
    setError("");
    try {
      const data = await searchPapers(apiBase, text, topK);
      setResponse(data);
      setSelectedPaperId(data.papers?.[0]?.paper_id || "");
      setHealth((current) => (current.state === "ok" ? current : { state: "ok", message: "scholar-search-api" }));
    } catch (err) {
      setError(err.message || "检索失败");
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div className="brand-block">
            <div className="brand-mark" aria-hidden="true">
              <FileSearch size={22} />
            </div>
            <div>
              <h1>ScholarSearch</h1>
              <p>论文检索工作台</p>
            </div>
          </div>

          <form className="api-config" onSubmit={handleSaveApiBase}>
            <Server size={18} aria-hidden="true" />
            <label className="sr-only" htmlFor="api-base">
              API 地址
            </label>
            <input
              id="api-base"
              value={draftApiBase}
              onChange={(event) => setDraftApiBase(cleanApiBase(event.target.value))}
              spellCheck="false"
            />
            <button className="icon-button" type="submit" title="保存 API 地址">
              <Save size={17} />
            </button>
            <button className="icon-button ghost" type="button" onClick={() => checkHealth()} title="刷新后端状态">
              <RefreshCw size={17} />
            </button>
          </form>

          <ApiStatus state={health.state} message={health.message} />
        </header>

        <section className="query-zone">
          <div className="query-copy">
            <span className="eyebrow">
              <Sparkles size={15} />
              Scholar Agent
            </span>
            <h2>从问题到论文证据链</h2>
            <p className="query-note">把自然语言研究问题拆成检索计划、约束覆盖和可追溯证据。</p>
            <div className="query-highlights" aria-label="工作流">
              <span>意图解析</span>
              <span>多源召回</span>
              <span>证据排序</span>
            </div>
          </div>
          <SearchControls
            query={query}
            topK={topK}
            isSearching={isSearching}
            sampleQueries={SAMPLE_QUERIES}
            onQueryChange={setQuery}
            onTopKChange={setTopK}
            onSubmit={handleSearch}
            onSampleSelect={setQuery}
          />
        </section>

        {error ? (
          <div className="error-banner" role="alert">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        <section className="metric-strip" aria-label="检索指标">
          {metrics.map((item) => (
            <MetricTile item={item} key={item.label} />
          ))}
        </section>

        <section className="content-grid">
          <ResultList
            papers={response?.papers || []}
            selectedPaperId={selectedPaper?.paper_id || ""}
            isLoading={isSearching}
            onSelect={setSelectedPaperId}
          />

          <div className="detail-column">
            {response ? (
              <>
                <PaperDetail paper={selectedPaper} />
                <CoveragePanel coverage={response.coverage} />
              </>
            ) : (
              <EmptyState
                icon={Search}
                title="等待检索"
                description="输入研究问题后，系统会返回候选论文、约束覆盖、模型调用和检索成本。"
              />
            )}
          </div>

          <aside className="right-rail">
            <InsightRail
              icon={Layers3}
              title="解析意图"
              rows={[
                ["主意图", response?.parsed_query?.main_intent || "-"],
                ["领域", (response?.parsed_query?.research_field || []).join(" / ") || "-"],
                ["候选来源", topSourceText(response?.papers || [])],
              ]}
            />
            <InsightRail
              icon={Network}
              title="模型服务"
              rows={[
                ["启用", (response?.cost?.model_services?.enabled || []).join(" / ") || "-"],
                ["错误", String(response?.cost?.model_services?.errors?.length || 0)],
                ["策略", response?.cost?.model_services?.crawler_strategy ? "已检查" : "-"],
              ]}
            />
            <TracePanel plan={response?.plan} cost={response?.cost} />
            <InsightRail
              icon={Braces}
              title="原始响应"
              rows={[]}
              code={response ? JSON.stringify(response, null, 2) : "{ }"}
            />
          </aside>
        </section>
      </section>
    </main>
  );
}

function MetricTile({ item }) {
  const Icon = item.icon;
  return (
    <div className={`metric-tile metric-${item.tone || "neutral"}`}>
      <span className="metric-icon">
        <Icon size={18} />
      </span>
      <span className="metric-copy">
        <span>{item.label}</span>
        <small>{item.hint}</small>
      </span>
      <strong>{item.value}</strong>
    </div>
  );
}
