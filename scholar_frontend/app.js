// 中文功能说明：独立前端交互逻辑，负责保存 API 地址、调用后端检索接口并渲染结果。

const DEFAULT_API_BASE = "http://127.0.0.1:8765";
const API_BASE_KEY = "scholarSearchApiBase";

const apiForm = document.querySelector("#api-form");
const apiBaseInput = document.querySelector("#api-base");
const apiStatus = document.querySelector("#api-status");
const form = document.querySelector("#search-form");
const query = document.querySelector("#query");
const topk = document.querySelector("#topk");
const button = document.querySelector("#submit");
const results = document.querySelector("#results");
const resultMeta = document.querySelector("#result-meta");
const trace = document.querySelector("#trace");
const traceMeta = document.querySelector("#trace-meta");
const raw = document.querySelector("#raw");

apiBaseInput.value = localStorage.getItem(API_BASE_KEY) || DEFAULT_API_BASE;
checkHealth();

apiForm.addEventListener("submit", (event) => {
  event.preventDefault();
  localStorage.setItem(API_BASE_KEY, cleanApiBase(apiBaseInput.value));
  apiBaseInput.value = cleanApiBase(apiBaseInput.value);
  checkHealth();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  resultMeta.textContent = "Searching";
  try {
    const params = new URLSearchParams({ q: query.value, top_k: topk.value });
    const response = await fetch(`${apiBase()}/api/search?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || "request failed");
    }
    render(data);
    setApiStatus("API connected", "ok");
  } catch (error) {
    results.className = "empty";
    results.textContent = error.message;
    resultMeta.textContent = "Failed";
    setApiStatus("API error", "error");
  } finally {
    button.disabled = false;
  }
});

async function checkHealth() {
  try {
    const response = await fetch(`${apiBase()}/health`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    setApiStatus("API connected", "ok");
  } catch {
    setApiStatus("API unavailable", "error");
  }
}

function render(data) {
  const papers = data.papers || [];
  const cost = data.cost || {};
  resultMeta.textContent = `${papers.length} papers · ${cost.latency_sec || 0}s`;
  traceMeta.textContent = `${cost.actions_executed || 0} actions`;
  results.className = "";
  results.innerHTML = `
    <table>
      <thead>
        <tr>
          <th class="rank">Rank</th>
          <th>Paper</th>
          <th class="year">Year</th>
          <th class="score">Score</th>
        </tr>
      </thead>
      <tbody>
        ${papers.map((paper) => `
          <tr>
            <td>${paper.rank}</td>
            <td>
              <div class="paper-title">${escapeHtml(paper.title)}</div>
              <div class="evidence">${escapeHtml((paper.evidence || [])[0] || "")}</div>
              <div class="evidence">${escapeHtml(paper.relevance)} · ${escapeHtml((paper.sources || []).join(", "))}</div>
            </td>
            <td>${paper.year || ""}</td>
            <td>${Number(paper.score || 0).toFixed(3)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  const parsed = data.parsed_query || {};
  const coverage = data.coverage || {};
  const modelServices = cost.model_services || {};
  const serviceErrors = modelServices.errors || [];
  trace.innerHTML = `
    <div class="kv"><span>Intent</span><strong>${escapeHtml(parsed.main_intent || "")}</strong></div>
    <div class="kv"><span>Fields</span><div class="chips">${chips(parsed.research_field)}</div></div>
    <div class="kv"><span>Constraints</span><div class="chips">${chips(parsed.must_have_constraints)}</div></div>
    <div class="kv"><span>Coverage</span><div class="chips">${chips(Object.entries(coverage.coverage || {}).map(([k, v]) => `${k}: ${v}`))}</div></div>
    <div class="kv"><span>Models</span><div class="chips">${chips(modelServices.enabled)}</div></div>
    <div class="kv"><span>Errors</span><div>${serviceErrors.length}</div></div>
    <div class="kv"><span>Stop</span><div>${escapeHtml(coverage.reason || "")}</div></div>
  `;
  raw.textContent = JSON.stringify(
    {
      cost,
      next_queries: coverage.next_queries || [],
      model_services: modelServices,
    },
    null,
    2,
  );
}

function apiBase() {
  return cleanApiBase(apiBaseInput.value || DEFAULT_API_BASE);
}

function cleanApiBase(value) {
  return String(value || DEFAULT_API_BASE).trim().replace(/\/+$/, "");
}

function setApiStatus(text, state) {
  apiStatus.textContent = text;
  apiStatus.className = `status ${state}`;
}

function chips(values) {
  return (values || []).map((value) => `<span class="chip">${escapeHtml(value)}</span>`).join("");
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#039;",
  }[char]));
}
