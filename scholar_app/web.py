from __future__ import annotations

import json
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cost_control_cache.pipeline import SearchPipeline
from result_composition.composer import ResultComposer


class DemoServer:
    def __init__(self, pipeline: SearchPipeline, *, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self.composer = ResultComposer()

    def serve_forever(self) -> None:
        pipeline = self.pipeline
        composer = self.composer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_html(_page_html())
                    return
                if parsed.path == "/api/search":
                    params = parse_qs(parsed.query)
                    query = (params.get("q") or [""])[0].strip()
                    top_k = _safe_int((params.get("top_k") or ["10"])[0], default=10)
                    if not query:
                        self._send_json({"error": "query is required"}, HTTPStatus.BAD_REQUEST)
                        return
                    response = pipeline.search(query, top_k=top_k)
                    self._send_json(composer.to_jsonable(response))
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args) -> None:
                return

            def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_html(self, html: str) -> None:
                payload = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"ScholarSearch demo server running at http://{self.host}:{self.port}")
        server.serve_forever()


def run_server(
    processed_dir: Path,
    *,
    host: str,
    port: int,
    paper_limit: int | None,
    chunk_limit: int | None,
    max_chunks_per_paper: int,
    per_query_top_k: int,
    backend: str = "auto",
) -> None:
    pipeline = SearchPipeline(
        processed_dir,
        paper_limit=paper_limit,
        chunk_limit=chunk_limit,
        max_chunks_per_paper=max_chunks_per_paper,
        per_query_top_k=per_query_top_k,
        backend=backend,
    )
    DemoServer(pipeline, host=host, port=port).serve_forever()


def _safe_int(value: str, *, default: int) -> int:
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _page_html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ScholarSearch Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d7dce2;
      --ink: #17202a;
      --muted: #5f6b7a;
      --accent: #0b6bcb;
      --accent-dark: #064f99;
      --soft: #eef5ff;
      --warn: #8a5a00;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    .shell {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }}
    .topbar {{
      min-height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    main {{
      padding: 20px 0 32px;
    }}
    form {{
      display: grid;
      grid-template-columns: 1fr 96px 112px;
      gap: 10px;
      align-items: stretch;
      margin-bottom: 18px;
    }}
    input, button {{
      height: 42px;
      border-radius: 6px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    input {{
      width: 100%;
      border: 1px solid var(--line);
      padding: 0 12px;
      background: var(--panel);
      color: var(--ink);
    }}
    button {{
      border: 1px solid var(--accent-dark);
      background: var(--accent);
      color: white;
      font-weight: 650;
      cursor: pointer;
    }}
    button:disabled {{
      opacity: 0.7;
      cursor: progress;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr);
      gap: 16px;
      align-items: start;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .section-head {{
      min-height: 44px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    h2 {{
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: #fbfcfd;
    }}
    .rank {{ width: 56px; }}
    .year {{ width: 72px; }}
    .score {{ width: 82px; }}
    .paper-title {{
      font-weight: 650;
      line-height: 1.35;
    }}
    .evidence {{
      margin-top: 6px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .side-body {{
      padding: 12px 14px;
      display: grid;
      gap: 14px;
      font-size: 13px;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 116px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
    }}
    .kv span:first-child {{
      color: var(--muted);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      display: inline-flex;
      min-height: 24px;
      align-items: center;
      border: 1px solid #c7dbef;
      background: var(--soft);
      border-radius: 999px;
      padding: 2px 8px;
      color: #154b78;
      line-height: 1.2;
    }}
    pre {{
      margin: 0;
      padding: 12px 14px;
      overflow: auto;
      background: #0f1720;
      color: #d9e6f2;
      font-size: 12px;
      line-height: 1.45;
    }}
    .empty {{
      padding: 24px 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 760px) {{
      .topbar {{
        align-items: flex-start;
        flex-direction: column;
        padding: 14px 0;
      }}
      form {{
        grid-template-columns: 1fr;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      table {{
        min-width: 720px;
      }}
      section:first-child {{
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="shell topbar">
      <h1>ScholarSearch Demo</h1>
      <div class="meta">Offline pipeline · PaSa processed data</div>
    </div>
  </header>
  <main class="shell">
    <form id="search-form">
      <input id="query" name="query" autocomplete="off" value="{escape('What works are related to the field of image retrieval?')}">
      <input id="topk" name="topk" type="number" min="1" max="50" value="10" aria-label="Top K">
      <button id="submit" type="submit">Search</button>
    </form>
    <div class="grid">
      <section>
        <div class="section-head">
          <h2>Results</h2>
          <span id="result-meta" class="meta">Ready</span>
        </div>
        <div id="results" class="empty">Run a query to inspect ranked papers.</div>
      </section>
      <section>
        <div class="section-head">
          <h2>Trace</h2>
          <span id="trace-meta" class="meta">0 calls</span>
        </div>
        <div id="trace" class="side-body"></div>
        <pre id="raw"></pre>
      </section>
    </div>
  </main>
  <script>
    const form = document.querySelector('#search-form');
    const query = document.querySelector('#query');
    const topk = document.querySelector('#topk');
    const button = document.querySelector('#submit');
    const results = document.querySelector('#results');
    const resultMeta = document.querySelector('#result-meta');
    const trace = document.querySelector('#trace');
    const traceMeta = document.querySelector('#trace-meta');
    const raw = document.querySelector('#raw');

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      button.disabled = true;
      resultMeta.textContent = 'Searching';
      try {{
        const params = new URLSearchParams({{ q: query.value, top_k: topk.value }});
        const response = await fetch(`/api/search?${{params.toString()}}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'request failed');
        render(data);
      }} catch (error) {{
        results.className = 'empty';
        results.textContent = error.message;
      }} finally {{
        button.disabled = false;
      }}
    }});

    function render(data) {{
      resultMeta.textContent = `${{data.papers.length}} papers · ${{data.cost.latency_sec}}s`;
      traceMeta.textContent = `${{data.cost.actions_executed}} actions`;
      results.className = '';
      results.innerHTML = `
        <table>
          <thead><tr>
            <th class="rank">Rank</th><th>Paper</th><th class="year">Year</th><th class="score">Score</th>
          </tr></thead>
          <tbody>
            ${{data.papers.map((paper) => `
              <tr>
                <td>${{paper.rank}}</td>
                <td>
                  <div class="paper-title">${{escapeHtml(paper.title)}}</div>
                  <div class="evidence">${{escapeHtml((paper.evidence || [])[0] || '')}}</div>
                  <div class="evidence">${{paper.relevance}} · ${{paper.sources.join(', ')}}</div>
                </td>
                <td>${{paper.year || ''}}</td>
                <td>${{Number(paper.score).toFixed(3)}}</td>
              </tr>
            `).join('')}}
          </tbody>
        </table>
      `;
      trace.innerHTML = `
        <div class="kv"><span>Intent</span><strong>${{escapeHtml(data.parsed_query.main_intent)}}</strong></div>
        <div class="kv"><span>Fields</span><div class="chips">${{chips(data.parsed_query.research_field)}}</div></div>
        <div class="kv"><span>Constraints</span><div class="chips">${{chips(data.parsed_query.must_have_constraints)}}</div></div>
        <div class="kv"><span>Coverage</span><div class="chips">${{chips(Object.entries(data.coverage.coverage).map(([k, v]) => `${{k}}: ${{v}}`))}}</div></div>
        <div class="kv"><span>Stop</span><div>${{escapeHtml(data.coverage.reason)}}</div></div>
      `;
      raw.textContent = JSON.stringify({{ cost: data.cost, next_queries: data.coverage.next_queries }}, null, 2);
    }}

    function chips(values) {{
      return (values || []).map((value) => `<span class="chip">${{escapeHtml(value)}}</span>`).join('');
    }}

    function escapeHtml(value) {{
      return String(value || '').replace(/[&<>"']/g, (char) => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }}[char]));
    }}
  </script>
</body>
</html>"""
