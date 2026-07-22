# 中文功能说明：外部学术搜索 API 客户端，当前实现 Semantic Scholar 轻量论文搜索。

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class AcademicSearchResult:
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    url: str | None = None
    external_ids: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


class SemanticScholarClient:
    """Small wrapper around the Semantic Scholar Academic Graph paper search API."""

    DEFAULT_FIELDS = (
        "paperId,title,abstract,year,venue,citationCount,externalIds,url,openAccessPdf"
    )

    def __init__(
        self,
        *,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        api_key: str = "",
        timeout: float = 8.0,
        fields: str = DEFAULT_FIELDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = max(0.1, float(timeout))
        self.fields = fields

    def search(self, query: str, *, limit: int) -> list[AcademicSearchResult]:
        clean_query = " ".join(str(query or "").split())
        if not clean_query:
            return []
        params = urlencode(
            {
                "query": clean_query,
                "limit": max(1, min(int(limit), 100)),
                "fields": self.fields,
            }
        )
        request = Request(
            f"{self.base_url}/paper/search?{params}",
            headers=self._headers(),
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AcademicSearchError(str(exc)) from exc
        return [_result_from_row(row) for row in payload.get("data") or [] if isinstance(row, dict)]

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "ScholarSearchAgent/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers


class AcademicSearchError(RuntimeError):
    pass


def _result_from_row(row: dict[str, Any]) -> AcademicSearchResult:
    paper_id = str(row.get("paperId") or "")
    title = str(row.get("title") or paper_id)
    return AcademicSearchResult(
        paper_id=f"s2:{paper_id}" if paper_id and not paper_id.startswith("s2:") else paper_id,
        title=title,
        abstract=str(row.get("abstract") or ""),
        year=_safe_int(row.get("year")),
        venue=str(row.get("venue") or "") or None,
        citation_count=_safe_int(row.get("citationCount")),
        url=str(row.get("url") or "") or _open_access_url(row),
        external_ids=row.get("externalIds") if isinstance(row.get("externalIds"), dict) else {},
        raw=row,
    )


def _open_access_url(row: dict[str, Any]) -> str | None:
    open_access = row.get("openAccessPdf")
    if not isinstance(open_access, dict):
        return None
    url = str(open_access.get("url") or "")
    return url or None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
