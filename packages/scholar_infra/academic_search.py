# 中文功能说明：外部学术搜索 API 客户端，当前实现 Semantic Scholar 轻量论文搜索。

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
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
    corpus_id: str | None = None
    influential_citation_count: int | None = None
    reference_count: int | None = None
    fields_of_study: tuple[str, ...] = ()
    publication_types: tuple[str, ...] = ()
    publication_date: str | None = None
    authors: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class AcademicSnippetResult:
    paper_id: str
    title: str
    text: str
    score: float
    corpus_id: str | None = None
    section: str | None = None
    snippet_kind: str | None = None
    authors: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None


class SemanticScholarClient:
    """Small wrapper around the Semantic Scholar Academic Graph paper search API."""

    DEFAULT_FIELDS = (
        "paperId,corpusId,title,abstract,year,venue,citationCount,influentialCitationCount,"
        "referenceCount,externalIds,url,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,"
        "publicationTypes,publicationDate,authors"
    )
    SNIPPET_FIELDS = "snippet.text,snippet.snippetKind,snippet.section"

    def __init__(
        self,
        *,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        api_key: str = "",
        timeout: float = 8.0,
        fields: str = DEFAULT_FIELDS,
        max_retries: int = 2,
        retry_backoff_sec: float = 1.0,
        min_interval_sec: float = 0.0,
        cache_size: int = 256,
        cache_path: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = max(0.1, float(timeout))
        self.fields = fields
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.cache_size = max(0, int(cache_size))
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._state_lock = RLock()
        self._throttle_lock = Lock()
        self._last_request_at = 0.0
        self.request_count = 0
        self.logical_request_count = 0
        self.retry_count = 0
        self.cache_hits = 0
        self.error_count = 0
        self.last_error: str | None = None
        self._load_cache()

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[AcademicSearchResult]:
        clean_query = _normalize_search_query(query)
        if not clean_query:
            return []
        params: dict[str, Any] = {
            "query": clean_query,
            "limit": max(1, min(int(limit), 100)),
            "fields": self.fields,
        }
        params.update(_clean_filters(filters))
        payload = self._get_json("/paper/search", params)
        return [_result_from_row(row) for row in payload.get("data") or [] if isinstance(row, dict)]

    def search_snippets(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[AcademicSnippetResult]:
        clean_query = _normalize_search_query(query)
        if not clean_query:
            return []
        params: dict[str, Any] = {
            "query": clean_query,
            "limit": max(1, min(int(limit), 1000)),
            "fields": self.SNIPPET_FIELDS,
        }
        params.update(_clean_filters(filters))
        payload = self._get_json("/snippet/search", params)
        return [_snippet_from_row(row) for row in payload.get("data") or [] if isinstance(row, dict)]

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "academic_search_api_calls": self.request_count,
                "academic_search_cache_hits": self.cache_hits,
                "academic_search_errors": self.error_count,
                "academic_search_last_error": self.last_error,
                "requests": self.logical_request_count,
                "retries": self.retry_count,
                "cache_hits": self.cache_hits,
                "cache_entries": len(self._cache),
            }

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}?{urlencode(params)}"
        cached = self._cache_get(url)
        if cached is not None:
            with self._state_lock:
                self.cache_hits += 1
            return cached

        with self._state_lock:
            self.logical_request_count += 1
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            request = Request(url, headers=self._headers(), method="GET")
            with self._state_lock:
                self.request_count += 1
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise json.JSONDecodeError("expected a JSON object", "", 0)
                with self._state_lock:
                    self.last_error = None
                self._cache_put(url, payload)
                return payload
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_exc = exc
                with self._state_lock:
                    self.error_count += 1
                    self.last_error = str(exc)
                if attempt >= self.max_retries or not _retryable(exc):
                    break
                with self._state_lock:
                    self.retry_count += 1
                time.sleep(_retry_delay(exc, attempt, self.retry_backoff_sec))
        raise AcademicSearchError(str(last_exc or "Semantic Scholar request failed")) from last_exc

    def _throttle(self) -> None:
        if self.min_interval_sec <= 0:
            return
        with self._throttle_lock:
            now = time.monotonic()
            remaining = self.min_interval_sec - (now - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._state_lock:
            if self.cache_size <= 0 or key not in self._cache:
                return None
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

    def _cache_put(self, key: str, value: dict[str, Any]) -> None:
        if self.cache_size <= 0:
            return
        with self._state_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            self._persist_cache()

    def _load_cache(self) -> None:
        if self.cache_path is None or self.cache_size <= 0 or not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key, value in list(payload.items())[-self.cache_size :]:
                    if isinstance(value, dict):
                        self._cache[str(key)] = value
        except (OSError, json.JSONDecodeError):
            return

    def _persist_cache(self) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
            temporary.write_text(
                json.dumps(self._cache, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(self.cache_path)
        except OSError:
            return

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
        corpus_id=str(row.get("corpusId") or "") or None,
        influential_citation_count=_safe_int(row.get("influentialCitationCount")),
        reference_count=_safe_int(row.get("referenceCount")),
        fields_of_study=_fields_of_study(row),
        publication_types=tuple(str(value) for value in row.get("publicationTypes") or [] if value),
        publication_date=str(row.get("publicationDate") or "") or None,
        authors=tuple(
            str(author.get("name") or "")
            for author in row.get("authors") or []
            if isinstance(author, dict) and author.get("name")
        ),
        raw=row,
    )


def _snippet_from_row(row: dict[str, Any]) -> AcademicSnippetResult:
    raw_paper = row.get("paper")
    raw_snippet = row.get("snippet")
    paper: dict[str, Any] = raw_paper if isinstance(raw_paper, dict) else {}
    snippet: dict[str, Any] = raw_snippet if isinstance(raw_snippet, dict) else {}
    corpus_id = str(paper.get("corpusId") or "") or None
    raw_paper_id = str(paper.get("paperId") or "")
    if raw_paper_id:
        paper_id = raw_paper_id if raw_paper_id.startswith("s2:") else f"s2:{raw_paper_id}"
    elif corpus_id:
        paper_id = f"s2-corpus:{corpus_id}"
    else:
        paper_id = ""
    return AcademicSnippetResult(
        paper_id=paper_id,
        title=str(paper.get("title") or paper_id),
        text=str(snippet.get("text") or ""),
        score=_safe_float(row.get("score")),
        corpus_id=corpus_id,
        section=str(snippet.get("section") or "") or None,
        snippet_kind=str(snippet.get("snippetKind") or "") or None,
        authors=tuple(_flatten_authors(paper.get("authors"))),
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


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_search_query(value: Any) -> str:
    clean = " ".join(str(value or "").split())
    return re.sub(r"(?<=\w)-(?=\w)", " ", clean)


def _clean_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    allowed = {
        "publicationDateOrYear",
        "year",
        "venue",
        "fieldsOfStudy",
        "publicationTypes",
        "minCitationCount",
        "openAccessPdf",
    }
    return {
        key: value
        for key, value in (filters or {}).items()
        if key in allowed and value is not None and value != "" and value != []
    }


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (URLError, TimeoutError, json.JSONDecodeError))


def _retry_delay(exc: Exception, attempt: int, backoff: float) -> float:
    if isinstance(exc, HTTPError) and exc.headers is not None:
        try:
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass
    return backoff * (2**attempt)


def _fields_of_study(row: dict[str, Any]) -> tuple[str, ...]:
    values = [str(value) for value in row.get("fieldsOfStudy") or [] if value]
    for item in row.get("s2FieldsOfStudy") or []:
        if isinstance(item, dict) and item.get("category"):
            values.append(str(item["category"]))
    return tuple(dict.fromkeys(values))


def _flatten_authors(value: Any) -> list[str]:
    result: list[str] = []
    for item in value or []:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and item.get("name"):
            result.append(str(item["name"]))
        elif isinstance(item, list):
            result.extend(str(name) for name in item if name)
    return result
