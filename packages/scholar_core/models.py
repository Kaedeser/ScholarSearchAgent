# 中文功能说明：共享领域数据模型，定义论文、查询意图、搜索计划、候选结果和响应结构。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    source: str = "pasa"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryIntent:
    main_intent: str
    research_field: list[str]
    must_have_constraints: list[str]
    soft_constraints: list[str]
    excluded_meanings: list[str]
    time_range: tuple[int | None, int | None] | None
    venues: list[str]
    sub_queries: list[str]
    query_tokens: list[str]


@dataclass(frozen=True)
class SearchAction:
    source: str
    query: str
    top_k: int
    weight: float = 1.0
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchPlan:
    round: int
    search_actions: list[SearchAction]
    expand_citations_for: list[str]
    budget: dict[str, Any]


@dataclass
class Candidate:
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    canonical_id: str | None = None
    aliases: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    raw_scores: dict[str, float] = field(default_factory=dict)
    snippets: list[str] = field(default_factory=list)
    matched_constraints: list[str] = field(default_factory=list)
    missing_constraints: list[str] = field(default_factory=list)
    relevance: str = "unranked"
    final_score: float = 0.0
    first_seen_round: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageReport:
    coverage: dict[str, str]
    matched_constraints: list[str]
    missing_constraints: list[str]
    next_queries: list[str]
    should_continue: bool
    reason: str


@dataclass(frozen=True)
class SearchResponse:
    query: str
    parsed_query: QueryIntent
    plan: SearchPlan
    papers: list[Candidate]
    coverage: CoverageReport
    cost: dict[str, Any]
