# 中文功能说明：候选论文归一与去重模块，合并 arXiv/title 别名、来源、分数和章节元数据。

from __future__ import annotations

import hashlib
import re

from packages.scholar_core.models import Candidate
from packages.scholar_core.text import normalize_title


ARXIV_RE = re.compile(r"^arxiv:(.+)$", re.IGNORECASE)


def canonical_id(candidate: Candidate) -> str:
    paper_id = candidate.paper_id.strip()
    match = ARXIV_RE.match(paper_id)
    if match:
        return f"arxiv:{match.group(1).lower()}"
    normalized_title = normalize_title(candidate.title)
    digest = hashlib.sha1(normalized_title.encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


class CandidateNormalizer:
    def merge(self, candidates: list[Candidate]) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        for candidate in candidates:
            key = canonical_id(candidate)
            candidate.canonical_id = key
            candidate.aliases.add(candidate.paper_id)
            existing = merged.get(key)
            if existing is None:
                merged[key] = candidate
                continue
            existing.aliases.update(candidate.aliases)
            existing.sources.update(candidate.sources)
            existing.snippets.extend(snippet for snippet in candidate.snippets if snippet not in existing.snippets)
            for score_name, value in candidate.raw_scores.items():
                existing.raw_scores[score_name] = max(existing.raw_scores.get(score_name, 0.0), value)
            _merge_metadata(existing.metadata, candidate.metadata)
            if not existing.abstract and candidate.abstract:
                existing.abstract = candidate.abstract
            if existing.year is None and candidate.year is not None:
                existing.year = candidate.year
            if existing.citation_count is None and candidate.citation_count is not None:
                existing.citation_count = candidate.citation_count
        return list(merged.values())


def _merge_metadata(target: dict, source: dict) -> None:
    for key, value in source.items():
        if key == "section_title" and value:
            section_titles = target.setdefault("section_titles", [])
            if value not in section_titles:
                section_titles.append(value)
            continue
        if key == "section_titles" and isinstance(value, list):
            section_titles = target.setdefault("section_titles", [])
            for section_title in value:
                if section_title and section_title not in section_titles:
                    section_titles.append(section_title)
            continue
        target.setdefault(key, value)
