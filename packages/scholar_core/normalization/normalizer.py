# 中文功能说明：候选论文归一与去重模块，合并 arXiv/title 别名、来源、分数和章节元数据。

from __future__ import annotations

import hashlib
import re

from packages.scholar_core.models import Candidate
from packages.scholar_core.text import normalize_title


ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
STABLE_ID_PREFIXES = ("arxiv:", "doi:", "s2:", "s2-corpus:", "pmid:", "acl:")


def canonical_id(candidate: Candidate) -> str:
    stable_ids = _stable_ids(candidate)
    for prefix in STABLE_ID_PREFIXES:
        match = next((value for value in stable_ids if value.startswith(prefix)), None)
        if match:
            return match
    return _title_id(candidate.title)


class CandidateNormalizer:
    def merge(self, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        parents = list(range(len(candidates)))
        identities: dict[str, int] = {}

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for index, candidate in enumerate(candidates):
            for identity in _identity_keys(candidate):
                previous = identities.get(identity)
                if previous is None:
                    identities[identity] = index
                else:
                    union(previous, index)

        groups: dict[int, list[Candidate]] = {}
        for index, candidate in enumerate(candidates):
            groups.setdefault(find(index), []).append(candidate)

        results: list[Candidate] = []
        for group in groups.values():
            existing = group[0]
            all_stable_ids: set[str] = set()
            for candidate in group:
                candidate.aliases.add(candidate.paper_id)
                normalized_aliases = _stable_ids(candidate)
                candidate.aliases.update(normalized_aliases)
                all_stable_ids.update(normalized_aliases)
                if candidate is not existing:
                    _merge_candidate(existing, candidate)
            existing.aliases.update(all_stable_ids)
            existing.canonical_id = _preferred_group_id(all_stable_ids, existing.title)
            results.append(existing)
        return results


def _merge_candidate(existing: Candidate, candidate: Candidate) -> None:
    existing.aliases.update(candidate.aliases)
    existing.sources.update(candidate.sources)
    existing.snippets.extend(snippet for snippet in candidate.snippets if snippet not in existing.snippets)
    for score_name, value in candidate.raw_scores.items():
        existing.raw_scores[score_name] = max(existing.raw_scores.get(score_name, 0.0), value)
    _merge_metadata(existing.metadata, candidate.metadata)
    if len(candidate.abstract or "") > len(existing.abstract or ""):
        existing.abstract = candidate.abstract
    if existing.year is None and candidate.year is not None:
        existing.year = candidate.year
    if not existing.venue and candidate.venue:
        existing.venue = candidate.venue
    if (candidate.citation_count or 0) > (existing.citation_count or 0):
        existing.citation_count = candidate.citation_count


def _identity_keys(candidate: Candidate) -> set[str]:
    keys = _stable_ids(candidate)
    if candidate.title.strip():
        keys.add(_title_id(candidate.title))
    return keys


def _stable_ids(candidate: Candidate) -> set[str]:
    values = {candidate.paper_id, *candidate.aliases}
    external_ids = candidate.metadata.get("external_ids") or {}
    if isinstance(external_ids, dict):
        for key, value in external_ids.items():
            if value not in (None, ""):
                values.add(f"{key}:{value}")
    return {normalized for value in values if (normalized := _normalize_stable_id(value))}


def _normalize_stable_id(value: object) -> str:
    clean = str(value or "").strip()
    lowered = clean.lower()
    if lowered.startswith("https://doi.org/"):
        lowered = f"doi:{lowered.removeprefix('https://doi.org/')}"
    if lowered.startswith("http://doi.org/"):
        lowered = f"doi:{lowered.removeprefix('http://doi.org/')}"
    if lowered.startswith("arxiv:"):
        arxiv_id = ARXIV_VERSION_RE.sub("", lowered.removeprefix("arxiv:"))
        return f"arxiv:{arxiv_id}"
    if lowered.startswith("corpusid:"):
        return f"s2-corpus:{lowered.removeprefix('corpusid:')}"
    if lowered.startswith("paperid:"):
        return f"s2:{lowered.removeprefix('paperid:')}"
    if lowered.startswith(STABLE_ID_PREFIXES):
        return lowered
    return ""


def _preferred_group_id(stable_ids: set[str], title: str) -> str:
    for prefix in STABLE_ID_PREFIXES:
        match = next((value for value in sorted(stable_ids) if value.startswith(prefix)), None)
        if match:
            return match
    return _title_id(title)


def _title_id(title: str) -> str:
    normalized_title = normalize_title(title)
    digest = hashlib.sha1(normalized_title.encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


def _merge_metadata(target: dict, source: dict) -> None:
    for key, value in source.items():
        if key in {"source_ranks", "source_weights"} and isinstance(value, dict):
            merged = target.setdefault(key, {})
            if isinstance(merged, dict):
                for source_key, source_value in value.items():
                    if key == "source_ranks":
                        try:
                            existing = int(merged.get(source_key, source_value))
                            incoming = int(source_value)
                            merged[source_key] = min(existing, incoming)
                        except (TypeError, ValueError):
                            merged.setdefault(source_key, source_value)
                    else:
                        merged[source_key] = max(float(merged.get(source_key, 0.0) or 0.0), float(source_value or 0.0))
            continue
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
