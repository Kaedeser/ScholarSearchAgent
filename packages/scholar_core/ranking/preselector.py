# Rule-based candidate preselection before selector reranking.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.scholar_core.models import Candidate, QueryIntent
from packages.scholar_core.planning.planner import query_profile_kind
from packages.scholar_core.text import tokenize


@dataclass(frozen=True)
class CandidatePreselection:
    candidates: list[Candidate]
    metadata: dict[str, Any]


class CandidatePreselector:
    def select(
        self,
        candidates: list[Candidate],
        intent: QueryIntent,
        *,
        top_k: int,
        pool_limit: int,
    ) -> CandidatePreselection:
        pool = candidates[: max(1, pool_limit)]
        target = min(max(1, top_k), len(pool))
        profile = query_profile_kind(intent)
        original_rank = {id(candidate): index for index, candidate in enumerate(pool, start=1)}
        scored = [
            (_preselect_score(candidate, profile, original_rank[id(candidate)]), candidate)
            for candidate in pool
        ]
        for score, candidate in scored:
            candidate.metadata["selector_preselect_score"] = round(score, 6)

        if len(pool) <= target:
            for rank, candidate in enumerate(pool, start=1):
                candidate.metadata["selector_preselect_rank"] = rank
                candidate.metadata["selector_preselect_reason"] = "pool<=target"
            return CandidatePreselection(
                candidates=pool,
                metadata={
                    "enabled": True,
                    "profile": profile,
                    "input_candidates": len(pool),
                    "selected_candidates": len(pool),
                    "target_candidates": target,
                    "pool_limit": pool_limit,
                    "reason_counts": {"pool<=target": len(pool)},
                },
            )

        selected: list[Candidate] = []
        selected_ids: set[str] = set()
        reason_counts: dict[str, int] = {}

        def add(candidate: Candidate, reason: str) -> bool:
            paper_id = candidate.canonical_id or candidate.paper_id
            if not paper_id or paper_id in selected_ids:
                return False
            selected_ids.add(paper_id)
            candidate.metadata["selector_preselect_reason"] = reason
            selected.append(candidate)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            return True

        head_keep = _protected_head_count(profile, target)
        for candidate in pool[:head_keep]:
            if len(selected) >= target:
                break
            add(candidate, "protected_rule_head")

        lanes = [
            ("constraint_coverage", _constraint_lane, _lane_quota(profile, target, "constraint")),
            ("multi_source_support", _multi_source_lane, _lane_quota(profile, target, "multi_source")),
            ("title_anchor", _title_lane, _lane_quota(profile, target, "title")),
            ("dense_lexical_bridge", _dense_lane, _lane_quota(profile, target, "dense")),
            ("sparse_lexical_bridge", _sparse_lane, _lane_quota(profile, target, "sparse")),
            ("academic_evidence", _academic_lane, _lane_quota(profile, target, "academic")),
            ("alias_or_graph", _alias_graph_lane, _lane_quota(profile, target, "alias")),
        ]
        scored_desc = sorted(scored, key=lambda item: item[0], reverse=True)
        for reason, predicate, quota in lanes:
            added = 0
            for _score, candidate in scored_desc:
                if len(selected) >= target or added >= quota:
                    break
                if predicate(candidate) and add(candidate, reason):
                    added += 1

        deferred: list[Candidate] = []
        for _score, candidate in scored_desc:
            if len(selected) >= target:
                break
            if _too_redundant(candidate, selected) and not _high_confidence_candidate(candidate):
                deferred.append(candidate)
                continue
            add(candidate, "score_fill")

        for candidate in deferred:
            if len(selected) >= target:
                break
            add(candidate, "redundant_score_fill")

        for candidate in pool:
            if len(selected) >= target:
                break
            add(candidate, "original_order_fill")

        for rank, candidate in enumerate(selected, start=1):
            candidate.metadata["selector_preselect_rank"] = rank

        return CandidatePreselection(
            candidates=selected,
            metadata={
                "enabled": True,
                "profile": profile,
                "input_candidates": len(pool),
                "selected_candidates": len(selected),
                "target_candidates": target,
                "pool_limit": pool_limit,
                "protected_head": head_keep,
                "reason_counts": reason_counts,
                "selected_source_counts": _source_counts(selected),
            },
        )


def _preselect_score(candidate: Candidate, profile: str, original_rank: int) -> float:
    ranks = candidate.metadata.get("source_ranks") or {}
    if not isinstance(ranks, dict):
        ranks = {}
    score = float(candidate.final_score)
    score += 0.16 / (1.0 + original_rank / 45.0)
    score += 0.09 * min(4, _support_count(candidate))
    score += 0.08 * min(3, len(candidate.matched_constraints))
    score += 0.05 * min(2, len(ranks))
    score += _rank_bonus(ranks, "local_title_bm25", 0.28, 24)
    score += _rank_bonus(ranks, "local_chunk_bm25", 0.18, 30)
    score += _rank_bonus(ranks, "qdrant_dense_paper", 0.22, 38)
    score += _rank_bonus(ranks, "qdrant_sparse_paper", 0.16, 44)
    score += _rank_bonus(ranks, "local_tfidf", 0.08, 30)
    score += _rank_bonus(ranks, "neo4j_concept", 0.07, 24)
    score += _rank_bonus(ranks, "semantic_scholar", 0.24, 32)
    score += _rank_bonus(ranks, "semantic_scholar_snippet", 0.2, 38)
    score += 0.18 * _safe_float(candidate.metadata.get("soft_alias_bonus"))
    score += 0.32 * _safe_float(candidate.metadata.get("strong_alias_bonus"))
    score += 0.2 * _safe_float(candidate.metadata.get("graph_alias_bonus"))
    if profile in {"real_multi_answer", "survey_or_list"} and _support_count(candidate) >= 2:
        score += 0.1
    if profile in {"auto_locator", "foundational_or_origin"} and (
        _rank_present(ranks, "local_title_bm25", 50) or _alias_graph_lane(candidate)
    ):
        score += 0.12
    if _dense_only_weak(candidate):
        score -= 0.16
    if candidate.relevance == "weakly_relevant" and _support_count(candidate) <= 1:
        score -= 0.08
    return score


def _rank_bonus(ranks: dict[str, Any], source: str, weight: float, scale: float) -> float:
    if source not in ranks:
        return 0.0
    rank = max(1, _safe_int(ranks.get(source)))
    return weight / (1.0 + rank / scale)


def _rank_present(ranks: dict[str, Any], source: str, max_rank: int) -> bool:
    return source in ranks and max(1, _safe_int(ranks.get(source))) <= max_rank


def _protected_head_count(profile: str, target: int) -> int:
    if profile in {"auto_locator", "foundational_or_origin"}:
        return min(target, max(8, target // 4))
    if profile in {"real_multi_answer", "survey_or_list"}:
        return min(target, max(6, target // 5))
    return min(target, max(7, target // 5))


def _lane_quota(profile: str, target: int, lane: str) -> int:
    base = {
        "constraint": max(4, target // 5),
        "multi_source": max(6, target // 4),
        "title": max(5, target // 5),
        "dense": max(5, target // 5),
        "sparse": max(4, target // 6),
        "alias": max(2, target // 10),
        "academic": max(4, target // 6),
    }
    if profile in {"real_multi_answer", "survey_or_list"}:
        base["multi_source"] += 3
        base["dense"] += 2
        base["alias"] = max(1, target // 14)
    if profile in {"auto_locator", "foundational_or_origin"}:
        base["title"] += 3
        base["alias"] += 2
    if profile == "dataset_or_benchmark":
        base["dense"] += 2
        base["sparse"] += 2
    return min(target, base[lane])


def _constraint_lane(candidate: Candidate) -> bool:
    return bool(candidate.matched_constraints) and candidate.relevance != "weakly_relevant"


def _multi_source_lane(candidate: Candidate) -> bool:
    return _support_count(candidate) >= 2


def _title_lane(candidate: Candidate) -> bool:
    ranks = candidate.metadata.get("source_ranks") or {}
    return isinstance(ranks, dict) and _rank_present(ranks, "local_title_bm25", 80)


def _dense_lane(candidate: Candidate) -> bool:
    if "qdrant_dense_paper" not in candidate.sources:
        return False
    return _support_count(candidate) >= 2 or bool(candidate.matched_constraints) or _has_alias_signal(candidate)


def _sparse_lane(candidate: Candidate) -> bool:
    if "qdrant_sparse_paper" not in candidate.sources:
        return False
    return _support_count(candidate) >= 2 or _title_lane(candidate) or bool(candidate.matched_constraints)


def _alias_graph_lane(candidate: Candidate) -> bool:
    return _has_alias_signal(candidate) or "neo4j_concept" in candidate.sources


def _academic_lane(candidate: Candidate) -> bool:
    ranks = candidate.metadata.get("source_ranks") or {}
    if not isinstance(ranks, dict):
        return False
    return _rank_present(ranks, "semantic_scholar", 80) and (
        _rank_present(ranks, "semantic_scholar_snippet", 120)
        or bool(candidate.matched_constraints)
        or candidate.relevance != "weakly_relevant"
    )


def _support_count(candidate: Candidate) -> int:
    ranks = candidate.metadata.get("source_ranks") or {}
    if not isinstance(ranks, dict):
        return 0
    support_sources = {
        "local_title_bm25",
        "local_chunk_bm25",
        "local_tfidf",
        "qdrant_dense_paper",
        "qdrant_sparse_paper",
        "neo4j_concept",
        "semantic_scholar",
        "semantic_scholar_snippet",
    }
    return sum(1 for source in support_sources if source in ranks and max(1, _safe_int(ranks[source])) <= 120)


def _has_alias_signal(candidate: Candidate) -> bool:
    return max(
        _safe_float(candidate.metadata.get("soft_alias_bonus")),
        _safe_float(candidate.metadata.get("strong_alias_bonus")),
        _safe_float(candidate.metadata.get("graph_alias_bonus")),
    ) >= 0.35


def _dense_only_weak(candidate: Candidate) -> bool:
    return (
        "qdrant_dense_paper" in candidate.sources
        and _support_count(candidate) <= 1
        and not candidate.matched_constraints
        and not _has_alias_signal(candidate)
    )


def _too_redundant(candidate: Candidate, selected: list[Candidate]) -> bool:
    title_tokens = set(tokenize(candidate.title))
    if not title_tokens:
        return False
    for item in selected[:30]:
        other = set(tokenize(item.title))
        if not other:
            continue
        if len(title_tokens & other) / max(1, len(title_tokens | other)) >= 0.92:
            return True
    return False


def _high_confidence_candidate(candidate: Candidate) -> bool:
    return candidate.relevance == "highly_relevant" or _support_count(candidate) >= 3 or _has_alias_signal(candidate)


def _source_counts(candidates: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        for source in candidate.sources:
            counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
