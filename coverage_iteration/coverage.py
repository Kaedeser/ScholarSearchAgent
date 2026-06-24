from __future__ import annotations

from collections import Counter

from scholar_common.models import Candidate, CoverageReport, QueryIntent


class CoverageAnalyzer:
    def analyze(
        self,
        intent: QueryIntent,
        ranked: list[Candidate],
        *,
        min_high_relevant: int = 5,
    ) -> CoverageReport:
        top = ranked[: min(10, len(ranked))]
        matched_counts: Counter[str] = Counter()
        for candidate in top:
            matched_counts.update(candidate.matched_constraints)
        coverage: dict[str, str] = {}
        matched: list[str] = []
        missing: list[str] = []
        for constraint in intent.must_have_constraints:
            count = matched_counts.get(constraint, 0)
            if count >= 2:
                coverage[constraint] = "covered"
                matched.append(constraint)
            elif count == 1:
                coverage[constraint] = "weak"
                matched.append(constraint)
            else:
                coverage[constraint] = "missing"
                missing.append(constraint)
        high_count = sum(1 for candidate in ranked if candidate.relevance == "highly_relevant")
        next_queries = self._next_queries(intent, missing)
        should_continue = bool(missing and high_count < min_high_relevant and next_queries)
        if should_continue:
            reason = "top results still miss required query constraints"
        elif high_count >= min_high_relevant:
            reason = "enough high relevance candidates found for demo budget"
        else:
            reason = "no useful next query generated"
        return CoverageReport(
            coverage=coverage,
            matched_constraints=matched,
            missing_constraints=missing,
            next_queries=next_queries,
            should_continue=should_continue,
            reason=reason,
        )

    def _next_queries(self, intent: QueryIntent, missing: list[str]) -> list[str]:
        if not missing:
            return []
        anchors = intent.research_field[:2] + intent.soft_constraints[:3]
        queries: list[str] = []
        for constraint in missing[:3]:
            terms = [constraint] + anchors
            queries.append(" ".join(dict.fromkeys(term for term in terms if term)))
        return queries
