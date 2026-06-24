from __future__ import annotations

from math import log1p

from scholar_common.models import Candidate, QueryIntent
from scholar_common.text import best_snippet, tokenize


class CandidateRanker:
    def rank(self, candidates: list[Candidate], intent: QueryIntent, *, top_k: int) -> list[Candidate]:
        query_tokens = set(intent.query_tokens)
        ranked: list[Candidate] = []
        for candidate in candidates:
            matched, missing = self._constraint_coverage(candidate, intent)
            candidate.matched_constraints = matched
            candidate.missing_constraints = missing
            candidate.relevance = self._label(candidate, intent, matched, missing)
            candidate.final_score = self._score(candidate, intent, matched, missing)
            if not candidate.snippets:
                candidate.snippets.append(best_snippet(f"{candidate.title}. {candidate.abstract}", query_tokens))
            ranked.append(candidate)
        ranked.sort(key=lambda item: item.final_score, reverse=True)
        return ranked[:top_k]

    def _constraint_coverage(self, candidate: Candidate, intent: QueryIntent) -> tuple[list[str], list[str]]:
        text = f"{candidate.title} {candidate.abstract} {' '.join(candidate.snippets)}".lower()
        matched: list[str] = []
        missing: list[str] = []
        for constraint in intent.must_have_constraints:
            tokens = tokenize(constraint)
            if not tokens:
                continue
            overlap = sum(1 for token in tokens if token in text)
            if overlap >= max(1, min(2, len(tokens))):
                matched.append(constraint)
            else:
                missing.append(constraint)
        return matched, missing

    def _label(
        self,
        candidate: Candidate,
        intent: QueryIntent,
        matched: list[str],
        missing: list[str],
    ) -> str:
        total = max(1, len(intent.must_have_constraints))
        coverage = len(matched) / total
        if coverage >= 0.6 and candidate.raw_scores:
            return "highly_relevant"
        if coverage >= 0.3 or len(candidate.sources) >= 2:
            return "partially_relevant"
        return "weakly_relevant"

    def _score(
        self,
        candidate: Candidate,
        intent: QueryIntent,
        matched: list[str],
        missing: list[str],
    ) -> float:
        total_constraints = max(1, len(intent.must_have_constraints))
        selector_relevance = len(matched) / total_constraints
        keyword_match = self._keyword_overlap(candidate, intent)
        retrieval_signal = self._retrieval_signal(candidate)
        source_confidence = min(1.0, len(candidate.sources) / 3)
        citation_authority = min(1.0, log1p(candidate.citation_count or 0) / 8)
        recency_score = self._recency(candidate.year)
        missing_penalty = min(0.25, 0.03 * len(missing))
        return (
            0.42 * selector_relevance
            + 0.25 * retrieval_signal
            + 0.16 * keyword_match
            + 0.07 * source_confidence
            + 0.05 * citation_authority
            + 0.05 * recency_score
            - missing_penalty
        )

    def _keyword_overlap(self, candidate: Candidate, intent: QueryIntent) -> float:
        candidate_tokens = set(tokenize(f"{candidate.title} {candidate.abstract}"))
        if not intent.query_tokens:
            return 0.0
        return len(candidate_tokens & set(intent.query_tokens)) / max(1, len(set(intent.query_tokens)))

    def _retrieval_signal(self, candidate: Candidate) -> float:
        if not candidate.raw_scores:
            return 0.0
        best = max(candidate.raw_scores.values())
        return best / (best + 5.0)

    def _recency(self, year: int | None) -> float:
        if year is None:
            return 0.2
        if year >= 2023:
            return 1.0
        if year >= 2019:
            return 0.8
        if year >= 2015:
            return 0.6
        return 0.4
