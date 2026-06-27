# 中文功能说明：相关性排序模块，基于约束覆盖、召回信号、关键词、引用和年份计算候选分数。

from __future__ import annotations

from math import log1p

from packages.scholar_core.models import Candidate, QueryIntent
from packages.scholar_core.text import best_snippet, normalize_title, tokenize


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
        text = f"{candidate.title} {candidate.abstract} {' '.join(candidate.snippets)}"
        normalized_text = normalize_title(text)
        candidate_tokens = set(tokenize(text))
        matched: list[str] = []
        missing: list[str] = []
        for constraint in intent.must_have_constraints:
            tokens = tokenize(constraint)
            if not tokens:
                continue
            normalized_constraint = normalize_title(constraint)
            exact_phrase = len(tokens) > 1 and normalized_constraint in normalized_text
            overlap = sum(1 for token in tokens if token in candidate_tokens)
            if exact_phrase:
                overlap = max(overlap, len(tokens))
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
        coverage = self._coverage_ratio(matched, intent.must_have_constraints)
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
        selector_relevance = self._coverage_ratio(matched, intent.must_have_constraints)
        keyword_match = self._keyword_overlap(candidate, intent)
        retrieval_signal = self._retrieval_signal(candidate)
        source_confidence = min(1.0, len(candidate.sources) / 3)
        citation_authority = min(1.0, log1p(candidate.citation_count or 0) / 8)
        recency_score = self._recency(candidate.year)
        exact_phrase_bonus = self._exact_phrase_bonus(candidate, matched)
        missing_penalty = min(0.45, 0.07 * sum(self._constraint_weight(item) for item in missing))
        return (
            0.5 * selector_relevance
            + 0.2 * retrieval_signal
            + 0.15 * keyword_match
            + 0.05 * source_confidence
            + 0.04 * citation_authority
            + 0.03 * recency_score
            + 0.03 * exact_phrase_bonus
            - missing_penalty
        )

    def _keyword_overlap(self, candidate: Candidate, intent: QueryIntent) -> float:
        candidate_tokens = set(tokenize(f"{candidate.title} {candidate.abstract} {' '.join(candidate.snippets)}"))
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

    def _exact_phrase_bonus(self, candidate: Candidate, matched: list[str]) -> float:
        if not matched:
            return 0.0
        normalized_text = normalize_title(f"{candidate.title} {candidate.abstract} {' '.join(candidate.snippets)}")
        phrase_matches = 0
        phrase_total = 0
        for constraint in matched:
            tokens = tokenize(constraint)
            if len(tokens) <= 1:
                continue
            phrase_total += 1
            if normalize_title(constraint) in normalized_text:
                phrase_matches += 1
        if not phrase_total:
            return 0.0
        return phrase_matches / phrase_total

    def _coverage_ratio(self, matched: list[str], constraints: list[str]) -> float:
        total = sum(self._constraint_weight(item) for item in constraints)
        if not total:
            return 0.0
        matched_set = set(matched)
        return sum(self._constraint_weight(item) for item in constraints if item in matched_set) / total

    def _constraint_weight(self, constraint: str) -> float:
        tokens = tokenize(constraint)
        if not tokens:
            return 0.0
        if len(tokens) > 1 or "-" in constraint:
            return 1.6
        if len(tokens[0]) <= 3:
            return 0.7
        return 1.0
