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
        source_rank_signal = self._source_rank_signal(candidate)
        rrf_signal = self._rrf_signal(candidate)
        paper_sparse_synergy = self._paper_sparse_synergy(candidate)
        source_confidence = min(1.0, len(candidate.sources) / 3)
        citation_authority = min(1.0, log1p(candidate.citation_count or 0) / 8)
        recency_score = self._recency(candidate.year)
        exact_phrase_bonus = self._exact_phrase_bonus(candidate, matched)
        soft_alias_bonus = self._soft_alias_bonus(candidate, intent)
        strong_alias_bonus = self._strong_alias_bonus(candidate, intent)
        graph_alias_bonus = self._graph_alias_bonus(candidate)
        candidate.metadata["soft_alias_bonus"] = round(soft_alias_bonus, 6)
        candidate.metadata["strong_alias_bonus"] = round(strong_alias_bonus, 6)
        candidate.metadata["graph_alias_bonus"] = round(graph_alias_bonus, 6)
        missing_penalty = min(0.45, 0.07 * sum(self._constraint_weight(item) for item in missing))
        missing_penalty = max(
            0.0,
            missing_penalty - 0.15 * soft_alias_bonus - 0.25 * strong_alias_bonus - 0.18 * graph_alias_bonus,
        )
        return (
            0.28 * selector_relevance
            + 0.05 * retrieval_signal
            + 0.48 * source_rank_signal
            + 0.3 * rrf_signal
            + 0.22 * paper_sparse_synergy
            + 0.1 * keyword_match
            + 0.12 * soft_alias_bonus
            + 0.42 * strong_alias_bonus
            + 0.18 * graph_alias_bonus
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

    def _rrf_signal(self, candidate: Candidate) -> float:
        rrf = sum(float(value) for key, value in candidate.raw_scores.items() if key.startswith("rrf:"))
        return min(1.0, rrf / 0.055)

    def _source_rank_signal(self, candidate: Candidate) -> float:
        ranks = candidate.metadata.get("source_ranks") or {}
        if not isinstance(ranks, dict) or not ranks:
            return 0.0
        weighted = 0.0
        total_weight = 0.0
        source_weights = {
            "local_title_bm25": 1.15,
            "local_chunk_bm25": 1.05,
            "local_tfidf": 0.85,
            "qdrant_dense_paper": 1.2,
            "qdrant_sparse_paper": 0.55,
            "neo4j_alias": 0.75,
            "neo4j_concept": 0.65,
        }
        for source, raw_rank in ranks.items():
            rank = max(1, _safe_int(raw_rank))
            weight = source_weights.get(str(source), 0.7)
            weighted += weight / (1.0 + rank / 8.0)
            total_weight += weight
        if total_weight <= 0:
            return 0.0
        return min(1.0, weighted / min(total_weight, 2.8))

    def _paper_sparse_synergy(self, candidate: Candidate) -> float:
        ranks = candidate.metadata.get("source_ranks") or {}
        if not isinstance(ranks, dict) or "qdrant_sparse_paper" not in ranks:
            return 0.0
        sparse_rank = max(1, _safe_int(ranks.get("qdrant_sparse_paper")))
        lexical_ranks = [
            max(1, _safe_int(ranks[source]))
            for source in ("local_title_bm25", "local_chunk_bm25")
            if source in ranks
        ]
        if not lexical_ranks:
            return 0.18 / (1.0 + sparse_rank / 20.0)
        best_lexical = min(lexical_ranks)
        return min(1.0, 0.55 / (1.0 + sparse_rank / 20.0) + 0.45 / (1.0 + best_lexical / 16.0))

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

    def _soft_alias_bonus(self, candidate: Candidate, intent: QueryIntent) -> float:
        if not intent.soft_constraints:
            return 0.0
        text = f"{candidate.title} {candidate.abstract} {' '.join(candidate.snippets)}"
        normalized_text = normalize_title(text)
        candidate_tokens = set(tokenize(text))
        score = 0.0
        for constraint in intent.soft_constraints:
            tokens = tokenize(constraint)
            if not tokens:
                continue
            if len(tokens) > 1 and normalize_title(constraint) in normalized_text:
                score += min(0.5, 0.12 * len(tokens))
                continue
            if len(tokens) == 1 and len(tokens[0]) >= 5 and tokens[0] in candidate_tokens:
                score += 0.5
        return min(1.0, score)

    def _strong_alias_bonus(self, candidate: Candidate, intent: QueryIntent) -> float:
        if not intent.soft_constraints:
            return 0.0
        normalized_text = normalize_title(f"{candidate.title} {candidate.abstract} {' '.join(candidate.snippets)}")
        score = 0.0
        matches: list[str] = []
        for constraint in intent.soft_constraints:
            tokens = tokenize(constraint)
            normalized_constraint = normalize_title(constraint)
            if (len(tokens) < 3 and "-" not in constraint) or not normalized_constraint:
                continue
            if normalized_constraint not in normalized_text:
                continue
            matches.append(constraint)
            if len(tokens) >= 5:
                score += 0.6
            elif len(tokens) >= 4:
                score += 0.45
            elif "-" in constraint:
                score += 0.35
            else:
                score += 0.25
        candidate.metadata["strong_alias_matches"] = matches[:6]
        return min(1.25, score)

    def _graph_alias_bonus(self, candidate: Candidate) -> float:
        if "neo4j_alias" not in candidate.sources:
            return 0.0
        support = _safe_int(candidate.metadata.get("alias_support"))
        relations = [str(item) for item in candidate.metadata.get("alias_relations") or []]
        matched_terms = [str(item) for item in candidate.metadata.get("alias_matched_terms") or []]
        score = 0.18
        score += min(0.4, 0.08 * max(0, support))
        if matched_terms:
            score += min(0.36, 0.12 * len(matched_terms))
        if candidate.metadata.get("alias_to_concept") or "alias_to_concept_mentions" in relations:
            score += 0.12
        if any(relation in {"PROPOSES", "USES_DATASET", "EVALUATES_ON", "ADDRESSES"} for relation in relations):
            score += 0.24
        return min(1.0, score)

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


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
