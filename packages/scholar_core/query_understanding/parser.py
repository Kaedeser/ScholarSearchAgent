# 中文功能说明：查询理解模块，使用规则抽取研究领域、约束、时间范围和子查询。

from __future__ import annotations

import re
from collections import Counter

from packages.scholar_core.models import QueryIntent
from packages.scholar_core.text import STOPWORDS, compact_terms, normalize_space, tokenize


FIELD_HINTS = {
    "image": "computer vision",
    "retrieval": "information retrieval",
    "segmentation": "semantic segmentation",
    "language": "natural language processing",
    "pretraining": "large language model pretraining",
    "model": "machine learning",
    "models": "machine learning",
    "graph": "graph learning",
    "video": "video understanding",
    "sign": "sign language recognition",
    "diffusion": "generative modeling",
    "recommendation": "recommender systems",
    "ranking": "information retrieval",
    "superpixels": "computer vision",
    "patches": "computer vision",
}

SYNONYMS = {
    "image retrieval": ["visual search", "image-text retrieval", "cross-modal retrieval"],
    "semantic segmentation": ["image segmentation", "region-based segmentation", "pixel labeling"],
    "pretraining": ["pre-training", "self-supervised learning", "representation learning"],
    "large language model": ["LLM", "language model"],
    "superpixels": ["region proposals", "image regions"],
    "image patches": ["patch-based", "local image regions"],
    "spatio-temporal": ["spatiotemporal", "temporal spatial"],
}

EXCLUSION_PATTERNS = (
    r"not\s+about\s+([^,.?;]+)",
    r"exclude\s+([^,.?;]+)",
    r"except\s+([^,.?;]+)",
)

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


class QueryParser:
    """Rule-based parser used as an offline stand-in for the later LLM parser."""

    def parse(self, query: str) -> QueryIntent:
        clean = normalize_space(query)
        query_tokens = tokenize(clean)
        token_counts = Counter(query_tokens)
        key_terms = compact_terms(token_counts.elements(), limit=12)
        phrases = self._important_phrases(clean)
        fields = self._research_fields(query_tokens, phrases)
        must_have = self._must_have_constraints(key_terms, phrases)
        soft = self._soft_constraints(clean, key_terms, phrases)
        excluded = self._excluded_meanings(clean)
        time_range = self._time_range(clean)
        venues = self._venues(clean)
        sub_queries = self._sub_queries(clean, key_terms, phrases, soft)
        main_intent = self._main_intent(clean, must_have)
        return QueryIntent(
            main_intent=main_intent,
            research_field=fields,
            must_have_constraints=must_have,
            soft_constraints=soft,
            excluded_meanings=excluded,
            time_range=time_range,
            venues=venues,
            sub_queries=sub_queries,
            query_tokens=query_tokens,
        )

    def _important_phrases(self, query: str) -> list[str]:
        lowered = query.lower()
        phrases: list[str] = []
        for phrase in SYNONYMS:
            if phrase in lowered:
                phrases.append(phrase)
        for pattern in (
            r"\b[a-z]+-[a-z]+\b",
            r"\b[a-z]+\s+retrieval\b",
            r"\bsemantic\s+segmentation\b",
            r"\blanguage\s+model(?:s)?\b",
            r"\bimage\s+patch(?:es)?\b",
            r"\bregion-based\s+method(?:s)?\b",
        ):
            for match in re.finditer(pattern, lowered):
                phrase = normalize_space(match.group(0))
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
        return phrases

    def _research_fields(self, tokens: list[str], phrases: list[str]) -> list[str]:
        fields: list[str] = []
        for phrase in phrases:
            if phrase in SYNONYMS:
                fields.append(phrase)
        for token in tokens:
            hint = FIELD_HINTS.get(token)
            if hint:
                fields.append(hint)
        return _unique(fields)[:5] or ["scholarly paper search"]

    def _must_have_constraints(self, key_terms: list[str], phrases: list[str]) -> list[str]:
        constraints = phrases[:]
        for term in key_terms:
            if term not in STOPWORDS and term not in constraints:
                constraints.append(term)
        return constraints[:8]

    def _soft_constraints(self, query: str, key_terms: list[str], phrases: list[str]) -> list[str]:
        lowered = query.lower()
        soft: list[str] = []
        for phrase in phrases:
            soft.extend(SYNONYMS.get(phrase, []))
        if "better" in lowered or "improve" in lowered:
            soft.extend(["performance improvement", "empirical comparison"])
        if "small" in lowered or "smaller" in lowered:
            soft.extend(["data efficiency", "data pruning"])
        if "active learning" in lowered:
            soft.extend(["sample efficiency", "annotation cost"])
        for term in key_terms:
            if len(soft) >= 10:
                break
            if term not in soft:
                soft.append(term)
        return _unique(soft)[:10]

    def _excluded_meanings(self, query: str) -> list[str]:
        lowered = query.lower()
        excluded: list[str] = []
        for pattern in EXCLUSION_PATTERNS:
            for match in re.finditer(pattern, lowered):
                excluded.append(normalize_space(match.group(1)))
        return _unique(excluded)

    def _time_range(self, query: str) -> tuple[int | None, int | None] | None:
        lowered = query.lower()
        years = [int(value) for value in YEAR_RE.findall(lowered)]
        if not years:
            return None
        if "after" in lowered or "since" in lowered:
            return (min(years), None)
        if "before" in lowered or "until" in lowered:
            return (None, max(years))
        if len(years) >= 2:
            return (min(years), max(years))
        return (years[0], years[0])

    def _venues(self, query: str) -> list[str]:
        known = ("acl", "emnlp", "neurips", "iclr", "icml", "cvpr", "iccv", "eccv", "sigir", "kdd")
        lowered = query.lower()
        return [venue.upper() for venue in known if venue in lowered]

    def _sub_queries(self, query: str, key_terms: list[str], phrases: list[str], soft: list[str]) -> list[str]:
        base_terms = " ".join(key_terms[:8])
        queries = [query]
        if base_terms and base_terms.lower() != query.lower():
            queries.append(base_terms)
        if phrases:
            queries.append(" ".join(phrases + key_terms[:4]))
        synonym_terms = phrases[:]
        for phrase in phrases:
            synonym_terms.extend(SYNONYMS.get(phrase, [])[:2])
        if synonym_terms:
            queries.append(" ".join(synonym_terms[:8]))
        if soft:
            queries.append(" ".join((key_terms[:5] + soft[:5])[:10]))
        return _unique([normalize_space(item) for item in queries if item])[:5]

    def _main_intent(self, query: str, constraints: list[str]) -> str:
        if constraints:
            return f"find scholarly papers about {', '.join(constraints[:4])}"
        return f"find scholarly papers relevant to: {query}"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = normalize_space(value)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result
