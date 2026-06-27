# 中文功能说明：查询理解模块，使用规则抽取研究领域、约束、时间范围和子查询。

from __future__ import annotations

import re
from collections import Counter

from packages.scholar_core.models import QueryIntent
from packages.scholar_core.text import STOPWORDS, compact_terms, normalize_space, tokenize


FIELD_HINTS = {
    "anomaly": "anomaly detection",
    "image": "computer vision",
    "ips": "counterfactual learning",
    "snips": "counterfactual learning",
    "hubert": "speech representation learning",
    "speech": "speech processing",
    "retrieval": "information retrieval",
    "segmentation": "semantic segmentation",
    "language": "natural language processing",
    "llm": "large language models",
    "llms": "large language models",
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
    "anomaly score": ["anomaly detection score", "outlier score", "reconstruction error"],
    "deep q-learning": ["deep q network", "dqn", "target network"],
    "discriminator loss": ["adversarial loss", "gan discriminator", "anomaly score"],
    "edit operation": ["edit operations", "sequence editing", "seq2edit"],
    "edit operation prediction": ["token-level edit operation", "seq2edit"],
    "hubert": ["self-supervised speech representation", "speech units"],
    "hubert codes": ["hubert units", "self-supervised speech representation", "discrete speech units"],
    "image retrieval": ["visual search", "image-text retrieval", "cross-modal retrieval"],
    "image-text": ["image text", "vision-language", "cross-modal"],
    "in-context learning": ["icl", "few-shot prompting", "emergent ability"],
    "instance-level segmentation": ["instance segmentation", "panoptic segmentation", "mask classification"],
    "inverse propensity score": ["inverse propensity scoring", "ips", "counterfactual learning"],
    "semantic segmentation": ["image segmentation", "region-based segmentation", "pixel labeling"],
    "semantic tokens": ["discrete speech units", "speech tokens", "semantic units"],
    "selection bias": ["sample selection bias", "counterfactual learning", "propensity scoring"],
    "self-normalized ips": ["snips", "self-normalized inverse propensity scoring"],
    "supervised fine-tuned": ["supervised fine-tuning", "sft", "instruction tuning"],
    "pretraining": ["pre-training", "self-supervised learning", "representation learning"],
    "large language model": ["llm", "llms", "language model"],
    "mask classification": ["mask classification based segmentation", "mask transformer", "set prediction segmentation"],
    "reconstruction error": ["reconstruction loss", "anomaly score", "reconstruction-based anomaly detection"],
    "reinforcement learning": ["rl", "rlhf", "reinforcement learning from human feedback"],
    "scaling law": ["scaling laws", "model scaling", "scaling behavior"],
    "superpixels": ["region proposals", "image regions"],
    "image patches": ["patch-based", "local image regions"],
    "spatio-temporal": ["spatiotemporal", "temporal spatial"],
    "target networks": ["deep q-learning target network", "dqn target network"],
    "token-level edit": ["edit operation prediction", "seq2edit", "token-level edit operation"],
    "video-text": ["video text", "video-language", "multimodal video"],
}

KEY_PHRASE_PATTERNS = (
    r"\bhubert\s+codes?\b",
    r"\bsemantic\s+tokens?\b",
    r"\bspeech\s+tokens?\b",
    r"\bmask\s+classification(?:-based)?\b",
    r"\binstance(?:-level)?\s+segmentation\b",
    r"\binverse\s+propensity\s+score(?:ing)?\b",
    r"\bself-normalized\s+ips\b",
    r"\bselection\s+bias\b",
    r"\btarget\s+networks?\b",
    r"\bdeep\s+q-learning\b",
    r"\bin-context\s+learning\b",
    r"\bscaling\s+laws?\b",
    r"\bvideo-text\b",
    r"\bimage-text\b",
    r"\breconstruction\s+error\b",
    r"\bdiscriminator\s+loss\b",
    r"\banomaly\s+score\b",
    r"\btoken-level\s+edit(?:\s+operation)?\b",
    r"\bedit\s+operation\s+prediction\b",
    r"\blanguage\s+model(?:s)?\b",
    r"\bimage\s+patch(?:es)?\b",
    r"\bregion-based\s+method(?:s)?\b",
    r"\bsemantic\s+segmentation\b",
    r"\b[a-z]+-[a-z]+\b",
    r"\b[a-z]+\s+retrieval\b",
)

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
        relation_cues = self._relation_cues(clean)
        fields = self._research_fields(query_tokens, phrases)
        must_have = self._must_have_constraints(key_terms, phrases, relation_cues)
        soft = self._soft_constraints(clean, key_terms, phrases, relation_cues)
        excluded = self._excluded_meanings(clean)
        time_range = self._time_range(clean)
        venues = self._venues(clean)
        sub_queries = self._sub_queries(clean, key_terms, phrases, soft, relation_cues)
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
            if _contains_phrase(lowered, phrase):
                phrases.append(phrase)
        for pattern in KEY_PHRASE_PATTERNS:
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

    def _must_have_constraints(
        self,
        key_terms: list[str],
        phrases: list[str],
        relation_cues: list[str],
    ) -> list[str]:
        constraints = phrases[:]
        for term in key_terms:
            if _useful_constraint(term) and term not in constraints:
                constraints.append(term)
        return constraints[:8]

    def _soft_constraints(
        self,
        query: str,
        key_terms: list[str],
        phrases: list[str],
        relation_cues: list[str],
    ) -> list[str]:
        lowered = query.lower()
        soft: list[str] = relation_cues[:]
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

    def _relation_cues(self, query: str) -> list[str]:
        lowered = query.lower()
        cues: list[str] = []
        if any(term in lowered for term in ("first", "introduced", "initially", "pioneer")):
            cues.append("first proposed")
        if any(term in lowered for term in ("negative impact", "negatively impact", "harm", "degrade", "worse")):
            cues.append("performance degradation")
        if "better" in lowered and ("than" in lowered or "bigger" in lowered or "larger" in lowered):
            cues.append("better than larger baseline")
        if any(term in lowered for term in ("claiming", "claim", "show that", "showing that")):
            cues.append("empirical finding")
        if any(term in lowered for term in ("analyzes", "analyse", "analyze", "analysis")):
            cues.append("analysis")
        return _unique(cues)[:4]

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

    def _sub_queries(
        self,
        query: str,
        key_terms: list[str],
        phrases: list[str],
        soft: list[str],
        relation_cues: list[str],
    ) -> list[str]:
        base_terms = " ".join(key_terms[:8])
        query_tokens = tokenize(query)
        cleaned_query = " ".join(compact_terms(query_tokens, limit=12))
        queries: list[str] = []
        if phrases:
            queries.append(" ".join(phrases[:5] + key_terms[:5]))
        if base_terms:
            queries.append(base_terms)
        if cleaned_query:
            queries.append(cleaned_query)
        synonym_terms = phrases[:]
        for phrase in phrases:
            synonym_terms.extend(SYNONYMS.get(phrase, [])[:2])
        if synonym_terms:
            queries.append(" ".join(synonym_terms[:8]))
        if phrases:
            for phrase in phrases[:2]:
                aliases = SYNONYMS.get(phrase, [])[:3]
                queries.append(" ".join([phrase, *aliases, *key_terms[:3]]))
        if relation_cues and phrases:
            queries.append(" ".join([*relation_cues[:2], *phrases[:3], *key_terms[:3]]))
        if soft:
            queries.append(" ".join((key_terms[:5] + soft[:5])[:10]))
        if not queries:
            queries.append(query)
        return _unique([normalize_space(item) for item in queries if item])[:6]

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


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _useful_constraint(term: str) -> bool:
    if term in STOPWORDS:
        return False
    if len(term) <= 2 and not any(char.isdigit() for char in term):
        return False
    return True
