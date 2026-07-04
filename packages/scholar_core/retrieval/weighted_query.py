# 中文功能说明：检索前查询词项加权，借鉴 RAGFlow 的“分词权重 + 短语 boost”思路，为 ES/Qdrant/BM25 提供统一 query 表示。

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from packages.scholar_core.text import STOPWORDS, normalize_title, token_counter, tokenize


_ACRONYM_RE = re.compile(r"^[a-z]{2,12}(?:-[a-z0-9]{2,12})?$")
_NUMBER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_PHRASE_FEATURE_PREFIX = "phrase:"

_LOW_VALUE_TERMS = {
    "approach",
    "approaches",
    "based",
    "difficult",
    "direct",
    "diverse",
    "field",
    "high",
    "quality",
    "paper",
    "papers",
    "processing",
    "research",
    "studies",
    "study",
    "task",
    "tasks",
    "technique",
    "techniques",
    "address",
    "apply",
    "explaining",
    "standout",
    "transformed",
    "unifies",
    "utilizing",
    "want",
    "why",
    "work",
    "works",
}

_HIGH_VALUE_TERMS = {
    "adapter",
    "agent",
    "agents",
    "alignment",
    "anomaly",
    "benchmark",
    "classification",
    "clip",
    "code",
    "corpus",
    "cot",
    "dataset",
    "datasets",
    "detection",
    "diffusion",
    "dpo",
    "embedding",
    "evaluation",
    "factuality",
    "feedback",
    "gan",
    "gameplay",
    "gpt",
    "gnn",
    "graph",
    "hubert",
    "humaneval",
    "hotpotqa",
    "llm",
    "llms",
    "mbpp",
    "mdp",
    "moe",
    "multimodal",
    "ner",
    "nerf",
    "proof",
    "prompt",
    "pretraining",
    "ranking",
    "ranker",
    "rankers",
    "reasoning",
    "reranking",
    "retrieval",
    "reward",
    "re",
    "rlhf",
    "segmentation",
    "slam",
    "scaling",
    "token",
    "tokens",
    "theorem",
    "pointnet",
    "xlnet",
    "sfuda",
    "boed",
    "audio",
    "localization",
    "transformer",
    "vlm",
    "vlms",
    "ee",
}

_DOMAIN_SUFFIXES = (
    "algorithm",
    "benchmark",
    "classification",
    "corpus",
    "dataset",
    "detection",
    "embedding",
    "evaluation",
    "generation",
    "learning",
    "model",
    "models",
    "network",
    "networks",
    "prediction",
    "ranking",
    "reasoning",
    "retrieval",
    "segmentation",
    "token",
    "tokens",
)

_ALIAS_GROUPS = (
    (
        ("cot", "cot prompting", "chain thought", "chain of thought", "chain of thought prompting"),
        ("cot", "prompting"),
        ("cot prompting", "chain of thought prompting"),
    ),
    (
        ("llm", "llms", "large language model", "large language models"),
        ("llm", "llms"),
        ("large language models",),
    ),
    (
        ("vlm", "vlms", "vision language", "vision language model", "vision language models"),
        ("vlm", "vlms"),
        ("vision language models",),
    ),
    (
        ("rlhf", "reinforcement learning human feedback", "reinforcement learning from human feedback"),
        ("rlhf", "feedback", "reward"),
        (
            "reinforcement learning from human feedback",
            "human feedback",
            "preference fine tuning",
            "factually augmented rlhf",
        ),
    ),
    (
        ("dpo", "direct preference optimization"),
        ("dpo",),
        ("direct preference optimization",),
    ),
    (
        ("nerf", "neural radiance field", "neural radiance fields"),
        ("nerf",),
        ("neural radiance fields",),
    ),
    (
        ("clip adapter", "clip-adapter"),
        ("clip", "adapter"),
        ("clip adapter",),
    ),
    (
        ("hotpotqa", "hotpotqa dataset"),
        ("hotpotqa",),
        ("multi-hop question answering", "hotpotqa dataset"),
    ),
    (
        ("moe", "mixture of experts", "moe architecture"),
        ("moe",),
        ("mixture of experts", "sparse mixture of experts"),
    ),
    (
        ("visual-llm", "visual llm", "visual-llm models"),
        ("visual-llm", "llm"),
        ("vision-language model", "multimodal large language model"),
    ),
    (
        ("autoregressive transformer", "autoregressive transformers"),
        ("autoregressive", "transformer"),
        ("autoregressive video generation", "video generation with autoregressive transformers"),
    ),
    (
        ("generate videos", "generate video", "video generation"),
        ("video", "generation"),
        ("video generation", "video synthesis"),
    ),
    (
        (
            "reinforcement learning to optimize diffusion models",
            "reinforcement learning optimize diffusion",
            "optimize diffusion models",
        ),
        ("diffusion", "video", "reward", "feedback"),
        (
            "video diffusion alignment",
            "reward gradients",
            "human feedback",
            "diffusion model alignment",
            "text to video diffusion",
        ),
    ),
    (
        ("video diffusion", "video diffusion models"),
        ("diffusion", "video"),
        ("video diffusion models", "text to video diffusion", "diffusion model alignment"),
    ),
    (
        ("commonsense machine translation", "common sense machine translation", "commonsense problems"),
        ("commonsense", "translation"),
        ("commonsense machine translation", "commonsense reasoning"),
    ),
    (
        ("identity preservation video generation", "identity-preserving video generation"),
        ("identity", "video", "generation"),
        ("identity-preserving video generation", "personalized video generation"),
    ),
    (
        ("vocabulary watermarking", "watermarking language models"),
        ("watermarking", "watermark"),
        ("quality-preserving watermarking", "watermark robustness"),
    ),
    (
        ("dpo training", "dpo vision-language models"),
        ("dpo",),
        ("direct preference optimization", "dpo vision-language models"),
    ),
    (
        ("pc games", "play pc games", "computer games", "action role playing games"),
        ("gameplay", "control", "agents"),
        ("computer control", "game playing", "open world game agents", "gameplay videos"),
    ),
    (
        ("rank search results", "search results", "llm rank", "llm reranking"),
        ("ranking", "reranking", "ranker", "llm"),
        ("large language model reranker", "document reranking", "passage ranking", "zero shot rankers"),
    ),
    (
        ("in-context learning performance", "information extraction tasks", "supervised fine-tuned small language models"),
        ("icl", "ner", "re", "ee"),
        (
            "few shot information extraction",
            "few shot information extractor",
            "biomedical information extraction",
            "sequence labeling",
        ),
    ),
    (
        ("long thought data", "theorem proving data", "proof data"),
        ("proof", "reasoning", "theorem"),
        (
            "theorem and proof data",
            "large scale theorem proving data",
            "theorem proving data synthesis",
            "proof data synthesis",
            "mathematical reasoning data",
        ),
    ),
    (
        ("scaling law", "multi-module models", "mixed modal"),
        ("scaling", "multimodal"),
        ("multimodal scaling laws", "mixed modal language models", "contrastive language image learning"),
    ),
    (
        ("negative impact", "negatively impact"),
        ("reward", "rlhf"),
        ("rlhf generalisation diversity", "reward collapse", "vanishing gradients", "reinforcement finetuning"),
    ),
    (
        ("smaller dataset", "smaller datasets", "less training data"),
        ("data", "pretraining"),
        (
            "data pruning for pretraining",
            "data efficient llms",
            "less training data",
            "fewer data",
            "deduplicating training data",
        ),
    ),
    (
        ("long video description", "long videos", "long video"),
        ("video",),
        (
            "long video captioning",
            "long form video understanding",
            "hour long videos",
            "dense video captions",
            "long video comprehension",
        ),
    ),
    (
        ("point cloud", "point-based methodologies"),
        ("pointnet",),
        ("pointnet", "point sets", "3d classification", "point cloud segmentation"),
    ),
    (
        ("hierarchical transformer", "stacked hierarchical acoustic tokens"),
        ("audio",),
        ("uniaudio", "audio foundation model", "audio generation", "semantic tokens"),
    ),
    (
        ("gumbel-softmax", "boed", "contextual optimization", "contextual optimisation"),
        ("boed",),
        ("bayesian experimental design", "contextual optimisation", "causal decision making"),
    ),
    (
        ("dataset condensation", "bi-level optimization"),
        ("dataset",),
        ("squeeze recover relabel", "imagenet scale", "dataset condensation"),
    ),
    (
        ("sfuda", "source domain data estimation"),
        ("sfuda",),
        ("source-free domain adaptation", "semantic segmentation", "source domain data estimation"),
    ),
    (
        ("explicit localization information", "recovering explicit localization"),
        ("localization",),
        ("perceptual grouping", "contrastive vision language models", "localization information"),
    ),
    (
        ("large web corpus", "web corpus versus wikipedia"),
        ("xlnet",),
        ("xlnet", "cloze driven pretraining", "self attention networks", "wikipedia corpus"),
    ),
    (
        ("oversmoothing", "attention-based gnns"),
        ("gnn",),
        ("oversmoothing graph neural networks", "over smoothing bert", "attention based gnns"),
    ),
)


@dataclass(frozen=True)
class WeightedTerm:
    term: str
    weight: float


@dataclass(frozen=True)
class WeightedQuery:
    original: str
    clean_query: str
    terms: list[WeightedTerm]
    phrases: list[WeightedTerm]

    @property
    def expanded_query(self) -> str:
        values = [term.term for term in self.terms]
        values.extend(phrase.term for phrase in self.phrases[:6])
        return " ".join(dict.fromkeys(values))


def analyze_weighted_query(query: str, *, phrase_hints: list[str] | tuple[str, ...] = ()) -> WeightedQuery:
    normalized = normalize_title(query)
    tokens = tokenize(normalized, keep_stopwords=True)
    base_terms = [_weighted_term(token) for token in tokens if _keep_term(token)]
    alias_terms, alias_phrases = _alias_expansions(normalized)
    phrases = _extract_weighted_phrases(normalized, base_terms, phrase_hints=phrase_hints)
    phrases.extend(alias_phrases)
    terms = [*base_terms, *alias_terms]
    terms = _dedupe_terms(terms)[:24]
    phrases = _dedupe_terms(phrases)
    clean_query = " ".join(term.term for term in terms[:16])
    return WeightedQuery(original=query, clean_query=clean_query, terms=terms, phrases=phrases[:12])


def weighted_token_map(query: str) -> dict[str, float]:
    analyzed = analyze_weighted_query(query)
    weights = {term.term: term.weight for term in analyzed.terms}
    for phrase in analyzed.phrases:
        for token in _kept_tokens(phrase.term):
            weights[token] = max(weights.get(token, 0.0), min(2.4, phrase.weight * 0.7))
    return weights


def weighted_query_tokens(query: str) -> list[str]:
    analyzed = analyze_weighted_query(query)
    tokens: list[str] = []
    for term in analyzed.terms:
        tokens.append(term.term)
    for phrase in analyzed.phrases:
        tokens.extend(_kept_tokens(phrase.term))
    return list(dict.fromkeys(tokens))


def dense_retrieval_query_text(query: str) -> str:
    analyzed = analyze_weighted_query(query)
    parts: list[str] = []
    if analyzed.clean_query:
        parts.append(analyzed.clean_query)
    phrases = [phrase.term for phrase in analyzed.phrases[:6]]
    if phrases:
        parts.append(" ".join(phrases))
    strong_terms = [term.term for term in analyzed.terms if term.weight >= 1.5]
    if strong_terms:
        parts.append(" ".join(strong_terms[:12]))
    text = " ".join(dict.fromkeys(part for part in parts if part))
    return text or normalize_title(query) or str(query or "")


def weighted_sparse_query_feature_map(query: str) -> dict[str, float]:
    analyzed = analyze_weighted_query(query)
    features: dict[str, float] = {}
    for term in analyzed.terms:
        features[term.term] = max(features.get(term.term, 0.0), term.weight)
    for phrase in analyzed.phrases:
        features[_phrase_feature(phrase.term)] = max(features.get(_phrase_feature(phrase.term), 0.0), phrase.weight)
        for token in _kept_tokens(phrase.term):
            features[token] = max(features.get(token, 0.0), min(2.4, phrase.weight * 0.7))
    return features


def sparse_document_feature_map(text: str, *, phrase_limit: int = 80) -> dict[str, float]:
    normalized = normalize_title(text)
    counts = token_counter(normalized, keep_stopwords=True)
    features = {token: float(value) for token, value in counts.items() if _keep_term(token)}
    if not features:
        return {}
    phrase_counts = _document_phrase_counts(normalized)
    _, alias_phrases = _alias_expansions(normalized)
    for phrase, value in phrase_counts.most_common(phrase_limit):
        features[_phrase_feature(phrase)] = max(features.get(_phrase_feature(phrase), 0.0), round(value, 4))
    for phrase in alias_phrases:
        features[_phrase_feature(phrase.term)] = max(features.get(_phrase_feature(phrase.term), 0.0), 1.8)
    return features


def _weighted_term(token: str) -> WeightedTerm:
    weight = 1.0
    if token in _HIGH_VALUE_TERMS:
        weight += 1.2
    if "-" in token:
        weight += 0.7
    if _ACRONYM_RE.match(token) and token not in STOPWORDS:
        weight += 0.5
    if token.endswith(_DOMAIN_SUFFIXES):
        weight += 0.35
    if token in _LOW_VALUE_TERMS:
        weight *= 0.45
    if _NUMBER_RE.match(token):
        weight *= 0.6
    return WeightedTerm(token, round(max(0.1, min(weight, 3.0)), 4))


def _keep_term(token: str) -> bool:
    if not token or token in STOPWORDS:
        if token not in _HIGH_VALUE_TERMS:
            return False
    if token in _LOW_VALUE_TERMS:
        return False
    if len(token) <= 2 and token not in _HIGH_VALUE_TERMS:
        return False
    return True


def _extract_weighted_phrases(
    normalized_query: str,
    terms: list[WeightedTerm],
    *,
    phrase_hints: list[str] | tuple[str, ...],
) -> list[WeightedTerm]:
    phrases: list[WeightedTerm] = []
    for hint in phrase_hints:
        clean = normalize_title(hint)
        if clean and re.search(rf"(?<![a-z0-9]){re.escape(clean)}(?![a-z0-9])", normalized_query):
            phrases.append(WeightedTerm(clean, 3.0))
    tokens = [term.term for term in terms]
    weights = {term.term: term.weight for term in terms}
    for size in (4, 3, 2):
        for index in range(0, max(0, len(tokens) - size + 1)):
            ngram = tokens[index : index + size]
            if not _useful_phrase(ngram):
                continue
            phrase = " ".join(ngram)
            phrase_weight = max(weights.get(token, 1.0) for token in ngram) + 0.55 * (size - 1)
            phrases.append(WeightedTerm(phrase, round(min(4.5, phrase_weight), 4)))
    return _dedupe_terms(phrases)


def _alias_expansions(normalized_query: str) -> tuple[list[WeightedTerm], list[WeightedTerm]]:
    terms: list[WeightedTerm] = []
    phrases: list[WeightedTerm] = []
    for aliases, expanded_terms, expanded_phrases in _ALIAS_GROUPS:
        if not any(_contains_phrase(normalized_query, alias) for alias in aliases):
            continue
        for term in expanded_terms:
            terms.append(_weighted_term(term))
        for phrase in expanded_phrases:
            phrases.append(WeightedTerm(normalize_title(phrase), 4.8))
    return terms, phrases


def _document_phrase_counts(normalized_text: str) -> Counter[str]:
    tokens = _kept_tokens(normalized_text)
    phrase_counts: Counter[str] = Counter()
    for size in (4, 3, 2):
        for index in range(0, max(0, len(tokens) - size + 1)):
            ngram = tokens[index : index + size]
            if _useful_phrase(ngram):
                phrase_counts[" ".join(ngram)] += 1.0 + 0.15 * (size - 1)
    return phrase_counts


def _phrase_feature(phrase: str) -> str:
    return f"{_PHRASE_FEATURE_PREFIX}{normalize_title(phrase)}"


def _kept_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text, keep_stopwords=True) if _keep_term(token)]


def _contains_phrase(normalized_query: str, phrase: str) -> bool:
    clean = normalize_title(phrase)
    return bool(clean and re.search(rf"(?<![a-z0-9]){re.escape(clean)}(?![a-z0-9])", normalized_query))


def _useful_phrase(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    if any(token in _HIGH_VALUE_TERMS or "-" in token for token in tokens):
        return True
    if any(token.endswith(_DOMAIN_SUFFIXES) for token in tokens):
        return True
    return False


def _dedupe_terms(values: list[WeightedTerm]) -> list[WeightedTerm]:
    by_term: dict[str, float] = {}
    for item in values:
        clean = " ".join(item.term.split())
        if not clean:
            continue
        by_term[clean] = max(by_term.get(clean, 0.0), item.weight)
    ordered = sorted(by_term.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    return [WeightedTerm(term, weight) for term, weight in ordered]
