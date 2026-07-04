# 中文功能说明：共享文本处理工具，提供规范化、分词、摘要片段和稀疏向量相似度能力。

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from math import sqrt
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{1,}", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9]+")

STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "aligned",
    "analysed",
    "analyzed",
    "applied",
    "among",
    "and",
    "any",
    "approach",
    "approaches",
    "architecture",
    "architectures",
    "are",
    "as",
    "assumed",
    "at",
    "based",
    "been",
    "being",
    "be",
    "by",
    "between",
    "both",
    "but",
    "better",
    "bigger",
    "can",
    "classic",
    "could",
    "create",
    "created",
    "creates",
    "dataset",
    "datasets",
    "compare",
    "comparisons",
    "comparison",
    "versus",
    "vs",
    "origin",
    "originated",
    "originating",
    "foundational",
    "give",
    "gave",
    "given",
    "cite",
    "cited",
    "cites",
    "claim",
    "claims",
    "claiming",
    "construct",
    "constructed",
    "constructing",
    "develop",
    "developed",
    "developing",
    "develops",
    "does",
    "different",
    "various",
    "individual",
    "such",
    "automatically",
    "prompt-based",
    "discuss",
    "discussed",
    "discusses",
    "discussing",
    "done",
    "demonstrate",
    "demonstrated",
    "demonstrates",
    "during",
    "conduct",
    "conducted",
    "conducting",
    "employ",
    "employed",
    "employs",
    "example",
    "examples",
    "explore",
    "explored",
    "explores",
    "exploring",
    "edge",
    "evaluated",
    "evaluating",
    "field",
    "find",
    "focus",
    "focused",
    "focuses",
    "focusing",
    "first",
    "for",
    "from",
    "have",
    "has",
    "harder",
    "hardness",
    "easier",
    "established",
    "help",
    "how",
    "play",
    "pc",
    "games",
    "than",
    "known",
    "known-as",
    "write",
    "in",
    "including",
    "include",
    "includes",
    "implemented",
    "implements",
    "inductive",
    "various",
    "introduced",
    "introduces",
    "improve",
    "improved",
    "improving",
    "into",
    "know",
    "known",
    "knowledgeable",
    "large",
    "lead",
    "leads",
    "level",
    "list",
    "like",
    "me",
    "other",
    "capacity",
    "region",
    "regions",
    "realm",
    "result",
    "results",
    "revealed",
    "rise",
    "method",
    "methods",
    "model",
    "models",
    "more",
    "most",
    "mention",
    "mentioned",
    "mentions",
    "name",
    "named",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "please",
    "referring",
    "propose",
    "proposed",
    "proposes",
    "provide",
    "providing",
    "present",
    "potential",
    "publication",
    "publications",
    "references",
    "related",
    "research",
    "researches",
    "search",
    "separate",
    "single-task",
    "review",
    "reviews",
    "show",
    "showcase",
    "shows",
    "smaller",
    "some",
    "simple",
    "specify",
    "statistic",
    "statistics",
    "studies",
    "study",
    "studied",
    "task",
    "tasks",
    "that",
    "teaching",
    "themselves",
    "technique",
    "techniques",
    "topic",
    "topics",
    "cutting",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "type",
    "types",
    "tell",
    "to",
    "towards",
    "two",
    "under",
    "use",
    "used",
    "using",
    "sufficient",
    "supporting",
    "via",
    "talk",
    "what",
    "when",
    "were",
    "where",
    "which",
    "who",
    "with",
    "work",
    "works",
    "you",
}


def normalize_space(text: str | None) -> str:
    return " ".join(str(text or "").split())


def normalize_title(title: str | None) -> str:
    text = unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode("ascii")
    text = _PUNCT_RE.sub(" ", text.lower())
    return normalize_space(text)


def tokenize(text: str | None, *, keep_stopwords: bool = False) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    tokens = [match.group(0).lower().strip("-") for match in _TOKEN_RE.finditer(normalized)]
    tokens = [token for token in tokens if token and len(token) > 1]
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


def token_counter(text: str | None, *, keep_stopwords: bool = False) -> Counter[str]:
    return Counter(tokenize(text, keep_stopwords=keep_stopwords))


def cosine_sparse(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def compact_terms(tokens: Iterable[str], limit: int = 10) -> list[str]:
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(limit)]


def best_snippet(text: str, query_tokens: set[str], max_chars: int = 260) -> str:
    clean = normalize_space(text)
    if not clean:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", clean)
    best = max(sentences, key=lambda sentence: len(set(tokenize(sentence)) & query_tokens), default=clean)
    if len(best) <= max_chars:
        return best
    return best[: max_chars - 3].rstrip() + "..."
