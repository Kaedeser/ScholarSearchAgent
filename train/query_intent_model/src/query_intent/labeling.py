from __future__ import annotations

import re


GATE_LABELS = ("paper_search", "non_paper_search")

INTENT_LABELS = (
    "survey_search",
    "method_search",
    "dataset_search",
    "metric_search",
    "mechanism_search",
    "citation_trace",
    "comparison_search",
    "application_search",
)


_CITATION_PATTERNS = (
    r"\bwho\s+(first\s+)?proposed\b",
    r"\bwho\s+(first\s+)?introduced\b",
    r"\borigin(s)?\b",
    r"\bcoined\b",
    r"\binspired\s+by\b",
    r"\bsource\s+of\b",
)

_COMPARISON_PATTERNS = (
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bdifference(s)?\s+between\b",
    r"\bbetter\s+than\b",
)

_DATASET_PATTERNS = (
    r"\bdataset(s)?\b",
    r"\bdata\s+set(s)?\b",
    r"\bbenchmark(s)?\b",
    r"\bcorpus\b",
    r"\bcorpora\b",
    r"\busing\s+[A-Z][A-Za-z0-9_-]{1,}\b",
)

_METRIC_PATTERNS = (
    r"\bmetric(s)?\b",
    r"\baccuracy\b",
    r"\bprecision\b",
    r"\brecall\b",
    r"\bf1\b",
    r"\bauc\b",
    r"\bbleu\b",
    r"\brouge\b",
    r"\bperformance\b",
    r"\bimprov(e|ed|ement|ing)\b",
    r"\boutperform(s|ed|ing)?\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
)

_MECHANISM_PATTERNS = (
    r"\bmechanism(s)?\b",
    r"\bwhy\b",
    r"\bunderstand(ing)?\b",
    r"\bexplain(s|ed|ing|ability|able)?\b",
    r"\binterpret(s|ed|ing|ability|able)?\b",
    r"\banaly(sis|ze|ses)\b",
    r"\bfailure\s+mode(s)?\b",
    r"\bdiagnos(e|is|tic|tics)\b",
    r"\bhow\s+does\b",
)

_SURVEY_PATTERNS = (
    r"\bsurvey(s)?\b",
    r"\boverview(s)?\b",
    r"\breview(s)?\b",
    r"\bliterature\s+review\b",
    r"\brelated\s+(work|works|to)\b",
    r"\bwhat\s+works\s+are\s+related\b",
    r"\blatest\s+progress\b",
)

_APPLICATION_PATTERNS = (
    r"\bapplication(s)?\b",
    r"\bapply(ing|ied)?\b",
    r"\buse\s+of\b",
    r"\bfor\s+[a-z0-9 -]+(detection|recognition|prediction|diagnosis|segmentation)\b",
)


def weak_intent_label(text: str) -> str:
    """Return a deterministic weak intent label for an academic search query."""

    if _matches_any(_CITATION_PATTERNS, text):
        return "citation_trace"
    if _matches_any(_COMPARISON_PATTERNS, text):
        return "comparison_search"
    if _matches_any(_DATASET_PATTERNS, text):
        return "dataset_search"
    if _matches_any(_METRIC_PATTERNS, text):
        return "metric_search"
    if _matches_any(_MECHANISM_PATTERNS, text):
        return "mechanism_search"
    if _matches_any(_SURVEY_PATTERNS, text):
        return "survey_search"
    if _matches_any(_APPLICATION_PATTERNS, text):
        return "application_search"
    return "method_search"


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
