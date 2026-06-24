from __future__ import annotations

import hashlib
import re
import unicodedata

_ARXIV_URL_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_arxiv_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = _ARXIV_URL_RE.sub("", text)
    text = text.replace(".pdf", "")
    text = text.split("/")[-1]
    text = _ARXIV_VERSION_RE.sub("", text)
    return text.strip().lower() or None


def paper_id_from_arxiv(value: str | None) -> str | None:
    arxiv_id = normalize_arxiv_id(value)
    if not arxiv_id:
        return None
    return f"arxiv:{arxiv_id}"


def title_hash(title: str) -> str:
    normalized = " ".join(title.lower().split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


def slugify_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_text.lower() if ch.isalpha())


def estimate_tokens(text: str) -> int:
    # Good enough for ingestion budgets before model-specific tokenizers are added.
    return max(1, len(text) // 4)

