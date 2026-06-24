from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return package_root().parents[2]


def default_pasa_data_dir() -> Path:
    return repo_root() / "数据集" / "pasa" / "data"


def sentence_transformers_source_dir() -> Path:
    return package_root() / "framework" / "sentence-transformers"
