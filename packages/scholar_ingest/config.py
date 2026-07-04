# 中文功能说明：数据导入配置模块，读取数据库、索引库、向量库和数据路径配置。

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _config_value(values: Mapping[str, str], name: str, default: str = "") -> str:
    return os.getenv(name) or values.get(name) or default


def _config_bool(values: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = _config_value(values, name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _config_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = _config_value(values, name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _config_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = _config_value(values, name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _config_path(values: Mapping[str, str], name: str, default: Path, *, base_dir: Path) -> Path:
    value = os.getenv(name) or values.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


@dataclass(frozen=True)
class Settings:
    module_root: Path
    workspace_root: Path
    config_path: Path
    pasa_data_root: Path
    processed_dir: Path
    mysql_host: str
    mysql_port: int
    mysql_username: str
    mysql_password: str
    mysql_database: str
    elasticsearch_url: str
    elasticsearch_username: str
    elasticsearch_password: str
    papers_index: str
    chunks_index: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    qdrant_sparse_vector_name: str
    qdrant_sparse_vector_size: int
    qdrant_dense_paper_enabled: bool
    qdrant_dense_paper_collection: str
    qdrant_sparse_paper_enabled: bool
    qdrant_sparse_paper_collection: str
    qdrant_dense_vector_name: str
    qdrant_dense_vector_size: int
    dense_embedding_backend: str
    dense_embedding_model: str
    dense_embedding_device: str
    dense_embedding_base_url: str
    dense_embedding_api_key: str
    dense_embedding_timeout_sec: float
    dense_embedding_verify_ssl: bool
    neo4j_retrieval_enabled: bool
    neo4j_http_url: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    neo4j_graph_name: str
    neo4j_max_seed_papers: int
    neo4j_max_neighbors: int
    neo4j_min_concept_confidence: float

    @classmethod
    def from_env(cls) -> "Settings":
        module_root = Path(__file__).resolve().parent
        agent_root = module_root.parents[1]
        workspace_root = agent_root.parent
        default_config = _default_config_path(agent_root)
        config_path = Path(os.getenv("SCHOLAR_SEARCH_CONFIG", default_config)).expanduser().resolve()
        config_values = _read_env_file(config_path)
        config_base_dir = config_path.parent
        default_pasa = workspace_root / "数据集" / "pasa" / "data"
        return cls(
            module_root=module_root,
            workspace_root=workspace_root,
            config_path=config_path,
            pasa_data_root=_config_path(config_values, "PASA_DATA_ROOT", default_pasa, base_dir=config_base_dir),
            processed_dir=_config_path(
                config_values,
                "PROCESSED_DIR",
                agent_root / "data_ingestion_indexing" / "data_processed",
                base_dir=config_base_dir,
            ),
            mysql_host=_config_value(config_values, "MYSQL_HOST", "10.99.24.182"),
            mysql_port=int(_config_value(config_values, "MYSQL_PORT", "48752")),
            mysql_username=_config_value(config_values, "MYSQL_USERNAME", "root"),
            mysql_password=_config_value(config_values, "MYSQL_PASSWORD", ""),
            mysql_database=_config_value(config_values, "MYSQL_DATABASE", "scholar_search"),
            elasticsearch_url=_config_value(
                config_values,
                "ELASTICSEARCH_URL",
                "http://10.99.24.182:32097",
            ).rstrip("/"),
            elasticsearch_username=_config_value(config_values, "ELASTICSEARCH_USERNAME", "kaede"),
            elasticsearch_password=_config_value(config_values, "ELASTICSEARCH_PASSWORD", ""),
            papers_index=_config_value(config_values, "PAPERS_INDEX", "saiti3_papers_v1"),
            chunks_index=_config_value(config_values, "CHUNKS_INDEX", "saiti3_paper_chunks_v1"),
            qdrant_url=_config_value(config_values, "QDRANT_URL", "http://10.99.24.182:32333").rstrip("/"),
            qdrant_api_key=_config_value(config_values, "QDRANT_API_KEY", ""),
            qdrant_collection=_config_value(config_values, "QDRANT_COLLECTION", "saiti3_paper_chunks_v1"),
            qdrant_sparse_vector_name=_config_value(config_values, "QDRANT_SPARSE_VECTOR_NAME", "text"),
            qdrant_sparse_vector_size=_config_int(config_values, "QDRANT_SPARSE_VECTOR_SIZE", 65536),
            qdrant_dense_paper_enabled=_config_bool(config_values, "QDRANT_DENSE_PAPER_ENABLED", False),
            qdrant_dense_paper_collection=_config_value(
                config_values,
                "QDRANT_DENSE_PAPER_COLLECTION",
                "saiti3_papers_dense_v1",
            ),
            qdrant_sparse_paper_enabled=_config_bool(config_values, "QDRANT_SPARSE_PAPER_ENABLED", False),
            qdrant_sparse_paper_collection=_config_value(
                config_values,
                "QDRANT_SPARSE_PAPER_COLLECTION",
                "saiti3_papers_sparse_v1",
            ),
            qdrant_dense_vector_name=_config_value(config_values, "QDRANT_DENSE_VECTOR_NAME", ""),
            qdrant_dense_vector_size=max(1, _config_int(config_values, "QDRANT_DENSE_VECTOR_SIZE", 768)),
            dense_embedding_backend=_config_value(config_values, "DENSE_EMBEDDING_BACKEND", "sentence_transformers"),
            dense_embedding_model=_config_value(config_values, "DENSE_EMBEDDING_MODEL", ""),
            dense_embedding_device=_config_value(config_values, "DENSE_EMBEDDING_DEVICE", ""),
            dense_embedding_base_url=_config_value(
                config_values,
                "DENSE_EMBEDDING_BASE_URL",
                _config_value(config_values, "GPUSTACK_BASE_URL", ""),
            ).rstrip("/"),
            dense_embedding_api_key=_config_value(
                config_values,
                "DENSE_EMBEDDING_API_KEY",
                _config_value(config_values, "GPUSTACK_API_KEY", ""),
            ),
            dense_embedding_timeout_sec=max(0.1, _config_float(config_values, "DENSE_EMBEDDING_TIMEOUT_SEC", 120.0)),
            dense_embedding_verify_ssl=_config_bool(config_values, "DENSE_EMBEDDING_VERIFY_SSL", True),
            neo4j_http_url=_config_value(config_values, "NEO4J_HTTP_URL", "http://10.99.24.182:30474").rstrip("/"),
            neo4j_user=_config_value(config_values, "NEO4J_USER", "neo4j"),
            neo4j_password=_config_value(config_values, "NEO4J_PASSWORD", ""),
            neo4j_database=_config_value(config_values, "NEO4J_DATABASE", "neo4j"),
            neo4j_graph_name=_config_value(config_values, "NEO4J_GRAPH_NAME", "paper"),
            neo4j_max_seed_papers=max(1, _config_int(config_values, "NEO4J_MAX_SEED_PAPERS", 3)),
            neo4j_max_neighbors=max(1, _config_int(config_values, "NEO4J_MAX_NEIGHBORS", 30)),
            neo4j_min_concept_confidence=max(0.0, min(1.0, _config_float(config_values, "NEO4J_MIN_CONCEPT_CONFIDENCE", 0.65))),
            neo4j_retrieval_enabled=_config_bool(
                config_values,
                "NEO4J_RETRIEVAL_ENABLED",
                False,
            ),
        )


def _default_config_path(agent_root: Path) -> Path:
    candidates = [
        agent_root / "configs" / "database.env",
        agent_root / "config" / "database.env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
