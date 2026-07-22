# 中文功能说明：统一配置读取模块，负责从环境变量和 env 文件加载模型服务等运行配置。

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


@dataclass(frozen=True)
class ModelServiceSettings:
    enabled: bool
    query_intent_enabled: bool
    query_intent_url: str
    query_intent_mode: str
    query_rewrite_enabled: bool
    query_rewrite_base_url: str
    query_rewrite_api_key: str
    query_rewrite_model: str
    query_rewrite_max_rewrites: int
    query_rewrite_max_tokens: int
    query_rewrite_timeout_sec: float
    query_rewrite_verify_ssl: bool
    query_rewrite_cache_path: str
    selector_reranker_enabled: bool
    selector_reranker_url: str
    selector_reranker_candidate_limit: int
    selector_reranker_pool_limit: int
    selector_reranker_protected_head: int
    crawler_strategy_enabled: bool
    crawler_strategy_url: str
    crawler_strategy_top_n: int
    timeout_sec: float


@dataclass(frozen=True)
class ScholarSearchSettings:
    agent_root: Path
    config_path: Path
    model_services: ModelServiceSettings
    academic_search_enabled: bool
    academic_search_provider: str
    academic_search_base_url: str
    academic_search_api_key: str
    academic_search_timeout_sec: float
    academic_search_query_limit: int
    academic_search_top_k: int
    academic_search_snippet_enabled: bool
    academic_search_snippet_top_k: int
    academic_search_max_retries: int
    academic_search_retry_backoff_sec: float
    academic_search_min_interval_sec: float
    academic_search_cache_size: int
    academic_search_cache_path: str
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
    def from_env(cls) -> "ScholarSearchSettings":
        agent_root = Path(__file__).resolve().parents[2]
        default_config = _default_config_path(agent_root)
        config_path = Path(os.getenv("SCHOLAR_SEARCH_CONFIG", default_config)).expanduser().resolve()
        config_values = _read_env_file(config_path)
        enabled = _config_bool(config_values, "MODEL_SERVICES_ENABLED", True)
        return cls(
            agent_root=agent_root,
            config_path=config_path,
            model_services=ModelServiceSettings(
                enabled=enabled,
                query_intent_enabled=_config_bool(config_values, "QUERY_INTENT_ENABLED", True),
                query_intent_url=_config_value(
                    config_values,
                    "QUERY_INTENT_SERVICE_URL",
                    "http://10.99.24.182:22436",
                ).rstrip("/"),
                query_intent_mode=_config_value(config_values, "QUERY_INTENT_MODE", "auto"),
                query_rewrite_enabled=_config_bool(config_values, "QUERY_REWRITE_ENABLED", False),
                query_rewrite_base_url=_config_value(
                    config_values,
                    "QUERY_REWRITE_BASE_URL",
                    _config_value(config_values, "GPUSTACK_BASE_URL", "http://127.0.0.1:80/v1-openai"),
                ).rstrip("/"),
                query_rewrite_api_key=_config_value(
                    config_values,
                    "QUERY_REWRITE_API_KEY",
                    _config_value(config_values, "GPUSTACK_API_KEY", ""),
                ),
                query_rewrite_model=_config_value(
                    config_values,
                    "QUERY_REWRITE_MODEL",
                    _config_value(config_values, "GPUSTACK_MODEL", ""),
                ),
                query_rewrite_max_rewrites=max(0, _config_int(config_values, "QUERY_REWRITE_MAX_REWRITES", 5)),
                query_rewrite_max_tokens=max(128, _config_int(config_values, "QUERY_REWRITE_MAX_TOKENS", 1024)),
                query_rewrite_timeout_sec=max(
                    0.1,
                    _config_float(
                        config_values,
                        "QUERY_REWRITE_TIMEOUT_SEC",
                        _config_float(config_values, "MODEL_SERVICE_TIMEOUT_SEC", 8.0),
                    ),
                ),
                query_rewrite_verify_ssl=_config_bool(config_values, "QUERY_REWRITE_VERIFY_SSL", True),
                query_rewrite_cache_path=_config_value(
                    config_values,
                    "QUERY_REWRITE_CACHE_PATH",
                    str(agent_root / "cost_control_cache" / "query_rewrite_cache.json"),
                ),
                selector_reranker_enabled=_config_bool(config_values, "SELECTOR_RERANKER_ENABLED", True),
                selector_reranker_url=_config_value(
                    config_values,
                    "SELECTOR_RERANKER_SERVICE_URL",
                    "http://10.99.24.182:32082",
                ).rstrip("/"),
                selector_reranker_candidate_limit=max(
                    1,
                    _config_int(config_values, "SELECTOR_RERANKER_CANDIDATE_LIMIT", 120),
                ),
                selector_reranker_pool_limit=max(
                    1,
                    _config_int(config_values, "SELECTOR_RERANKER_POOL_LIMIT", 500),
                ),
                selector_reranker_protected_head=max(
                    0,
                    _config_int(config_values, "SELECTOR_RERANKER_PROTECTED_HEAD", 0),
                ),
                crawler_strategy_enabled=_config_bool(config_values, "CRAWLER_STRATEGY_ENABLED", True),
                crawler_strategy_url=_config_value(
                    config_values,
                    "CRAWLER_STRATEGY_SERVICE_URL",
                    "http://10.99.24.182:32183",
                ).rstrip("/"),
                crawler_strategy_top_n=max(0, _config_int(config_values, "CRAWLER_STRATEGY_TOP_N", 3)),
                timeout_sec=max(0.1, _config_float(config_values, "MODEL_SERVICE_TIMEOUT_SEC", 8.0)),
            ),
            academic_search_enabled=_config_bool(config_values, "ACADEMIC_SEARCH_ENABLED", False),
            academic_search_provider=_config_value(
                config_values,
                "ACADEMIC_SEARCH_PROVIDER",
                "semantic_scholar",
            ),
            academic_search_base_url=_config_value(
                config_values,
                "ACADEMIC_SEARCH_BASE_URL",
                "https://api.semanticscholar.org/graph/v1",
            ).rstrip("/"),
            academic_search_api_key=_config_value(config_values, "ACADEMIC_SEARCH_API_KEY", ""),
            academic_search_timeout_sec=max(0.1, _config_float(config_values, "ACADEMIC_SEARCH_TIMEOUT_SEC", 8.0)),
            academic_search_query_limit=max(0, _config_int(config_values, "ACADEMIC_SEARCH_QUERY_LIMIT", 2)),
            academic_search_top_k=max(1, _config_int(config_values, "ACADEMIC_SEARCH_TOP_K", 20)),
            academic_search_snippet_enabled=_config_bool(
                config_values,
                "ACADEMIC_SEARCH_SNIPPET_ENABLED",
                True,
            ),
            academic_search_snippet_top_k=max(
                1,
                _config_int(config_values, "ACADEMIC_SEARCH_SNIPPET_TOP_K", 30),
            ),
            academic_search_max_retries=max(
                0,
                _config_int(config_values, "ACADEMIC_SEARCH_MAX_RETRIES", 2),
            ),
            academic_search_retry_backoff_sec=max(
                0.0,
                _config_float(config_values, "ACADEMIC_SEARCH_RETRY_BACKOFF_SEC", 1.0),
            ),
            academic_search_min_interval_sec=max(
                0.0,
                _config_float(config_values, "ACADEMIC_SEARCH_MIN_INTERVAL_SEC", 1.0),
            ),
            academic_search_cache_size=max(
                0,
                _config_int(config_values, "ACADEMIC_SEARCH_CACHE_SIZE", 256),
            ),
            academic_search_cache_path=_config_value(
                config_values,
                "ACADEMIC_SEARCH_CACHE_PATH",
                str(agent_root / "cost_control_cache" / "semantic_scholar_cache.json"),
            ),
            neo4j_http_url=_config_value(config_values, "NEO4J_HTTP_URL", "http://10.99.24.182:30474").rstrip("/"),
            neo4j_user=_config_value(config_values, "NEO4J_USER", "neo4j"),
            neo4j_password=_config_value(config_values, "NEO4J_PASSWORD", ""),
            neo4j_database=_config_value(config_values, "NEO4J_DATABASE", "neo4j"),
            neo4j_graph_name=_config_value(config_values, "NEO4J_GRAPH_NAME", "paper"),
            neo4j_max_seed_papers=max(1, _config_int(config_values, "NEO4J_MAX_SEED_PAPERS", 3)),
            neo4j_max_neighbors=max(1, _config_int(config_values, "NEO4J_MAX_NEIGHBORS", 30)),
            neo4j_min_concept_confidence=min(
                1.0,
                max(0.0, _config_float(config_values, "NEO4J_MIN_CONCEPT_CONFIDENCE", 0.65)),
            ),
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
