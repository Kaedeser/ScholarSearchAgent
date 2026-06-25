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
    selector_reranker_enabled: bool
    selector_reranker_url: str
    selector_reranker_candidate_limit: int
    crawler_strategy_enabled: bool
    crawler_strategy_url: str
    crawler_strategy_top_n: int
    timeout_sec: float


@dataclass(frozen=True)
class ScholarSearchSettings:
    agent_root: Path
    config_path: Path
    model_services: ModelServiceSettings

    @classmethod
    def from_env(cls) -> "ScholarSearchSettings":
        agent_root = Path(__file__).resolve().parents[2]
        default_config = _default_config_path(agent_root)
        config_path = Path(os.getenv("SCHOLAR_SEARCH_CONFIG", default_config)).expanduser().resolve()
        config_values = _read_env_file(config_path)
        enabled = _config_bool(config_values, "MODEL_SERVICES_ENABLED", False)
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
                selector_reranker_enabled=_config_bool(config_values, "SELECTOR_RERANKER_ENABLED", True),
                selector_reranker_url=_config_value(
                    config_values,
                    "SELECTOR_RERANKER_SERVICE_URL",
                    "http://10.99.24.182:32082",
                ).rstrip("/"),
                selector_reranker_candidate_limit=max(
                    1,
                    _config_int(config_values, "SELECTOR_RERANKER_CANDIDATE_LIMIT", 100),
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
