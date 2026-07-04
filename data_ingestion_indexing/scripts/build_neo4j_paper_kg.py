#!/usr/bin/env python3
# 中文功能说明：Neo4j 论文知识图谱构建脚本，负责导入论文、章节、引用和概念关系。

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import socket
import struct
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)

CLIENT_LONG_PASSWORD = 0x00000001
CLIENT_LONG_FLAG = 0x00000004
CLIENT_CONNECT_WITH_DB = 0x00000008
CLIENT_PROTOCOL_41 = 0x00000200
CLIENT_TRANSACTIONS = 0x00002000
CLIENT_SECURE_CONNECTION = 0x00008000
CLIENT_MULTI_RESULTS = 0x00020000
CLIENT_PLUGIN_AUTH = 0x00080000
CLIENT_CONNECT_ATTRS = 0x00100000
CLIENT_SESSION_TRACK = 0x00800000
CLIENT_DEPRECATE_EOF = 0x01000000
COM_QUERY = 0x03


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "study",
    "studies",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "with",
}

CONCEPT_HINTS = {
    "attention",
    "benchmark",
    "classification",
    "dataset",
    "detection",
    "diffusion",
    "embedding",
    "generation",
    "graph",
    "image",
    "language",
    "learning",
    "model",
    "network",
    "pretraining",
    "recommendation",
    "retrieval",
    "segmentation",
    "transformer",
    "video",
}

ALIAS_HINTS = CONCEPT_HINTS | {
    "adapter",
    "agent",
    "agents",
    "benchmark",
    "control",
    "data",
    "deblurring",
    "factuality",
    "matching",
    "navigation",
    "prompting",
    "pruning",
    "reasoning",
    "rendering",
    "token",
    "tokens",
}


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"").strip("'")
    return values


def default_config_path() -> Path:
    candidates = []
    explicit = os.getenv("SCHOLAR_SEARCH_CONFIG")
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/data/csp/config/database.env"),
            Path(__file__).resolve().parents[2] / "config" / "database.env",
            Path.cwd() / "config" / "database.env",
        ]
    )
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return candidate
    return Path("/data/csp/config/database.env")


CONFIG_PATH = default_config_path()
CONFIG_VALUES = read_env_file(CONFIG_PATH)


def config_value(name: str, default: str = "") -> str:
    return os.getenv(name) or CONFIG_VALUES.get(name) or default


def config_path(name: str, default: Path) -> Path:
    value = os.getenv(name) or CONFIG_VALUES.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (CONFIG_PATH.parent / path).resolve()


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}", flush=True)


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_title(value: Any) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\\'", "'").replace('\\"', '"')
    text = PUNCT_RE.sub(" ", text.lower())
    return SPACE_RE.sub(" ", text).strip()


def slugify_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_text.lower() if ch.isalpha())


def stable_hash(prefix: str, *parts: Any, length: int = 20) -> str:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def section_role(title: str | None) -> str:
    text = normalize_title(title)
    if "introduction" in text:
        return "introduction"
    if any(term in text for term in ("related", "background", "preliminaries", "literature")):
        return "related_work"
    if any(term in text for term in ("method", "approach", "model", "algorithm", "architecture")):
        return "method"
    if any(term in text for term in ("experiment", "evaluation", "result", "analysis")):
        return "experiment"
    if any(term in text for term in ("conclusion", "discussion")):
        return "conclusion"
    return "other"


def role_weight(role: str) -> float:
    return {
        "related_work": 1.10,
        "introduction": 1.05,
        "method": 1.00,
        "experiment": 0.85,
        "conclusion": 0.70,
        "other": 0.80,
    }.get(role, 0.80)


def concept_type(name: str) -> str:
    text = normalize_title(name)
    if "dataset" in text or "benchmark" in text:
        return "dataset"
    if any(term in text for term in ("f1", "recall", "precision", "accuracy", "auc")):
        return "metric"
    if any(term in text for term in ("transformer", "clip", "llm", "bert", "gpt", "model")):
        return "model"
    if any(term in text for term in ("method", "algorithm", "attention", "retrieval", "segmentation", "detection")):
        return "method"
    if any(term in text for term in ("vision", "language", "graph", "recommendation")):
        return "domain"
    return "other"


def extract_concepts(*texts: str, limit: int = 12) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for text in texts:
        tokens = [token.lower() for token in TOKEN_RE.findall(text or "")]
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
        for size in (2, 3):
            for index in range(0, max(0, len(tokens) - size + 1)):
                phrase_tokens = tokens[index : index + size]
                if not any(token in CONCEPT_HINTS for token in phrase_tokens):
                    continue
                phrase = " ".join(phrase_tokens)
                counts[phrase] = counts.get(phrase, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    concepts = []
    for phrase, count in ranked[:limit]:
        ctype = concept_type(phrase)
        concepts.append(
            {
                "concept_id": stable_hash(f"concept:{ctype}", phrase),
                "name": phrase,
                "normalized_name": normalize_title(phrase),
                "concept_type": ctype,
                "confidence": min(1.0, 0.55 + 0.08 * count),
            }
        )
    return concepts


def extract_aliases(paper: dict[str, Any], concepts: list[dict[str, Any]], *, limit: int = 16) -> list[dict[str, Any]]:
    aliases: dict[str, dict[str, Any]] = {}
    for concept in concepts:
        name = str(concept.get("name") or "")
        normalized = normalize_title(name)
        if normalized:
            aliases[normalized] = {
                "alias_id": stable_hash("alias", normalized),
                "name": name,
                "normalized_name": normalized,
                "target_concept_id": concept.get("concept_id"),
                "alias_type": concept.get("concept_type") or "concept",
                "confidence": 0.86,
                "source": "concept_phrase",
            }
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    tokens = [token.lower() for token in TOKEN_RE.findall(f"{title}. {abstract[:800]}")]
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    for size in (5, 4, 3, 2):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase_tokens = tokens[index : index + size]
            if not any(token in ALIAS_HINTS for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            normalized = normalize_title(phrase)
            if not normalized or normalized in aliases:
                continue
            ctype = concept_type(phrase)
            concept_id = stable_hash(f"concept:{ctype}", phrase)
            aliases[normalized] = {
                "alias_id": stable_hash("alias", normalized),
                "name": phrase,
                "normalized_name": normalized,
                "target_concept_id": concept_id,
                "alias_type": ctype,
                "confidence": 0.72 if size <= 3 else 0.78,
                "source": "title_abstract_phrase",
            }
            if len(aliases) >= limit:
                return list(aliases.values())[:limit]
    return list(aliases.values())[:limit]


class Neo4jError(RuntimeError):
    pass


class Neo4jHTTP:
    def __init__(self, url: str, username: str, password: str, database: str) -> None:
        self.url = url.rstrip("/")
        self.username = username
        self.password = password
        self.database = database
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Basic {token}",
        }

    def tx(self, statements: list[dict[str, Any]], database: str | None = None, timeout: int = 120) -> dict[str, Any]:
        db = database or self.database
        payload = json.dumps({"statements": statements}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/db/{db}/tx/commit",
            data=payload,
            method="POST",
            headers=self.headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise Neo4jError(f"Neo4j HTTP {exc.code}: {body}") from exc
        if result.get("errors"):
            raise Neo4jError(json_dumps(result["errors"]))
        return result

    def run(self, statement: str, parameters: dict[str, Any] | None = None, database: str | None = None) -> dict[str, Any]:
        return self.tx([{"statement": statement, "parameters": parameters or {}}], database=database)

    def query_value(self, statement: str, parameters: dict[str, Any] | None = None) -> Any:
        result = self.run(statement, parameters)
        results = result.get("results") or []
        data = (results[0].get("data") if results else []) or []
        if not data:
            return None
        row = data[0].get("row") or []
        return row[0] if row else None

    def create_database_if_possible(self, requested: str) -> str:
        if requested in {"neo4j", ""}:
            return "neo4j"
        try:
            self.run(f"CREATE DATABASE {requested} IF NOT EXISTS", database="system")
            log(f"Neo4j database ensured: {requested}")
            time.sleep(2)
            self.database = requested
            self.run("RETURN 1 AS ok")
            return requested
        except Exception as exc:
            log(f"Could not create/use database {requested}; falling back to neo4j namespace. Reason: {exc}")
            self.database = "neo4j"
            self.run("RETURN 1 AS ok", database="neo4j")
            return "neo4j"


class MySQLError(Exception):
    pass


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]


def sql_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


class MySQLClient:
    def __init__(self, host: str, port: int, username: str, password: str, database: str, timeout: int = 30) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def __enter__(self) -> "MySQLClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        _, handshake = _read_packet(sock)
        parsed = _parse_handshake(handshake)
        capabilities = (
            CLIENT_LONG_PASSWORD
            | CLIENT_LONG_FLAG
            | CLIENT_PROTOCOL_41
            | CLIENT_TRANSACTIONS
            | CLIENT_SECURE_CONNECTION
            | CLIENT_MULTI_RESULTS
            | CLIENT_PLUGIN_AUTH
            | CLIENT_CONNECT_ATTRS
            | CLIENT_SESSION_TRACK
            | CLIENT_DEPRECATE_EOF
            | CLIENT_CONNECT_WITH_DB
        )
        if parsed["plugin"] == "caching_sha2_password":
            token = _scramble_caching_sha2(self.password, parsed["seed"])
        else:
            token = _scramble_native(self.password, parsed["seed"])
        attrs = b""
        for key, value in {"_client_name": "neo4j_paper_kg"}.items():
            key_bytes = key.encode()
            value_bytes = value.encode()
            attrs += _lenenc_int(len(key_bytes)) + key_bytes + _lenenc_int(len(value_bytes)) + value_bytes
        payload = struct.pack("<IIB23s", capabilities, 64 * 1024 * 1024, 255, b"\x00" * 23)
        payload += self.username.encode() + b"\x00" + bytes([len(token)]) + token
        payload += self.database.encode() + b"\x00"
        payload += parsed["plugin"].encode() + b"\x00" + _lenenc_int(len(attrs)) + attrs
        _write_packet(sock, 1, payload)
        _, response = _read_packet(sock)
        _check_error(response)
        if response and response[0] == 0x01 and len(response) > 1 and response[1] == 3:
            _, response = _read_packet(sock)
            _check_error(response)
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def execute(self, sql: str) -> QueryResult:
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        _write_packet(self.sock, 0, bytes([COM_QUERY]) + sql.encode("utf-8"))
        _, payload = _read_packet(self.sock)
        _check_error(payload)
        if payload and payload[0] == 0x00:
            return QueryResult([], [])
        column_count, _ = _read_lenenc(payload, 0)
        assert column_count is not None
        columns = []
        for _ in range(column_count):
            _, column_packet = _read_packet(self.sock)
            _check_error(column_packet)
            columns.append(_parse_column_name(column_packet))
        rows = []
        while True:
            _, row_packet = _read_packet(self.sock)
            _check_error(row_packet)
            if row_packet[0] in (0xFE, 0x00) and len(row_packet) < 9:
                break
            rows.append(_parse_row(row_packet, column_count))
        return QueryResult(columns, rows)

    def scalar(self, sql: str) -> Any:
        result = self.execute(sql)
        return result.rows[0][0] if result.rows and result.rows[0] else None


def _read_packet(sock: socket.socket) -> tuple[int, bytes]:
    header = _recv_exact(sock, 4)
    length = header[0] | (header[1] << 8) | (header[2] << 16)
    return header[3], _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise MySQLError("short MySQL packet")
        payload += chunk
    return payload


def _write_packet(sock: socket.socket, sequence: int, payload: bytes) -> None:
    sock.sendall(struct.pack("<I", len(payload))[:3] + bytes([sequence]) + payload)


def _nul_split(data: bytes, start: int = 0) -> tuple[bytes, int]:
    end = data.find(b"\x00", start)
    return (data[start:], len(data)) if end < 0 else (data[start:end], end + 1)


def _lenenc_int(value: int) -> bytes:
    if value < 251:
        return bytes([value])
    if value < 2**16:
        return b"\xfc" + struct.pack("<H", value)
    if value < 2**24:
        return b"\xfd" + struct.pack("<I", value)[:3]
    return b"\xfe" + struct.pack("<Q", value)


def _read_lenenc(data: bytes, position: int = 0) -> tuple[int | None, int]:
    first = data[position]
    if first < 251:
        return first, position + 1
    if first == 0xFB:
        return None, position + 1
    if first == 0xFC:
        return struct.unpack("<H", data[position + 1 : position + 3])[0], position + 3
    if first == 0xFD:
        return data[position + 1] | (data[position + 2] << 8) | (data[position + 3] << 16), position + 4
    return struct.unpack("<Q", data[position + 1 : position + 9])[0], position + 9


def _parse_handshake(payload: bytes) -> dict[str, Any]:
    position = 1
    _, position = _nul_split(payload, position)
    connection_id_end = position + 4
    seed1 = payload[connection_id_end : connection_id_end + 8]
    position = connection_id_end + 9
    lower_caps = struct.unpack("<H", payload[position : position + 2])[0]
    position += 2 + 1 + 2
    upper_caps = struct.unpack("<H", payload[position : position + 2])[0]
    capabilities = lower_caps | (upper_caps << 16)
    position += 2
    auth_len = payload[position] if capabilities & CLIENT_PLUGIN_AUTH else 21
    position += 1 + 10
    seed2_len = max(13, auth_len - 8)
    seed2 = payload[position : position + seed2_len].split(b"\x00", 1)[0]
    position += seed2_len
    plugin = "mysql_native_password"
    if position < len(payload):
        plugin_bytes, _ = _nul_split(payload, position)
        plugin = plugin_bytes.decode("utf-8", "replace") or plugin
    return {"seed": seed1 + seed2, "plugin": plugin}


def _scramble_caching_sha2(password: str, seed: bytes) -> bytes:
    if not password:
        return b""
    stage1 = hashlib.sha256(password.encode()).digest()
    stage2 = hashlib.sha256(stage1).digest()
    stage3 = hashlib.sha256(stage2 + seed).digest()
    return bytes(left ^ right for left, right in zip(stage1, stage3))


def _scramble_native(password: str, seed: bytes) -> bytes:
    if not password:
        return b""
    stage1 = hashlib.sha1(password.encode()).digest()
    stage2 = hashlib.sha1(stage1).digest()
    stage3 = hashlib.sha1(seed + stage2).digest()
    return bytes(left ^ right for left, right in zip(stage1, stage3))


def _check_error(payload: bytes) -> None:
    if payload and payload[0] == 0xFF:
        code = struct.unpack("<H", payload[1:3])[0]
        message = payload[9:].decode("utf-8", "replace")
        raise MySQLError(f"MySQL error {code}: {message}")


def _parse_column_name(payload: bytes) -> str:
    position = 0
    values = []
    for _ in range(6):
        length, position = _read_lenenc(payload, position)
        assert length is not None
        values.append(payload[position : position + length].decode("utf-8", "replace"))
        position += length
    return values[-1]


def _parse_row(payload: bytes, column_count: int) -> list[Any]:
    position = 0
    row = []
    for _ in range(column_count):
        length, position = _read_lenenc(payload, position)
        if length is None:
            row.append(None)
        else:
            raw = payload[position : position + length]
            position += length
            row.append(raw.decode("utf-8", "replace"))
    return row


def neo4j_client(args: argparse.Namespace) -> Neo4jHTTP:
    return Neo4jHTTP(args.neo4j_url, args.neo4j_user, args.neo4j_password, args.neo4j_database)


def mysql_client(args: argparse.Namespace) -> MySQLClient:
    return MySQLClient(args.mysql_host, args.mysql_port, args.mysql_user, args.mysql_password, args.mysql_database)


def es_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if args.es_user:
        token = base64.b64encode(f"{args.es_user}:{args.es_password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    return headers


def es_request(
    args: argparse.Namespace,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    if not args.es_url:
        raise RuntimeError("ELASTICSEARCH_URL is required for ES chunk import")
    data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{args.es_url.rstrip('/')}/{path.lstrip('/')}",
        data=data,
        method=method,
        headers=es_headers(args),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def es_count(args: argparse.Namespace, index: str) -> int | None:
    if not args.es_url:
        return None
    try:
        return int((es_request(args, "GET", f"/{index}/_count", timeout=20) or {}).get("count") or 0)
    except Exception:
        return None


def create_constraints(neo: Neo4jHTTP, graph_name: str) -> None:
    statements = [
        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
        "CREATE CONSTRAINT reference_mention_id IF NOT EXISTS FOR (r:ReferenceMention) REQUIRE r.mention_id IS UNIQUE",
        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",
        "CREATE CONSTRAINT alias_id IF NOT EXISTS FOR (a:Alias) REQUIRE a.alias_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
        "CREATE INDEX alias_normalized_name IF NOT EXISTS FOR (a:Alias) ON (a.normalized_name)",
        "CREATE INDEX concept_id_lookup IF NOT EXISTS FOR (c:Concept) ON (c.concept_id)",
        "MERGE (g:Graph {name: $graph_name}) SET g.kind='paper_kg', g.updated_at=datetime()",
    ]
    for statement in statements:
        neo.run(statement, {"graph_name": graph_name})


def import_papers(neo: Neo4jHTTP, rows: list[dict[str, Any]], graph_name: str) -> None:
    if not rows:
        return
    statement = """
    UNWIND $rows AS row
    MERGE (p:Paper {paper_id: row.paper_id})
    SET p.graph_name = $graph_name,
        p.arxiv_id = row.arxiv_id,
        p.title = row.title,
        p.abstract = row.abstract,
        p.year = row.year,
        p.source = row.source,
        p.fulltext_key = row.fulltext_key,
        p.has_fulltext = row.has_fulltext,
        p.created_from_reference = false,
        p.updated_at = datetime()
    WITH p
    MATCH (g:Graph {name: $graph_name})
    MERGE (g)-[r:CONTAINS]->(p)
    SET r.graph_name = $graph_name,
        r.edge_type = 'graph_contains_paper',
        r.updated_at = datetime()
    """
    neo.run(statement, {"rows": rows, "graph_name": graph_name})


def import_sections(neo: Neo4jHTTP, rows: list[dict[str, Any]], graph_name: str) -> None:
    if not rows:
        return
    statement = """
    UNWIND $rows AS row
    MATCH (p:Paper {paper_id: row.paper_id})
    MERGE (s:Section {section_id: row.section_id})
    SET s.graph_name = $graph_name,
        s.paper_id = row.paper_id,
        s.section_title = row.section_title,
        s.section_index = row.section_index,
        s.section_role = row.section_role,
        s.reference_count = row.reference_count,
        s.updated_at = datetime()
    MERGE (p)-[r:HAS_SECTION]->(s)
    SET r.graph_name = $graph_name,
        r.edge_type = 'paper_has_section',
        r.updated_at = datetime()
    """
    neo.run(statement, {"rows": rows, "graph_name": graph_name})


def import_references_and_edges(neo: Neo4jHTTP, refs: list[dict[str, Any]], graph_name: str) -> None:
    if not refs:
        return
    statement_refs = """
    UNWIND $rows AS row
    MATCH (s:Section {section_id: row.section_id})
    MERGE (m:ReferenceMention {mention_id: row.mention_id})
    SET m.graph_name = $graph_name,
        m.source_paper_id = row.source_paper_id,
        m.section_id = row.section_id,
        m.reference_title = row.reference_title,
        m.normalized_title = row.normalized_title,
        m.target_paper_id = row.target_paper_id,
        m.match_method = row.match_method,
        m.match_score = row.match_score,
        m.updated_at = datetime()
    MERGE (s)-[r:MENTIONS_REFERENCE]->(m)
    SET r.graph_name = $graph_name,
        r.edge_type = 'section_mentions_reference',
        r.updated_at = datetime()
    """
    neo.run(statement_refs, {"rows": refs, "graph_name": graph_name})
    resolved = [row for row in refs if row.get("target_paper_id")]
    if not resolved:
        return
    statement_edges = """
    UNWIND $rows AS row
    MATCH (p:Paper {paper_id: row.source_paper_id})
    MERGE (t:Paper {paper_id: row.target_paper_id})
    ON CREATE SET t.graph_name = $graph_name, t.title = row.target_title, t.created_from_reference = true
    WITH p, t, row
    MATCH (m:ReferenceMention {mention_id: row.mention_id})
    MERGE (m)-[resolved:RESOLVES_TO]->(t)
    SET resolved.graph_name = $graph_name,
        resolved.edge_type = 'reference_resolves_to_paper',
        resolved.updated_at = datetime()
    MERGE (p)-[r:CITES {mention_id: row.mention_id}]->(t)
    SET r.graph_name = $graph_name,
        r.section_id = row.section_id,
        r.section_title = row.section_title,
        r.section_role = row.section_role,
        r.edge_weight = row.edge_weight,
        r.match_method = row.match_method,
        r.match_score = row.match_score,
        r.updated_at = datetime()
    """
    neo.run(statement_edges, {"rows": resolved, "graph_name": graph_name})


def import_concepts(
    neo: Neo4jHTTP,
    concepts: list[dict[str, Any]],
    paper_concepts: list[dict[str, Any]],
    graph_name: str,
) -> None:
    if concepts:
        statement_concepts = """
        UNWIND $rows AS row
        MERGE (c:Concept {concept_id: row.concept_id})
        SET c.graph_name = $graph_name,
            c.name = row.name,
            c.normalized_name = row.normalized_name,
            c.concept_type = row.concept_type,
            c.updated_at = datetime()
        """
        neo.run(statement_concepts, {"rows": concepts, "graph_name": graph_name})
    if paper_concepts:
        statement_edges = """
        UNWIND $rows AS row
        MATCH (p:Paper {paper_id: row.paper_id})
        MATCH (c:Concept {concept_id: row.concept_id})
        MERGE (p)-[r:MENTIONS_CONCEPT {concept_id: row.concept_id, evidence_field: row.evidence_field}]->(c)
        SET r.graph_name = $graph_name,
            r.evidence_text = row.evidence_text,
            r.confidence = row.confidence,
            r.extractor = row.extractor,
            r.updated_at = datetime()
        """
        neo.run(statement_edges, {"rows": paper_concepts, "graph_name": graph_name})


def import_aliases(neo: Neo4jHTTP, aliases: list[dict[str, Any]], graph_name: str) -> None:
    if not aliases:
        return
    statement = """
    UNWIND $rows AS row
    MERGE (c:Concept {concept_id: row.target_concept_id})
    ON CREATE SET c.name = row.name,
        c.normalized_name = row.normalized_name,
        c.concept_type = row.alias_type
    SET c.graph_name = $graph_name,
        c.updated_at = datetime()
    MERGE (a:Alias {alias_id: row.alias_id})
    SET a.graph_name = $graph_name,
        a.name = row.name,
        a.normalized_name = row.normalized_name,
        a.alias_type = row.alias_type,
        a.confidence = row.confidence,
        a.source = row.source,
        a.updated_at = datetime()
    MERGE (a)-[r:ALIAS_OF]->(c)
    SET r.graph_name = $graph_name,
        r.confidence = row.confidence,
        r.source = row.source,
        r.updated_at = datetime()
    """
    neo.run(statement, {"rows": aliases, "graph_name": graph_name})


def prepare_chunk_row(chunk: dict[str, Any], text_chars: int) -> dict[str, Any]:
    text = str(chunk.get("text") or "")
    return {
        "chunk_id": chunk.get("chunk_id"),
        "paper_id": chunk.get("paper_id"),
        "chunk_index": int(chunk["chunk_index"]) if chunk.get("chunk_index") not in (None, "") else None,
        "chunk_type": chunk.get("chunk_type"),
        "section_title": chunk.get("section_title"),
        "text_preview": text[:text_chars],
        "text_length": len(text),
        "token_estimate": int(chunk["token_estimate"]) if chunk.get("token_estimate") not in (None, "") else None,
        "source": chunk.get("source") or "pasa",
    }


def import_chunks(neo: Neo4jHTTP, rows: list[dict[str, Any]], graph_name: str) -> None:
    if not rows:
        return
    statement = """
    UNWIND $rows AS row
    MATCH (p:Paper {paper_id: row.paper_id})
    MERGE (ch:Chunk {chunk_id: row.chunk_id})
    SET ch.graph_name = $graph_name,
        ch.paper_id = row.paper_id,
        ch.chunk_index = row.chunk_index,
        ch.chunk_type = row.chunk_type,
        ch.section_title = row.section_title,
        ch.text_preview = row.text_preview,
        ch.text_length = row.text_length,
        ch.token_estimate = row.token_estimate,
        ch.source = row.source,
        ch.updated_at = datetime()
    MERGE (p)-[r:HAS_CHUNK]->(ch)
    SET r.graph_name = $graph_name,
        r.chunk_type = row.chunk_type,
        r.updated_at = datetime()
    """
    neo.run(statement, {"rows": rows, "graph_name": graph_name})


def delete_until_empty(
    neo: Neo4jHTTP,
    statement: str,
    parameters: dict[str, Any],
    label: str,
) -> int:
    total = 0
    while True:
        deleted = int(neo.query_value(statement, parameters) or 0)
        total += deleted
        if deleted == 0:
            break
        log(f"Deleted {label}: +{deleted}, total={total}")
    return total


def clear_neo4j_data(args: argparse.Namespace) -> dict[str, Any]:
    neo = neo4j_client(args)
    actual_db = neo.create_database_if_possible(args.neo4j_database)
    args.neo4j_database = actual_db
    batch_size = args.delete_batch_size
    stats: dict[str, Any] = {
        "requested_database": args.requested_database,
        "actual_database": actual_db,
        "graph_name": args.graph_name,
        "clear_scope": args.clear_scope,
    }
    if args.clear_scope == "database":
        stats["relationships_before"] = neo.query_value("MATCH ()-[r]->() RETURN count(r)")
        stats["nodes_before"] = neo.query_value("MATCH (n) RETURN count(n)")
        stats["relationships_deleted"] = delete_until_empty(
            neo,
            "MATCH ()-[r]->() WITH r LIMIT $limit WITH collect(r) AS rows FOREACH (row IN rows | DELETE row) RETURN size(rows)",
            {"limit": batch_size},
            "relationships",
        )
        stats["nodes_deleted"] = delete_until_empty(
            neo,
            "MATCH (n) WITH n LIMIT $limit WITH collect(n) AS rows FOREACH (row IN rows | DELETE row) RETURN size(rows)",
            {"limit": batch_size},
            "nodes",
        )
        stats["relationships_after"] = neo.query_value("MATCH ()-[r]->() RETURN count(r)")
        stats["nodes_after"] = neo.query_value("MATCH (n) RETURN count(n)")
        return stats

    stats["relationships_before"] = neo.query_value(
        "MATCH ()-[r]->() WHERE r.graph_name = $graph_name RETURN count(r)",
        {"graph_name": args.graph_name},
    )
    stats["nodes_before"] = neo.query_value(
        "MATCH (n) WHERE n.graph_name = $graph_name OR (n:Graph AND n.name = $graph_name) RETURN count(n)",
        {"graph_name": args.graph_name},
    )
    stats["relationships_deleted"] = delete_until_empty(
        neo,
        """
        MATCH ()-[r]->()
        WHERE r.graph_name = $graph_name
        WITH r LIMIT $limit
        WITH collect(r) AS rows
        FOREACH (row IN rows | DELETE row)
        RETURN size(rows)
        """,
        {"graph_name": args.graph_name, "limit": batch_size},
        f"{args.graph_name} relationships",
    )
    stats["nodes_deleted"] = delete_until_empty(
        neo,
        """
        MATCH (n)
        WHERE n.graph_name = $graph_name OR (n:Graph AND n.name = $graph_name)
        WITH n LIMIT $limit
        DETACH DELETE n
        RETURN count(n)
        """,
        {"graph_name": args.graph_name, "limit": batch_size},
        f"{args.graph_name} nodes",
    )
    stats["relationships_after"] = neo.query_value(
        "MATCH ()-[r]->() WHERE r.graph_name = $graph_name RETURN count(r)",
        {"graph_name": args.graph_name},
    )
    stats["nodes_after"] = neo.query_value(
        "MATCH (n) WHERE n.graph_name = $graph_name OR (n:Graph AND n.name = $graph_name) RETURN count(n)",
        {"graph_name": args.graph_name},
    )
    return stats


def load_title_map(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    id2paper_path = args.raw_root / "paper_database" / "id2paper.json"
    if not id2paper_path.exists():
        log(f"id2paper not found, citation resolution disabled: {id2paper_path}")
        return {}
    data = json.loads(id2paper_path.read_text(encoding="utf-8"))
    title_map = {}
    for arxiv_id, title in data.items():
        key = normalize_title(title)
        if key and key not in title_map:
            title_map[key] = (f"arxiv:{str(arxiv_id).lower()}", str(title))
    log(f"Loaded title map: {len(title_map)} titles")
    return title_map


def iter_mysql_papers(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    with mysql_client(args) as mysql:
        last_id = ""
        fetched = 0
        while True:
            where = f"paper_id > {sql_value(last_id)}" if last_id else "1=1"
            sql = (
                "SELECT paper_id, arxiv_id, title, abstract, year, source, fulltext_key, has_fulltext "
                f"FROM papers WHERE {where} ORDER BY paper_id LIMIT {args.mysql_page_size}"
            )
            result = mysql.execute(sql)
            if not result.rows:
                break
            for row in result.rows:
                item = dict(zip(result.columns, row))
                item["year"] = int(item["year"]) if item.get("year") not in (None, "") else None
                item["has_fulltext"] = str(item.get("has_fulltext") or "0") in {"1", "true", "True"}
                yield item
                fetched += 1
                last_id = item["paper_id"]
                if args.paper_limit and fetched >= args.paper_limit:
                    return


def iter_processed_papers(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    count = 0
    for row in read_jsonl(args.processed_dir / "papers.jsonl"):
        yield row
        count += 1
        if args.paper_limit and count >= args.paper_limit:
            return


def iter_mysql_chunks(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    with mysql_client(args) as mysql:
        last_id = ""
        fetched = 0
        while True:
            where = f"chunk_id > {sql_value(last_id)}" if last_id else "1=1"
            sql = (
                "SELECT chunk_id, paper_id, chunk_index, chunk_type, section_title, text, token_estimate, source "
                f"FROM paper_chunks WHERE {where} ORDER BY chunk_id LIMIT {args.mysql_page_size}"
            )
            result = mysql.execute(sql)
            if not result.rows:
                break
            for row in result.rows:
                item = dict(zip(result.columns, row))
                yield item
                fetched += 1
                last_id = str(item["chunk_id"])
                if args.chunk_limit and fetched >= args.chunk_limit:
                    return


def iter_processed_chunks(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    count = 0
    for row in read_jsonl(args.processed_dir / "paper_chunks.jsonl"):
        yield row
        count += 1
        if args.chunk_limit and count >= args.chunk_limit:
            return


def find_doc(zip_file: Any, paper: dict[str, Any]) -> dict[str, Any] | None:
    key = paper.get("fulltext_key") or slugify_title(str(paper.get("title") or ""))
    if key and key in zip_file.NameToInfo:
        with zip_file.open(key) as handle:
            return json.loads(handle.read().decode("utf-8"))
    return None


def prepare_paper_row(paper: dict[str, Any], abstract_chars: int) -> dict[str, Any]:
    abstract = paper.get("abstract")
    if abstract is not None:
        abstract = str(abstract)[:abstract_chars]
    return {
        "paper_id": paper.get("paper_id"),
        "arxiv_id": paper.get("arxiv_id"),
        "title": paper.get("title") or "",
        "abstract": abstract,
        "year": int(paper["year"]) if paper.get("year") not in (None, "") else None,
        "source": paper.get("source") or "pasa",
        "fulltext_key": paper.get("fulltext_key"),
        "has_fulltext": bool(paper.get("has_fulltext")),
    }


def rows_from_doc(
    paper: dict[str, Any],
    doc: dict[str, Any] | None,
    title_map: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paper_id = str(paper.get("paper_id") or "")
    sections = []
    references = []
    concepts_by_id: dict[str, dict[str, Any]] = {}
    paper_concepts = []

    paper_concept_list = extract_concepts(str(paper.get("title") or ""), str(paper.get("abstract") or ""))
    for concept in paper_concept_list:
        concepts_by_id[concept["concept_id"]] = concept
        paper_concepts.append(
            {
                "paper_id": paper_id,
                "concept_id": concept["concept_id"],
                "evidence_field": "title_abstract",
                "evidence_text": f"{paper.get('title') or ''}. {str(paper.get('abstract') or '')[:220]}",
                "confidence": concept["confidence"],
                "extractor": "rule_phrase",
            }
        )

    raw_sections = (doc or {}).get("sections") or {}
    if not isinstance(raw_sections, dict):
        return sections, references, list(concepts_by_id.values()), paper_concepts

    for index, (section_title, raw_refs) in enumerate(raw_sections.items()):
        refs = raw_refs if isinstance(raw_refs, list) else ([raw_refs] if raw_refs else [])
        role = section_role(str(section_title))
        section_id = stable_hash("section", paper_id, section_title)
        sections.append(
            {
                "section_id": section_id,
                "paper_id": paper_id,
                "section_title": str(section_title),
                "section_index": index,
                "section_role": role,
                "reference_count": len(refs),
            }
        )
        for concept in extract_concepts(str(section_title), limit=5):
            concepts_by_id[concept["concept_id"]] = concept
            paper_concepts.append(
                {
                    "paper_id": paper_id,
                    "concept_id": concept["concept_id"],
                    "evidence_field": "section_title",
                    "evidence_text": str(section_title)[:260],
                    "confidence": min(0.95, concept["confidence"] * 0.85),
                    "extractor": "rule_phrase",
                }
            )
        for ref_index, ref_title in enumerate(refs):
            reference_title = str(ref_title or "").strip()
            if not reference_title:
                continue
            normalized = normalize_title(reference_title)
            target = title_map.get(normalized)
            target_paper_id = target[0] if target else None
            target_title = target[1] if target else None
            match_method = "normalized" if target else "unresolved"
            match_score = 0.95 if target else 0.0
            references.append(
                {
                    "mention_id": stable_hash("ref", paper_id, section_title, ref_index, reference_title),
                    "source_paper_id": paper_id,
                    "section_id": section_id,
                    "section_title": str(section_title),
                    "section_role": role,
                    "reference_title": reference_title[:1000],
                    "normalized_title": normalized[:1000],
                    "target_paper_id": target_paper_id,
                    "target_title": target_title,
                    "match_method": match_method,
                    "match_score": match_score,
                    "edge_weight": round(role_weight(role) * match_score, 6) if target else 0.0,
                }
            )
    return sections, references, list(concepts_by_id.values()), paper_concepts


def build_graph(args: argparse.Namespace) -> dict[str, Any]:
    neo = neo4j_client(args)
    actual_db = neo.create_database_if_possible(args.neo4j_database)
    args.neo4j_database = actual_db
    create_constraints(neo, args.graph_name)

    title_map = load_title_map(args)
    zip_path = args.raw_root / "paper_database" / "cs_paper_2nd.zip"
    zip_file = None
    if zip_path.exists():
        import zipfile

        zip_file = zipfile.ZipFile(zip_path)
        log(f"Opened zip: {zip_path}")
    else:
        log(f"Zip not found; sections/citations disabled: {zip_path}")

    source_iter = iter_mysql_papers(args) if args.source == "mysql" else iter_processed_papers(args)
    stats = {
        "neo4j_database": actual_db,
        "paper_rows": 0,
        "section_rows": 0,
        "reference_rows": 0,
        "resolved_references": 0,
        "concept_rows": 0,
        "paper_concept_edges": 0,
        "alias_rows": 0,
        "es_papers": es_count(args, args.papers_index),
        "es_chunks": es_count(args, args.chunks_index),
    }

    paper_batch: list[dict[str, Any]] = []
    section_batch: list[dict[str, Any]] = []
    reference_batch: list[dict[str, Any]] = []
    concept_batch_by_id: dict[str, dict[str, Any]] = {}
    paper_concept_batch: list[dict[str, Any]] = []
    alias_batch_by_id: dict[str, dict[str, Any]] = {}

    def flush() -> None:
        nonlocal paper_batch, section_batch, reference_batch, concept_batch_by_id, paper_concept_batch, alias_batch_by_id
        import_papers(neo, paper_batch, args.graph_name)
        import_sections(neo, section_batch, args.graph_name)
        import_references_and_edges(neo, reference_batch, args.graph_name)
        import_concepts(neo, list(concept_batch_by_id.values()), paper_concept_batch, args.graph_name)
        import_aliases(neo, list(alias_batch_by_id.values()), args.graph_name)
        paper_batch = []
        section_batch = []
        reference_batch = []
        concept_batch_by_id = {}
        paper_concept_batch = []
        alias_batch_by_id = {}

    try:
        for paper in source_iter:
            paper_id = str(paper.get("paper_id") or "")
            if not paper_id:
                continue
            paper_row = prepare_paper_row(paper, args.abstract_chars)
            doc = find_doc(zip_file, paper) if zip_file else None
            sections, references, concepts, paper_concepts = rows_from_doc(paper_row, doc, title_map)
            paper_batch.append(paper_row)
            section_batch.extend(sections)
            reference_batch.extend(references)
            for concept in concepts:
                concept_batch_by_id[concept["concept_id"]] = concept
            aliases = extract_aliases(paper_row, concepts, limit=args.aliases_per_paper)
            for alias in aliases:
                alias_batch_by_id[alias["alias_id"]] = alias
            paper_concept_batch.extend(paper_concepts)

            stats["paper_rows"] += 1
            stats["section_rows"] += len(sections)
            stats["reference_rows"] += len(references)
            stats["resolved_references"] += sum(1 for row in references if row.get("target_paper_id"))
            stats["concept_rows"] += len(concepts)
            stats["paper_concept_edges"] += len(paper_concepts)
            stats["alias_rows"] += len(aliases)

            if len(paper_batch) >= args.batch_size:
                flush()
                log(
                    "Imported papers={paper_rows} sections={section_rows} refs={reference_rows} resolved={resolved_references}".format(
                        **stats
                    )
                )
        flush()
    finally:
        if zip_file is not None:
            zip_file.close()

    stats["neo4j_paper_count"] = neo.query_value("MATCH (p:Paper {graph_name: $graph_name}) RETURN count(p)", {"graph_name": args.graph_name})
    stats["neo4j_cites_count"] = neo.query_value("MATCH ()-[r:CITES {graph_name: $graph_name}]->() RETURN count(r)", {"graph_name": args.graph_name})
    stats["neo4j_concept_count"] = neo.query_value("MATCH (c:Concept {graph_name: $graph_name}) RETURN count(c)", {"graph_name": args.graph_name})
    stats["neo4j_alias_count"] = neo.query_value("MATCH (a:Alias {graph_name: $graph_name}) RETURN count(a)", {"graph_name": args.graph_name})
    return stats


def build_chunks(args: argparse.Namespace) -> dict[str, Any]:
    neo = neo4j_client(args)
    actual_db = neo.create_database_if_possible(args.neo4j_database)
    args.neo4j_database = actual_db
    create_constraints(neo, args.graph_name)
    source_iter = iter_mysql_chunks(args) if args.source == "mysql" else iter_processed_chunks(args)
    stats = {
        "neo4j_database": actual_db,
        "chunk_rows": 0,
        "skipped_rows": 0,
        "es_chunks": es_count(args, args.chunks_index),
    }
    batch: list[dict[str, Any]] = []
    for chunk in source_iter:
        if not chunk.get("chunk_id") or not chunk.get("paper_id"):
            stats["skipped_rows"] += 1
            continue
        batch.append(prepare_chunk_row(chunk, args.chunk_text_chars))
        stats["chunk_rows"] += 1
        if len(batch) >= args.batch_size:
            import_chunks(neo, batch, args.graph_name)
            batch = []
            log("Imported chunks={chunk_rows} skipped={skipped_rows}".format(**stats))
    if batch:
        import_chunks(neo, batch, args.graph_name)
    stats["neo4j_chunk_count"] = neo.query_value("MATCH (ch:Chunk {graph_name: $graph_name}) RETURN count(ch)", {"graph_name": args.graph_name})
    stats["neo4j_has_chunk_count"] = neo.query_value("MATCH ()-[r:HAS_CHUNK {graph_name: $graph_name}]->() RETURN count(r)", {"graph_name": args.graph_name})
    return stats


def build_aliases(args: argparse.Namespace) -> dict[str, Any]:
    neo = neo4j_client(args)
    actual_db = neo.create_database_if_possible(args.neo4j_database)
    args.neo4j_database = actual_db
    create_constraints(neo, args.graph_name)
    source_iter = iter_mysql_papers(args) if args.source == "mysql" else iter_processed_papers(args)
    stats = {
        "neo4j_database": actual_db,
        "paper_rows": 0,
        "alias_rows": 0,
        "concept_rows": 0,
    }
    concept_batch_by_id: dict[str, dict[str, Any]] = {}
    paper_concept_batch: list[dict[str, Any]] = []
    alias_batch_by_id: dict[str, dict[str, Any]] = {}

    def flush() -> None:
        nonlocal concept_batch_by_id, paper_concept_batch, alias_batch_by_id
        import_concepts(neo, list(concept_batch_by_id.values()), paper_concept_batch, args.graph_name)
        import_aliases(neo, list(alias_batch_by_id.values()), args.graph_name)
        concept_batch_by_id = {}
        paper_concept_batch = []
        alias_batch_by_id = {}

    for paper in source_iter:
        paper_id = str(paper.get("paper_id") or "")
        if not paper_id:
            continue
        paper_row = prepare_paper_row(paper, args.abstract_chars)
        concepts = extract_concepts(str(paper_row.get("title") or ""), str(paper_row.get("abstract") or ""))
        aliases = extract_aliases(paper_row, concepts, limit=args.aliases_per_paper)
        for concept in concepts:
            concept_batch_by_id[concept["concept_id"]] = concept
            paper_concept_batch.append(
                {
                    "paper_id": paper_id,
                    "concept_id": concept["concept_id"],
                    "evidence_field": "title_abstract_alias_build",
                    "evidence_text": f"{paper_row.get('title') or ''}. {str(paper_row.get('abstract') or '')[:220]}",
                    "confidence": concept["confidence"],
                    "extractor": "alias_build_rule_phrase",
                }
            )
        for alias in aliases:
            alias_batch_by_id[alias["alias_id"]] = alias
        stats["paper_rows"] += 1
        stats["concept_rows"] += len(concepts)
        stats["alias_rows"] += len(aliases)
        if len(alias_batch_by_id) >= args.batch_size:
            flush()
            log("Imported alias rows={alias_rows} papers={paper_rows}".format(**stats))
    flush()
    stats["neo4j_alias_count"] = neo.query_value("MATCH (a:Alias {graph_name: $graph_name}) RETURN count(a)", {"graph_name": args.graph_name})
    stats["neo4j_alias_edge_count"] = neo.query_value("MATCH ()-[r:ALIAS_OF {graph_name: $graph_name}]->() RETURN count(r)", {"graph_name": args.graph_name})
    return stats


def cmd_doctor(args: argparse.Namespace) -> None:
    report: dict[str, Any] = {
        "config_path": str(CONFIG_PATH),
        "processed_dir": str(args.processed_dir),
        "raw_root": str(args.raw_root),
        "raw_root_exists": args.raw_root.exists(),
        "papers_jsonl_exists": (args.processed_dir / "papers.jsonl").exists(),
        "zip_exists": (args.raw_root / "paper_database" / "cs_paper_2nd.zip").exists(),
        "neo4j_url": args.neo4j_url,
        "neo4j_database": args.neo4j_database,
        "mysql_host": args.mysql_host,
        "mysql_port": args.mysql_port,
        "mysql_database": args.mysql_database,
        "es_papers": es_count(args, args.papers_index),
        "es_chunks": es_count(args, args.chunks_index),
    }
    neo = neo4j_client(args)
    try:
        actual_db = neo.create_database_if_possible(args.neo4j_database)
        report["neo4j_actual_database"] = actual_db
        report["neo4j_ok"] = neo.query_value("RETURN 1")
    except Exception as exc:
        report["neo4j_error"] = str(exc)
    try:
        with mysql_client(args) as mysql:
            report["mysql_papers"] = mysql.scalar("SELECT COUNT(*) FROM papers")
            report["mysql_chunks"] = mysql.scalar("SELECT COUNT(*) FROM paper_chunks")
    except Exception as exc:
        report["mysql_error"] = str(exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_init(args: argparse.Namespace) -> None:
    neo = neo4j_client(args)
    actual_db = neo.create_database_if_possible(args.neo4j_database)
    args.neo4j_database = actual_db
    create_constraints(neo, args.graph_name)
    result = {
        "graph_name": args.graph_name,
        "requested_database": args.requested_database,
        "actual_database": actual_db,
        "paper_count": neo.query_value("MATCH (p:Paper {graph_name: $graph_name}) RETURN count(p)", {"graph_name": args.graph_name}),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    stats = build_graph(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_build_chunks(args: argparse.Namespace) -> None:
    stats = build_chunks(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_build_aliases(args: argparse.Namespace) -> None:
    stats = build_aliases(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_clear(args: argparse.Namespace) -> None:
    stats = clear_neo4j_data(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(config_value("CSP_BASE_DIR", "/data/csp")).expanduser()
    parser = argparse.ArgumentParser(description="Build ScholarSearch paper KG into Neo4j over HTTP")
    parser.add_argument("--graph-name", default=config_value("NEO4J_GRAPH_NAME", "paper"))
    parser.add_argument("--neo4j-url", default=config_value("NEO4J_HTTP_URL", "http://10.99.24.182:30474"))
    parser.add_argument("--neo4j-user", default=config_value("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=config_value("NEO4J_PASSWORD", ""))
    parser.add_argument("--neo4j-database", default=config_value("NEO4J_DATABASE", "paper"))
    parser.add_argument("--mysql-host", default=config_value("MYSQL_HOST", "10.99.24.182"))
    parser.add_argument("--mysql-port", type=int, default=int(config_value("MYSQL_PORT", "48752")))
    parser.add_argument("--mysql-user", default=config_value("MYSQL_USERNAME", "root"))
    parser.add_argument("--mysql-password", default=config_value("MYSQL_PASSWORD", ""))
    parser.add_argument("--mysql-database", default=config_value("MYSQL_DATABASE", "scholar_search"))
    parser.add_argument("--es-url", default=config_value("ELASTICSEARCH_URL", "http://10.99.24.182:32097").rstrip("/"))
    parser.add_argument("--es-user", default=config_value("ELASTICSEARCH_USERNAME", "kaede"))
    parser.add_argument("--es-password", default=config_value("ELASTICSEARCH_PASSWORD", ""))
    parser.add_argument("--papers-index", default=config_value("PAPERS_INDEX", "saiti3_papers_v1"))
    parser.add_argument("--chunks-index", default=config_value("CHUNKS_INDEX", "saiti3_paper_chunks_v1"))
    parser.add_argument("--raw-root", type=Path, default=config_path("PASA_DATA_ROOT", base_dir / "data_raw" / "pasa" / "data"))
    parser.add_argument("--processed-dir", type=Path, default=config_path("PROCESSED_DIR", base_dir / "data_processed"))
    parser.add_argument("--source", choices=("mysql", "processed"), default="mysql")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--mysql-page-size", type=int, default=1000)
    parser.add_argument("--delete-batch-size", type=int, default=20000)
    parser.add_argument("--clear-scope", choices=("database", "graph"), default="graph")
    parser.add_argument("--paper-limit", type=int, default=None)
    parser.add_argument("--chunk-limit", type=int, default=None)
    parser.add_argument("--chunk-text-chars", type=int, default=500)
    parser.add_argument("--abstract-chars", type=int, default=4000)
    parser.add_argument("--aliases-per-paper", type=int, default=16)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor").set_defaults(func=cmd_doctor)
    subparsers.add_parser("init").set_defaults(func=cmd_init)
    subparsers.add_parser("clear").set_defaults(func=cmd_clear)
    subparsers.add_parser("build").set_defaults(func=cmd_build)
    subparsers.add_parser("build-chunks").set_defaults(func=cmd_build_chunks)
    subparsers.add_parser("build-aliases").set_defaults(func=cmd_build_aliases)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.requested_database = args.neo4j_database
    if not args.neo4j_password:
        raise SystemExit("NEO4J_PASSWORD is required via env or --neo4j-password")
    args.func(args)


if __name__ == "__main__":
    main()
