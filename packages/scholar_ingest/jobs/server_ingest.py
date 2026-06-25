#!/usr/bin/env python3
# 中文功能说明：远端服务器数据导入脚本，负责在 cu 节点上转换、建库、加载和校验数据。

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterable


def read_env_file(path: Path) -> dict[str, str]:
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


def default_config_path() -> Path:
    script_path = Path(__file__).resolve()
    candidates = []
    if len(script_path.parents) > 2:
        candidates.append(script_path.parents[2] / "config" / "database.env")
    if len(script_path.parents) > 1:
        candidates.append(script_path.parents[1] / "config" / "database.env")
    candidates.append(Path.cwd() / "config" / "database.env")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


CONFIG_PATH = Path(os.getenv("SCHOLAR_SEARCH_CONFIG", str(default_config_path()))).expanduser()
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


BASE_DIR = Path(config_value("CSP_BASE_DIR", "/data/csp")).expanduser()
RAW_ROOT = config_path("PASA_DATA_ROOT", BASE_DIR / "data_raw" / "pasa" / "data")
PROCESSED_DIR = config_path("PROCESSED_DIR", BASE_DIR / "data_processed")
LOG_DIR = config_path("LOG_DIR", BASE_DIR / "logs")

ES_URL = config_value("ELASTICSEARCH_URL", "http://10.99.24.182:32097").rstrip("/")
ES_USER = config_value("ELASTICSEARCH_USERNAME", "kaede")
ES_PASSWORD = config_value("ELASTICSEARCH_PASSWORD", "")

MYSQL_HOST = config_value("MYSQL_HOST", "10.99.24.182")
MYSQL_PORT = int(config_value("MYSQL_PORT", "48752"))
MYSQL_USER = config_value("MYSQL_USERNAME", "root")
MYSQL_PASSWORD = config_value("MYSQL_PASSWORD", "")
MYSQL_DATABASE = config_value("MYSQL_DATABASE", "scholar_search")

QDRANT_URL = config_value("QDRANT_URL", "http://10.99.24.182:32333").rstrip("/")
QDRANT_API_KEY = config_value("QDRANT_API_KEY", "")
QDRANT_COLLECTION = config_value("QDRANT_COLLECTION", "saiti3_paper_chunks_v1")

PAPERS_INDEX = config_value("PAPERS_INDEX", "saiti3_papers_v1")
CHUNKS_INDEX = config_value("CHUNKS_INDEX", "saiti3_paper_chunks_v1")

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
POINT_NAMESPACE = uuid.UUID("6ed3b6e7-7d4e-4456-95f8-9e98a9de4ac0")

QUERY_FILES = (
    ("AutoScholarQuery", "train", "AutoScholarQuery/train.jsonl"),
    ("AutoScholarQuery", "dev", "AutoScholarQuery/dev.jsonl"),
    ("AutoScholarQuery", "test", "AutoScholarQuery/test.jsonl"),
    ("RealScholarQuery", "test", "RealScholarQuery/test.jsonl"),
)


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(text, flush=True)
    with (LOG_DIR / "server_ingest.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def write_jsonl_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_arxiv_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "https://arxiv.org/pdf/", "http://arxiv.org/pdf/", "arXiv:", "arxiv:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace(".pdf", "").split("/")[-1]
    if "v" in text:
        head, tail = text.rsplit("v", 1)
        if tail.isdigit():
            text = head
    return text.strip().lower() or None


def paper_id_from_arxiv(value: Any) -> str | None:
    arxiv_id = normalize_arxiv_id(value)
    return f"arxiv:{arxiv_id}" if arxiv_id else None


def title_hash(title: str) -> str:
    digest = hashlib.sha1(" ".join(title.lower().split()).encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


def slugify_title(title: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_text.lower() if ch.isalpha())


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def published_date(source_meta: dict[str, Any] | None) -> str | None:
    value = (source_meta or {}).get("published_time")
    if value is None or value == "":
        return None
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def year_from_arxiv(arxiv_id: str) -> int | None:
    normalized = normalize_arxiv_id(arxiv_id)
    if not normalized:
        return None
    prefix = normalized.split(".", 1)[0]
    if len(prefix) >= 2 and prefix[:2].isdigit():
        yy = int(prefix[:2])
        return 2000 + yy if yy < 90 else 1900 + yy
    return None


def load_zip_doc(zip_file: zipfile.ZipFile, fulltext_key: str | None) -> dict[str, Any] | None:
    if not fulltext_key or fulltext_key not in zip_file.NameToInfo:
        return None
    with zip_file.open(fulltext_key) as handle:
        return json.loads(handle.read().decode("utf-8"))


def chunks_for_paper(paper: dict[str, Any], doc: dict[str, Any] | None):
    paper_id = paper["paper_id"]
    idx = 0
    abstract = paper.get("abstract")
    text = f"Title: {paper['title']}\nAbstract: {abstract}" if abstract else f"Title: {paper['title']}"
    yield {
        "chunk_id": f"{paper_id}#chunk:{idx}",
        "paper_id": paper_id,
        "chunk_index": idx,
        "chunk_type": "title_abstract",
        "section_title": None,
        "text": text,
        "token_estimate": estimate_tokens(text),
        "source": "pasa",
    }
    idx += 1
    sections = (doc or {}).get("sections") or {}
    if isinstance(sections, dict):
        for section_title, references in sections.items():
            if not references:
                continue
            if isinstance(references, list):
                body = "; ".join(str(x) for x in references[:80])
            else:
                body = str(references)
            text = f"Title: {paper['title']}\nSection: {section_title}\nReferenced papers: {body}"
            yield {
                "chunk_id": f"{paper_id}#chunk:{idx}",
                "paper_id": paper_id,
                "chunk_index": idx,
                "chunk_type": "section_references",
                "section_title": str(section_title),
                "text": text,
                "token_estimate": estimate_tokens(text),
                "source": "pasa",
            }
            idx += 1


def convert_queries(limit: int | None = None) -> dict[str, int]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    query_count = gold_count = eval_count = 0
    with (PROCESSED_DIR / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as qh, \
        (PROCESSED_DIR / "gold_labels.jsonl").open("w", encoding="utf-8", newline="\n") as gh, \
        (PROCESSED_DIR / "eval_sets.jsonl").open("w", encoding="utf-8", newline="\n") as eh:
        for dataset_name, split_name, rel_path in QUERY_FILES:
            path = RAW_ROOT / rel_path
            if not path.exists():
                log(f"skip missing query file {path}")
                continue
            seen = 0
            for obj in read_jsonl(path):
                if limit is not None and seen >= limit:
                    break
                seen += 1
                qid = obj.get("qid") or f"{dataset_name}_{split_name}_{seen - 1}"
                answers = [str(x) for x in as_list(obj.get("answer"))]
                arxiv_ids = [normalize_arxiv_id(x) for x in as_list(obj.get("answer_arxiv_id"))]
                gold_paper_ids = [paper_id_from_arxiv(x) for x in arxiv_ids if x]
                published_time = published_date(obj.get("source_meta"))
                write_jsonl_row(qh, {
                    "qid": qid,
                    "dataset_name": dataset_name,
                    "split_name": split_name,
                    "query_text": obj.get("question", ""),
                    "published_time": published_time,
                    "answer_count": len(gold_paper_ids),
                    "source_path": rel_path,
                })
                query_count += 1
                for rank, paper_id in enumerate(gold_paper_ids, start=1):
                    write_jsonl_row(gh, {
                        "qid": qid,
                        "paper_id": paper_id,
                        "arxiv_id": paper_id.replace("arxiv:", "", 1),
                        "title": answers[rank - 1] if rank - 1 < len(answers) else None,
                        "label_rank": rank,
                        "source": dataset_name,
                    })
                    gold_count += 1
                write_jsonl_row(eh, {
                    "dataset_name": dataset_name,
                    "split_name": split_name,
                    "qid": qid,
                    "gold_paper_ids": gold_paper_ids,
                    "published_time": published_time,
                })
                eval_count += 1
            log(f"converted {dataset_name}/{split_name}: {seen} queries")
    return {"queries": query_count, "gold_labels": gold_count, "eval_sets": eval_count}


def convert_papers(limit: int | None = None) -> dict[str, int]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    id2paper_path = RAW_ROOT / "paper_database" / "id2paper.json"
    zip_path = RAW_ROOT / "paper_database" / "cs_paper_2nd.zip"
    log(f"loading id2paper from {id2paper_path}")
    with id2paper_path.open("r", encoding="utf-8") as handle:
        id2paper: dict[str, str] = json.load(handle)
    paper_count = chunk_count = papers_with_doc = 0
    with zipfile.ZipFile(zip_path) as zip_file, \
        (PROCESSED_DIR / "papers.jsonl").open("w", encoding="utf-8", newline="\n") as ph, \
        (PROCESSED_DIR / "paper_chunks.jsonl").open("w", encoding="utf-8", newline="\n") as ch:
        for idx, (arxiv_id, title) in enumerate(id2paper.items(), start=1):
            if limit is not None and idx > limit:
                break
            key = slugify_title(title)
            doc = load_zip_doc(zip_file, key)
            if doc:
                papers_with_doc += 1
            abstract = doc.get("abstract") if doc else None
            paper = {
                "paper_id": paper_id_from_arxiv(arxiv_id) or title_hash(title),
                "arxiv_id": normalize_arxiv_id(arxiv_id),
                "title": title,
                "abstract": abstract,
                "year": year_from_arxiv(arxiv_id),
                "published_time": None,
                "venue": None,
                "authors": [],
                "citation_count": None,
                "source": "pasa",
                "fulltext_key": key if doc else None,
                "has_fulltext": bool(doc),
            }
            write_jsonl_row(ph, paper)
            paper_count += 1
            for chunk in chunks_for_paper(paper, doc):
                write_jsonl_row(ch, chunk)
                chunk_count += 1
            if idx % 10000 == 0:
                log(f"converted papers={idx} chunks={chunk_count} with_doc={papers_with_doc}")
    return {"papers": paper_count, "paper_chunks": chunk_count, "papers_with_zip_doc": papers_with_doc}


def cmd_convert(args: argparse.Namespace) -> None:
    started = time.time()
    stats = {}
    stats.update(convert_queries(args.limit))
    stats.update(convert_papers(args.limit))
    stats["seconds"] = round(time.time() - started, 3)
    stats["processed_dir"] = str(PROCESSED_DIR)
    with (PROCESSED_DIR / "conversion_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    log(f"conversion complete {stats}")


def es_headers() -> dict[str, str]:
    token = base64.b64encode(f"{ES_USER}:{ES_PASSWORD}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def http_json(method: str, url: str, body: Any | None = None, headers: dict[str, str] | None = None, retries: int = 3) -> Any:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload) if payload else None
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", "replace")
            if exc.code < 500 or attempt >= retries:
                raise RuntimeError(f"{method} {url} failed: {exc.code} {payload}") from exc
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"{method} {url} failed after {retries} attempts: {exc}") from exc
        time.sleep(min(30, attempt * 5))
    raise RuntimeError(f"{method} {url} failed")


PAPERS_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "paper_id": {"type": "keyword"},
        "arxiv_id": {"type": "keyword"},
        "title": {"type": "text", "fields": {"raw": {"type": "keyword", "ignore_above": 512}}},
        "abstract": {"type": "text"},
        "year": {"type": "integer"},
        "published_time": {"type": "date"},
        "venue": {"type": "keyword"},
        "authors": {"type": "keyword"},
        "citation_count": {"type": "integer"},
        "source": {"type": "keyword"},
        "fulltext_key": {"type": "keyword", "ignore_above": 1024},
        "has_fulltext": {"type": "boolean"},
    }},
}

CHUNKS_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "chunk_id": {"type": "keyword"},
        "paper_id": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "chunk_type": {"type": "keyword"},
        "section_title": {"type": "text", "fields": {"raw": {"type": "keyword", "ignore_above": 512}}},
        "text": {"type": "text"},
        "token_estimate": {"type": "integer"},
        "source": {"type": "keyword"},
    }},
}


def es_delete_if_exists(index_name: str) -> None:
    try:
        http_json("DELETE", f"{ES_URL}/{index_name}", headers=es_headers())
        log(f"deleted ES index {index_name}")
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise


def cmd_reset_es(args: argparse.Namespace) -> None:
    es_delete_if_exists(PAPERS_INDEX)
    es_delete_if_exists(CHUNKS_INDEX)
    http_json("PUT", f"{ES_URL}/{PAPERS_INDEX}", PAPERS_MAPPING, es_headers())
    http_json("PUT", f"{ES_URL}/{CHUNKS_INDEX}", CHUNKS_MAPPING, es_headers())
    log(f"created ES indices {PAPERS_INDEX}, {CHUNKS_INDEX}")


def es_bulk(index_name: str, path: Path, id_field: str, batch_size: int) -> int:
    headers = es_headers()
    headers["Content-Type"] = "application/x-ndjson"
    count = 0
    batch: list[str] = []
    started = time.time()
    for row in read_jsonl(path):
        batch.append(json.dumps({"index": {"_index": index_name, "_id": row[id_field]}}, ensure_ascii=False))
        batch.append(json.dumps(row, ensure_ascii=False))
        count += 1
        if count % batch_size == 0:
            es_send_bulk(batch, headers)
            batch = []
            if count % (batch_size * 20) == 0:
                log(f"ES loaded {index_name}: {count}")
    if batch:
        es_send_bulk(batch, headers)
    log(f"ES loaded {index_name}: {count} rows in {round(time.time() - started, 2)}s")
    return count


def es_send_bulk(lines: list[str], headers: dict[str, str]) -> None:
    data = ("\n".join(lines) + "\n").encode("utf-8")
    req = urllib.request.Request(f"{ES_URL}/_bulk", data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("errors"):
        first_error = next((item for item in result.get("items", []) if item.get("index", {}).get("error")), None)
        raise RuntimeError(f"ES bulk errors: {first_error}")


def cmd_load_es(args: argparse.Namespace) -> None:
    es_bulk(PAPERS_INDEX, PROCESSED_DIR / "papers.jsonl", "paper_id", args.batch_size)
    es_bulk(CHUNKS_INDEX, PROCESSED_DIR / "paper_chunks.jsonl", "chunk_id", args.batch_size)


CLIENT_LONG_PASSWORD = 0x00000001
CLIENT_LONG_FLAG = 0x00000004
CLIENT_PROTOCOL_41 = 0x00000200
CLIENT_TRANSACTIONS = 0x00002000
CLIENT_SECURE_CONNECTION = 0x00008000
CLIENT_MULTI_RESULTS = 0x00020000
CLIENT_PLUGIN_AUTH = 0x00080000
CLIENT_CONNECT_ATTRS = 0x00100000
CLIENT_SESSION_TRACK = 0x00800000
CLIENT_DEPRECATE_EOF = 0x01000000
COM_QUERY = 0x03


class MySQLError(Exception):
    pass


def mysql_read_packet(sock):
    header = sock.recv(4)
    if len(header) < 4:
        raise MySQLError("short packet header")
    length = header[0] | (header[1] << 8) | (header[2] << 16)
    seq = header[3]
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise MySQLError("short packet payload")
        payload += chunk
    return seq, payload


def mysql_write_packet(sock, seq: int, payload: bytes) -> None:
    sock.sendall(struct.pack("<I", len(payload))[:3] + bytes([seq]) + payload)


def nul_split(data: bytes, start: int = 0):
    end = data.find(b"\x00", start)
    return (data[start:], len(data)) if end < 0 else (data[start:end], end + 1)


def lenenc_int(n: int) -> bytes:
    if n < 251:
        return bytes([n])
    if n < 2**16:
        return b"\xfc" + struct.pack("<H", n)
    if n < 2**24:
        return b"\xfd" + struct.pack("<I", n)[:3]
    return b"\xfe" + struct.pack("<Q", n)


def read_lenenc(data: bytes, pos: int = 0):
    first = data[pos]
    if first < 251:
        return first, pos + 1
    if first == 0xFC:
        return struct.unpack("<H", data[pos + 1:pos + 3])[0], pos + 3
    if first == 0xFD:
        return data[pos + 1] | (data[pos + 2] << 8) | (data[pos + 3] << 16), pos + 4
    if first == 0xFE:
        return struct.unpack("<Q", data[pos + 1:pos + 9])[0], pos + 9
    raise MySQLError("bad lenenc")


def parse_mysql_err(payload: bytes) -> str:
    code = struct.unpack("<H", payload[1:3])[0] if len(payload) >= 3 else None
    msg_start = 3
    state = ""
    if len(payload) >= 9 and payload[3:4] == b"#":
        state = payload[4:9].decode("ascii", "replace")
        msg_start = 9
    msg = payload[msg_start:].decode("utf-8", "replace")
    return f"ERR {code} {state} {msg}".strip()


def mysql_check_err(payload: bytes) -> None:
    if payload and payload[0] == 0xFF:
        raise MySQLError(parse_mysql_err(payload))


def scramble_native(password: str, seed: bytes) -> bytes:
    s1 = hashlib.sha1(password.encode()).digest()
    s2 = hashlib.sha1(s1).digest()
    s3 = hashlib.sha1(seed + s2).digest()
    return bytes(a ^ b for a, b in zip(s1, s3))


def scramble_caching_sha2(password: str, seed: bytes) -> bytes:
    s1 = hashlib.sha256(password.encode()).digest()
    s2 = hashlib.sha256(s1).digest()
    s3 = hashlib.sha256(s2 + seed).digest()
    return bytes(a ^ b for a, b in zip(s1, s3))


def mysql_parse_handshake(payload: bytes) -> dict[str, Any]:
    pos = 1
    server_version_raw, pos = nul_split(payload, pos)
    pos += 4
    auth1 = payload[pos:pos + 8]
    pos += 9
    pos += 2 + 1 + 2 + 2
    auth_len = payload[pos]
    pos += 1 + 10
    auth2_len = max(13, auth_len - 8) if auth_len else 13
    auth2 = payload[pos:pos + auth2_len]
    pos += auth2_len
    plugin = "mysql_native_password"
    if pos < len(payload):
        plugin_raw, pos = nul_split(payload, pos)
        if plugin_raw:
            plugin = plugin_raw.decode("ascii", "replace")
    return {"server_version": server_version_raw.decode("utf-8", "replace"), "seed": (auth1 + auth2).split(b"\x00")[0], "plugin": plugin}


def mysql_connect():
    sock = socket.create_connection((MYSQL_HOST, MYSQL_PORT), timeout=20)
    _, hs_payload = mysql_read_packet(sock)
    hs = mysql_parse_handshake(hs_payload)
    caps = CLIENT_LONG_PASSWORD | CLIENT_LONG_FLAG | CLIENT_PROTOCOL_41 | CLIENT_TRANSACTIONS | CLIENT_SECURE_CONNECTION | CLIENT_MULTI_RESULTS | CLIENT_PLUGIN_AUTH | CLIENT_CONNECT_ATTRS | CLIENT_SESSION_TRACK | CLIENT_DEPRECATE_EOF
    token = scramble_caching_sha2(MYSQL_PASSWORD, hs["seed"]) if hs["plugin"] == "caching_sha2_password" else scramble_native(MYSQL_PASSWORD, hs["seed"])
    attrs = b""
    for k, v in {"_client_name": "csp_server_ingest"}.items():
        kb = k.encode()
        vb = v.encode()
        attrs += lenenc_int(len(kb)) + kb + lenenc_int(len(vb)) + vb
    payload = struct.pack("<IIB23s", caps, 64 * 1024 * 1024, 255, b"\x00" * 23)
    payload += MYSQL_USER.encode() + b"\x00" + bytes([len(token)]) + token + hs["plugin"].encode() + b"\x00" + lenenc_int(len(attrs)) + attrs
    mysql_write_packet(sock, 1, payload)
    _, payload = mysql_read_packet(sock)
    mysql_check_err(payload)
    if payload and payload[0] == 0x01 and len(payload) > 1 and payload[1] == 3:
        _, payload = mysql_read_packet(sock)
        mysql_check_err(payload)
    return sock


def mysql_query(sock, sql: str) -> list[list[str]]:
    mysql_write_packet(sock, 0, bytes([COM_QUERY]) + sql.encode("utf-8"))
    _, payload = mysql_read_packet(sock)
    mysql_check_err(payload)
    if payload and payload[0] == 0x00:
        return []
    col_count, _ = read_lenenc(payload, 0)
    for _ in range(col_count):
        _, col = mysql_read_packet(sock)
        mysql_check_err(col)
    rows = []
    while True:
        _, row = mysql_read_packet(sock)
        mysql_check_err(row)
        if row[0] in (0xFE, 0x00) and len(row) < 9:
            break
        pos = 0
        vals = []
        for _ in range(col_count):
            ln, pos = read_lenenc(row, pos)
            vals.append(row[pos:pos + ln].decode("utf-8", "replace"))
            pos += ln
        rows.append(vals)
    return rows


def sql_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def mysql_insert_rows(sock, table: str, columns: list[str], rows: list[dict[str, Any]], update_columns: list[str] | None = None) -> None:
    if not rows:
        return
    values = []
    for row in rows:
        values.append("(" + ",".join(sql_value(row.get(col)) for col in columns) + ")")
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES " + ",".join(values)
    if update_columns:
        sql += " ON DUPLICATE KEY UPDATE " + ",".join(f"{col}=VALUES({col})" for col in update_columns)
    mysql_query(sock, sql)


def mysql_load_jsonl(sock, table: str, path: Path, columns: list[str], key_update: list[str], batch_size: int) -> int:
    count = 0
    batch: list[dict[str, Any]] = []
    started = time.time()
    for row in read_jsonl(path):
        batch.append(row)
        count += 1
        if len(batch) >= batch_size:
            mysql_insert_rows(sock, table, columns, batch, key_update)
            batch = []
            if count % (batch_size * 20) == 0:
                log(f"MySQL loaded {table}: {count}")
    if batch:
        mysql_insert_rows(sock, table, columns, batch, key_update)
    log(f"MySQL loaded {table}: {count} rows in {round(time.time() - started, 2)}s")
    return count


def cmd_reset_mysql(args: argparse.Namespace) -> None:
    sock = mysql_connect()
    try:
        mysql_query(sock, f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
        mysql_query(sock, f"USE {MYSQL_DATABASE}")
        mysql_query(sock, "SET FOREIGN_KEY_CHECKS=0")
        for table in ("search_results", "search_runs", "api_call_logs", "cost_logs", "ingestion_runs", "eval_sets", "gold_labels", "paper_chunks", "paper_identifiers", "papers", "queries", "datasets"):
            mysql_query(sock, f"DROP TABLE IF EXISTS {table}")
        mysql_query(sock, "SET FOREIGN_KEY_CHECKS=1")
        schema = Path(__file__).with_name("schema.sql")
        for stmt in split_sql(schema.read_text(encoding="utf-8-sig")):
            mysql_query(sock, stmt)
        log("MySQL schema recreated")
    finally:
        sock.close()


def split_sql(sql: str) -> list[str]:
    stmts = []
    buf: list[str] = []
    in_single = in_double = escape = line_comment = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
                buf.append(ch)
            i += 1
            continue
        if not in_single and not in_double and ch == "-" and nxt == "-":
            line_comment = True
            i += 2
            continue
        if escape:
            buf.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            buf = []
            if stmt:
                stmts.append(stmt)
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def cmd_load_mysql(args: argparse.Namespace) -> None:
    sock = mysql_connect()
    try:
        mysql_query(sock, f"USE {MYSQL_DATABASE}")
        mysql_load_jsonl(sock, "queries", PROCESSED_DIR / "queries.jsonl",
                         ["qid", "dataset_name", "split_name", "query_text", "published_time", "answer_count", "source_path"],
                         ["dataset_name", "split_name", "query_text", "published_time", "answer_count", "source_path"], args.batch_size)
        mysql_load_jsonl(sock, "papers", PROCESSED_DIR / "papers.jsonl",
                         ["paper_id", "arxiv_id", "title", "abstract", "year", "published_time", "venue", "authors", "citation_count", "source", "fulltext_key", "has_fulltext"],
                         ["arxiv_id", "title", "abstract", "year", "published_time", "venue", "authors", "citation_count", "source", "fulltext_key", "has_fulltext"], args.paper_batch_size)
        mysql_load_jsonl(sock, "paper_chunks", PROCESSED_DIR / "paper_chunks.jsonl",
                         ["chunk_id", "paper_id", "chunk_index", "chunk_type", "section_title", "text", "token_estimate", "source"],
                         ["paper_id", "chunk_index", "chunk_type", "section_title", "text", "token_estimate", "source"], args.chunk_batch_size)
        mysql_load_jsonl(sock, "gold_labels", PROCESSED_DIR / "gold_labels.jsonl",
                         ["qid", "paper_id", "arxiv_id", "title", "label_rank", "source"],
                         ["arxiv_id", "title", "label_rank", "source"], args.batch_size)
        # eval_sets has JSON array; import after query rows exist.
        mysql_load_jsonl(sock, "eval_sets", PROCESSED_DIR / "eval_sets.jsonl",
                         ["dataset_name", "split_name", "qid", "gold_paper_ids", "published_time"],
                         ["gold_paper_ids", "published_time"], args.batch_size)
    finally:
        sock.close()


def qdrant_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}


def cmd_init_qdrant(args: argparse.Namespace) -> None:
    try:
        http_json("DELETE", f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", headers=qdrant_headers())
        log(f"deleted Qdrant collection {QDRANT_COLLECTION}")
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise
    body = {
        "vectors": {},
        "sparse_vectors": {"text": {}},
        "optimizers_config": {
            "default_segment_number": 2,
            "indexing_threshold": 0,
        },
    }
    http_json("PUT", f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", body, qdrant_headers())
    log(f"created Qdrant collection {QDRANT_COLLECTION}; use load-qdrant for reproducible lexical sparse baseline vectors")


def lexical_sparse_vector(text: str, dimensions: int) -> dict[str, list[int] | list[float]]:
    weights: dict[int, float] = {}
    for token in TOKEN_RE.findall(text.lower()):
        index = zlib.crc32(token.encode("utf-8")) % dimensions
        weights[index] = weights.get(index, 0.0) + 1.0
    if not weights:
        return {"indices": [], "values": []}
    indices = sorted(weights)
    return {"indices": indices, "values": [round(weights[index], 6) for index in indices]}


def qdrant_point(row: dict[str, Any], vector_size: int) -> dict[str, Any]:
    text = str(row.get("text") or "")
    return {
        "id": str(uuid.uuid5(POINT_NAMESPACE, row["chunk_id"])),
        "vector": {"text": lexical_sparse_vector(text, vector_size)},
        "payload": {
            "chunk_id": row["chunk_id"],
            "paper_id": row["paper_id"],
            "chunk_index": row.get("chunk_index"),
            "chunk_type": row.get("chunk_type"),
            "section_title": row.get("section_title"),
            "source": row.get("source"),
            "token_estimate": row.get("token_estimate"),
        },
    }


def qdrant_upsert(points: list[dict[str, Any]], wait: bool = True) -> None:
    wait_value = "true" if wait else "false"
    http_json(
        "PUT",
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points?wait={wait_value}",
        {"points": points},
        qdrant_headers(),
    )


def cmd_load_qdrant(args: argparse.Namespace) -> None:
    count = 0
    batch: list[dict[str, Any]] = []
    started = time.time()
    input_file = Path(args.input_file) if args.input_file else PROCESSED_DIR / "paper_chunks.jsonl"
    for line_number, row in enumerate(read_jsonl(input_file), start=1):
        if line_number < args.start_line:
            continue
        if args.end_line is not None and line_number > args.end_line:
            break
        batch.append(qdrant_point(row, args.vector_size))
        count += 1
        if len(batch) >= args.batch_size:
            qdrant_upsert(batch, wait=args.wait)
            batch = []
            if count % (args.batch_size * 20) == 0:
                log(f"Qdrant loaded {QDRANT_COLLECTION}: {count}")
    if batch:
        qdrant_upsert(batch, wait=args.wait)
    log(f"Qdrant loaded {QDRANT_COLLECTION}: {count} points in {round(time.time() - started, 2)}s")


def cmd_status(args: argparse.Namespace) -> None:
    stats = {
        "config_path": str(CONFIG_PATH),
        "config_exists": CONFIG_PATH.exists(),
        "raw_root": str(RAW_ROOT),
        "processed_dir": str(PROCESSED_DIR),
        "log_dir": str(LOG_DIR),
        "mysql_host": MYSQL_HOST,
        "mysql_database": MYSQL_DATABASE,
        "elasticsearch_url": ES_URL,
        "papers_index": PAPERS_INDEX,
        "chunks_index": CHUNKS_INDEX,
        "qdrant_url": QDRANT_URL,
        "qdrant_collection": QDRANT_COLLECTION,
        "qdrant_api_key_set": bool(QDRANT_API_KEY),
    }
    for rel in ("AutoScholarQuery/train.jsonl", "AutoScholarQuery/dev.jsonl", "AutoScholarQuery/test.jsonl", "RealScholarQuery/test.jsonl", "paper_database/id2paper.json", "paper_database/cs_paper_2nd.zip"):
        path = RAW_ROOT / rel
        stats[rel] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
    if (PROCESSED_DIR / "conversion_stats.json").exists():
        stats["conversion_stats"] = json.loads((PROCESSED_DIR / "conversion_stats.json").read_text(encoding="utf-8"))
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def cmd_verify(args: argparse.Namespace) -> None:
    es_indices = http_json("GET", f"{ES_URL}/_cat/indices/saiti3*?format=json", headers=es_headers())
    qdrant = http_json("GET", f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", headers=qdrant_headers())
    sock = mysql_connect()
    mysql_counts = {}
    try:
        mysql_query(sock, f"USE {MYSQL_DATABASE}")
        for table in ("queries", "gold_labels", "eval_sets", "papers", "paper_chunks"):
            rows = mysql_query(sock, f"SELECT COUNT(*) FROM {table}")
            mysql_counts[table] = int(rows[0][0])
    finally:
        sock.close()
    print(json.dumps({"mysql": mysql_counts, "elasticsearch": es_indices, "qdrant": qdrant.get("result", {})}, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status"); p.set_defaults(func=cmd_status)
    p = sub.add_parser("convert"); p.add_argument("--limit", type=int, default=None); p.set_defaults(func=cmd_convert)
    p = sub.add_parser("reset-mysql"); p.set_defaults(func=cmd_reset_mysql)
    p = sub.add_parser("load-mysql")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--paper-batch-size", type=int, default=500)
    p.add_argument("--chunk-batch-size", type=int, default=100)
    p.set_defaults(func=cmd_load_mysql)
    p = sub.add_parser("reset-es"); p.set_defaults(func=cmd_reset_es)
    p = sub.add_parser("load-es"); p.add_argument("--batch-size", type=int, default=500); p.set_defaults(func=cmd_load_es)
    p = sub.add_parser("init-qdrant"); p.set_defaults(func=cmd_init_qdrant)
    p = sub.add_parser("load-qdrant")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--vector-size", type=int, default=65536)
    p.add_argument("--input-file", default=None, help="JSONL file to import; defaults to processed paper_chunks.jsonl")
    p.add_argument("--start-line", type=int, default=1, help="1-based inclusive JSONL line to start from")
    p.add_argument("--end-line", type=int, default=None, help="1-based inclusive JSONL line to stop at")
    p.add_argument("--no-wait", dest="wait", action="store_false", help="Do not wait for Qdrant to apply each upsert batch")
    p.set_defaults(wait=True)
    p.set_defaults(func=cmd_load_qdrant)
    p = sub.add_parser("verify"); p.set_defaults(func=cmd_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
