from __future__ import annotations

import hashlib
import json
import re
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .io_utils import read_jsonl


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

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MySQLError(Exception):
    pass


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]


def quote_identifier(value: str) -> str:
    if not IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier: {value}")
    return f"`{value}`"


def split_sql(sql: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    in_single = False
    in_double = False
    escape = False
    line_comment = False
    block_comment = False
    i = 0
    while i < len(sql):
        char = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
                buffer.append(char)
            i += 1
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue
        if not in_single and not in_double and char == "-" and nxt == "-":
            line_comment = True
            i += 2
            continue
        if not in_single and not in_double and char == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        if escape:
            buffer.append(char)
            escape = False
            i += 1
            continue
        if char == "\\":
            buffer.append(char)
            escape = True
            i += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        if char == ";" and not in_single and not in_double:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
        else:
            buffer.append(char)
        i += 1
    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


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


class MySQLClient:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        database: str = "",
        *,
        timeout: int = 20,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.timeout = timeout
        self.sock: socket.socket | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "MySQLClient":
        return cls(
            settings.mysql_host,
            settings.mysql_port,
            settings.mysql_username,
            settings.mysql_password,
            settings.mysql_database,
        )

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
        )
        if self.database:
            capabilities |= CLIENT_CONNECT_WITH_DB
        if parsed["plugin"] == "caching_sha2_password":
            token = _scramble_caching_sha2(self.password, parsed["seed"])
        else:
            token = _scramble_native(self.password, parsed["seed"])
        attrs = b""
        for key, value in {"_client_name": "scholar_ingest"}.items():
            key_bytes = key.encode()
            value_bytes = value.encode()
            attrs += _lenenc_int(len(key_bytes)) + key_bytes + _lenenc_int(len(value_bytes)) + value_bytes
        payload = struct.pack("<IIB23s", capabilities, 64 * 1024 * 1024, 255, b"\x00" * 23)
        payload += self.username.encode() + b"\x00" + bytes([len(token)]) + token
        if self.database:
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
        columns: list[str] = []
        for _ in range(column_count):
            _, column_packet = _read_packet(self.sock)
            _check_error(column_packet)
            columns.append(_parse_column_name(column_packet))
        rows: list[list[Any]] = []
        while True:
            _, row_packet = _read_packet(self.sock)
            _check_error(row_packet)
            if row_packet[0] in (0xFE, 0x00) and len(row_packet) < 9:
                break
            rows.append(_parse_row(row_packet, column_count))
        return QueryResult(columns, rows)

    def scalar(self, sql: str) -> Any:
        result = self.execute(sql)
        if not result.rows or not result.rows[0]:
            return None
        return result.rows[0][0]

    def execute_script(self, sql: str) -> int:
        count = 0
        for statement in split_sql(sql):
            self.execute(statement)
            count += 1
        return count

    def use_database(self, database: str) -> None:
        self.execute(f"USE {quote_identifier(database)}")
        self.database = database

    def init_schema(self, schema_path: Path, database: str, *, reset: bool = False) -> dict[str, Any]:
        self.execute(
            f"CREATE DATABASE IF NOT EXISTS {quote_identifier(database)} "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
        )
        self.use_database(database)
        if reset:
            self.drop_known_tables()
        statements = self.execute_script(schema_path.read_text(encoding="utf-8-sig"))
        return {"database": database, "schema_path": str(schema_path), "statements": statements, "reset": reset}

    def drop_known_tables(self) -> None:
        tables = (
            "search_results",
            "search_runs",
            "api_call_logs",
            "cost_logs",
            "ingestion_runs",
            "eval_sets",
            "gold_labels",
            "paper_chunks",
            "paper_identifiers",
            "papers",
            "queries",
            "datasets",
        )
        self.execute("SET FOREIGN_KEY_CHECKS=0")
        for table in tables:
            self.execute(f"DROP TABLE IF EXISTS {quote_identifier(table)}")
        self.execute("SET FOREIGN_KEY_CHECKS=1")

    def insert_rows(
        self,
        table: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        update_columns: list[str] | None = None,
    ) -> None:
        if not rows:
            return
        column_sql = ",".join(quote_identifier(column) for column in columns)
        values = []
        for row in rows:
            values.append("(" + ",".join(sql_value(row.get(column)) for column in columns) + ")")
        statement = f"INSERT INTO {quote_identifier(table)} ({column_sql}) VALUES " + ",".join(values)
        if update_columns:
            updates = ",".join(
                f"{quote_identifier(column)}=VALUES({quote_identifier(column)})" for column in update_columns
            )
            statement += " ON DUPLICATE KEY UPDATE " + updates
        self.execute(statement)

    def load_jsonl(
        self,
        table: str,
        path: Path,
        columns: list[str],
        update_columns: list[str],
        *,
        batch_size: int = 500,
        limit: int | None = None,
    ) -> int:
        count = 0
        batch: list[dict[str, Any]] = []
        for row in read_jsonl(path):
            batch.append(row)
            count += 1
            if len(batch) >= batch_size:
                self.insert_rows(table, columns, batch, update_columns)
                batch = []
            if limit is not None and count >= limit:
                break
        if batch:
            self.insert_rows(table, columns, batch, update_columns)
        return count

    def table_count(self, table: str) -> int:
        value = self.scalar(f"SELECT COUNT(*) FROM {quote_identifier(table)}")
        return int(value or 0)

    def fetch_paper(self, paper_id: str) -> dict[str, Any] | None:
        safe_id = sql_value(paper_id)
        row_json = self.scalar(
            "SELECT JSON_OBJECT("
            "'paper_id', paper_id, 'title', title, 'abstract', abstract, 'year', year, "
            "'venue', venue, 'citation_count', citation_count, 'source', source"
            f") FROM papers WHERE paper_id={safe_id} LIMIT 1"
        )
        if not row_json:
            return None
        return json.loads(str(row_json))


def _read_packet(sock: socket.socket) -> tuple[int, bytes]:
    header = _recv_exact(sock, 4)
    length = header[0] | (header[1] << 8) | (header[2] << 16)
    sequence = header[3]
    return sequence, _recv_exact(sock, length)


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
    if first == 0xFE:
        return struct.unpack("<Q", data[position + 1 : position + 9])[0], position + 9
    raise MySQLError("bad MySQL length-encoded integer")


def _parse_error(payload: bytes) -> str:
    code = struct.unpack("<H", payload[1:3])[0] if len(payload) >= 3 else None
    message_start = 3
    state = ""
    if len(payload) >= 9 and payload[3:4] == b"#":
        state = payload[4:9].decode("ascii", "replace")
        message_start = 9
    message = payload[message_start:].decode("utf-8", "replace")
    return f"ERR {code} {state} {message}".strip()


def _check_error(payload: bytes) -> None:
    if payload and payload[0] == 0xFF:
        raise MySQLError(_parse_error(payload))


def _scramble_native(password: str, seed: bytes) -> bytes:
    if not password:
        return b""
    stage1 = hashlib.sha1(password.encode()).digest()
    stage2 = hashlib.sha1(stage1).digest()
    stage3 = hashlib.sha1(seed + stage2).digest()
    return bytes(left ^ right for left, right in zip(stage1, stage3))


def _scramble_caching_sha2(password: str, seed: bytes) -> bytes:
    if not password:
        return b""
    stage1 = hashlib.sha256(password.encode()).digest()
    stage2 = hashlib.sha256(stage1).digest()
    stage3 = hashlib.sha256(stage2 + seed).digest()
    return bytes(left ^ right for left, right in zip(stage1, stage3))


def _parse_handshake(payload: bytes) -> dict[str, Any]:
    position = 1
    server_version, position = _nul_split(payload, position)
    position += 4
    auth1 = payload[position : position + 8]
    position += 9
    position += 2 + 1 + 2 + 2
    auth_length = payload[position]
    position += 1 + 10
    auth2_length = max(13, auth_length - 8) if auth_length else 13
    auth2 = payload[position : position + auth2_length]
    position += auth2_length
    plugin = "mysql_native_password"
    if position < len(payload):
        plugin_raw, _ = _nul_split(payload, position)
        if plugin_raw:
            plugin = plugin_raw.decode("ascii", "replace")
    return {
        "server_version": server_version.decode("utf-8", "replace"),
        "seed": (auth1 + auth2).split(b"\x00")[0],
        "plugin": plugin,
    }


def _parse_column_name(packet: bytes) -> str:
    position = 0
    values: list[str] = []
    for _ in range(6):
        length, position = _read_lenenc(packet, position)
        if length is None:
            values.append("")
        else:
            values.append(packet[position : position + length].decode("utf-8", "replace"))
            position += length
    return values[4] if len(values) >= 5 else ""


def _parse_row(packet: bytes, column_count: int) -> list[Any]:
    position = 0
    values: list[Any] = []
    for _ in range(column_count):
        length, position = _read_lenenc(packet, position)
        if length is None:
            values.append(None)
            continue
        values.append(packet[position : position + length].decode("utf-8", "replace"))
        position += length
    return values
