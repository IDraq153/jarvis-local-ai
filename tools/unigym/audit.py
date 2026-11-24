"""Registro de cada llamada a herramienta (Fase 6.4) en la base SQLite de JARVIS.

Tabla `tool_calls` en data/jarvis_memory.db (ignorada por git). Guarda qué se oyó, qué herramienta, con qué
argumentos, en qué estado terminó y qué revisión de los datos produjo. Texto truncado a 500 caracteres.
Estados: read, ask, proposed, confirmed (aplicada), cancelled, expired, rejected, error.
"""
import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls(
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, tool TEXT NOT NULL, level TEXT, status TEXT NOT NULL,
    heard TEXT, args TEXT, result TEXT, revision_before INTEGER, revision_after INTEGER, duration_ms INTEGER);
CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(ts);
"""
_MAX = 500


def _cut(v) -> str | None:
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s[:_MAX]


class AuditLog:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def record(self, tool: str, status: str, *, level: str | None = None, heard=None, args=None, result=None,
               revision_before: int | None = None, revision_after: int | None = None, duration_ms: int | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO tool_calls(ts, tool, level, status, heard, args, result, revision_before, revision_after, duration_ms)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (time.time(), tool, level, status, _cut(heard), _cut(args), _cut(result), revision_before, revision_after, duration_ms))
        self.db.commit()
        return cur.lastrowid

    def tail(self, n: int = 20) -> list[dict]:
        rows = self.db.execute("SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def close(self):
        self.db.close()
