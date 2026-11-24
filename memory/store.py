"""Almacén SQLite de la memoria de JARVIS: sesiones, turnos (recuerdos) y hechos.

Los vectores viven en la propia base como BLOB float32 y la búsqueda es por fuerza bruta
con numpy (coseno = producto punto, vectores normalizados). Para una memoria personal
(miles de filas) es instantáneo y no exige extensiones nativas en Windows.
Cada vector guarda el modelo que lo generó (`embed_model`): si cambias de modelo, los
antiguos se ignoran hasta `Memory.reindex()`. Sin vector (Ollama caído) se busca por palabras.
"""
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .text import content_words, words

LEXICAL_MIN = 0.4  # umbral de la búsqueda por palabras (sin embeddings)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY, started REAL NOT NULL, ended REAL, summary TEXT, turns INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS facts(
    id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL, session_id INTEGER,
    created REAL NOT NULL, updated REAL NOT NULL, embedding BLOB, embed_model TEXT);
CREATE TABLE IF NOT EXISTS turns(
    id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, ts REAL NOT NULL,
    user_text TEXT NOT NULL, reply_text TEXT NOT NULL, embedding BLOB, embed_model TEXT);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
"""


@dataclass
class Fact:
    id: int
    text: str
    source: str
    created: float
    updated: float


@dataclass
class TurnRec:
    id: int
    session_id: int
    ts: float
    user_text: str
    reply_text: str


def _pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


class MemoryStore:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        self.db.execute("PRAGMA user_version = 1")
        self.db.commit()
        self._cache: dict = {}  # (tabla, modelo) -> {"ids", "sess", "mat"}

    def close(self):
        self.db.close()

    # ---------- sesiones ----------
    def start_session(self, now: float | None = None) -> int:
        cur = self.db.execute("INSERT INTO sessions(started) VALUES (?)", (now or time.time(),))
        self.db.commit()
        return cur.lastrowid

    def end_session(self, sid: int, summary: str | None, now: float | None = None) -> None:
        self.db.execute("UPDATE sessions SET ended=?, summary=? WHERE id=?",
                        (now or time.time(), summary or None, sid))
        self.db.commit()

    def last_summary(self, exclude_session: int | None = None) -> tuple[str, float] | None:
        """(resumen, momento en que terminó) de la última sesión resumida."""
        row = self.db.execute(
            "SELECT summary, ended FROM sessions WHERE summary IS NOT NULL AND id IS NOT ? "
            "ORDER BY id DESC LIMIT 1", (exclude_session,)).fetchone()
        return (row["summary"], row["ended"]) if row else None

    # ---------- turnos (recuerdos) ----------
    def add_turn(self, sid: int, user: str, reply: str, vec: np.ndarray | None = None,
                 model: str | None = None, now: float | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO turns(session_id, ts, user_text, reply_text, embedding, embed_model) "
            "VALUES (?,?,?,?,?,?)",
            (sid, now or time.time(), user, reply,
             _pack(vec) if vec is not None else None, model if vec is not None else None))
        self.db.execute("UPDATE sessions SET turns = turns + 1 WHERE id=?", (sid,))
        self.db.commit()
        if vec is not None:
            self._append("turns", model, cur.lastrowid, sid, vec)
        return cur.lastrowid

    def session_turns(self, sid: int) -> list[TurnRec]:
        rows = self.db.execute("SELECT * FROM turns WHERE session_id=? ORDER BY id", (sid,)).fetchall()
        return [self._turn(r) for r in rows]

    # ---------- hechos ----------
    def add_fact(self, text: str, vec: np.ndarray | None, model: str | None, source: str,
                 session_id: int | None, dedupe_score: float = 0.90,
                 now: float | None = None) -> tuple[str, Fact]:
        """Guarda un hecho. Si ya hay uno casi igual lo ACTUALIZA (gana el más nuevo):
        así 'ahora vivo en X' corrige al antiguo en vez de duplicarlo."""
        now = now or time.time()
        dup_id = None
        if vec is not None:
            hit = self._search_vec("facts", vec, model, 1, dedupe_score)
            dup_id = hit[0][1] if hit else None
        if dup_id is None:
            key = " ".join(words(text))
            for r in self.db.execute("SELECT id, text FROM facts"):
                if " ".join(words(r["text"])) == key:
                    dup_id = r["id"]
                    break
        if dup_id is not None:
            self.db.execute("UPDATE facts SET text=?, updated=?, embedding=?, embed_model=? WHERE id=?",
                            (text, now, _pack(vec) if vec is not None else None,
                             model if vec is not None else None, dup_id))
            self.db.commit()
            self._cache.pop(("facts", model), None)
            return "updated", self.get_fact(dup_id)
        cur = self.db.execute(
            "INSERT INTO facts(text, source, session_id, created, updated, embedding, embed_model) "
            "VALUES (?,?,?,?,?,?,?)",
            (text, source, session_id, now, now,
             _pack(vec) if vec is not None else None, model if vec is not None else None))
        self.db.commit()
        self._cache.pop(("facts", model), None)
        return "added", self.get_fact(cur.lastrowid)

    def get_fact(self, fid: int) -> Fact | None:
        r = self.db.execute("SELECT * FROM facts WHERE id=?", (fid,)).fetchone()
        return self._fact(r) if r else None

    def all_facts(self) -> list[Fact]:
        rows = self.db.execute("SELECT * FROM facts ORDER BY updated DESC, id DESC").fetchall()
        return [self._fact(r) for r in rows]

    def delete_fact(self, fid: int) -> bool:
        cur = self.db.execute("DELETE FROM facts WHERE id=?", (fid,))
        self.db.commit()
        self._cache.clear()
        return cur.rowcount > 0

    # ---------- búsqueda ----------
    def search_facts(self, qvec, qtext: str, model: str | None, k: int,
                     min_score: float) -> list[tuple[float, Fact]]:
        if qvec is not None:
            hits = self._search_vec("facts", qvec, model, k, min_score)
        else:
            rows = self.db.execute("SELECT id, text FROM facts").fetchall()
            hits = self._search_lex(qtext, [(r["id"], r["text"]) for r in rows], k)
        return [(s, self.get_fact(i)) for s, i in hits]

    def search_turns(self, qvec, qtext: str, model: str | None, k: int, min_score: float,
                     exclude_session: int | None = None) -> list[tuple[float, TurnRec]]:
        if qvec is not None:
            hits = self._search_vec("turns", qvec, model, k, min_score, exclude_session)
        else:
            rows = self.db.execute(
                "SELECT id, user_text || ' ' || reply_text AS t FROM turns WHERE session_id IS NOT ?",
                (exclude_session,)).fetchall()
            hits = self._search_lex(qtext, [(r["id"], r["t"]) for r in rows], k)
        out = []
        for s, i in hits:
            r = self.db.execute("SELECT * FROM turns WHERE id=?", (i,)).fetchone()
            out.append((s, self._turn(r)))
        return out

    # ---------- reindexado ----------
    def pending_embeddings(self, table: str, model: str) -> list[tuple[int, str]]:
        """Filas sin vector o con el de otro modelo: (id, texto a embeber)."""
        text = "text" if table == "facts" else "user_text || char(10) || reply_text"
        rows = self.db.execute(
            f"SELECT id, {text} AS t FROM {table} "
            f"WHERE embedding IS NULL OR embed_model IS NOT ? ORDER BY id", (model,)).fetchall()
        return [(r["id"], r["t"]) for r in rows]

    def set_embedding(self, table: str, rid: int, vec: np.ndarray, model: str) -> None:
        self.db.execute(f"UPDATE {table} SET embedding=?, embed_model=? WHERE id=?",
                        (_pack(vec), model, rid))
        self.db.commit()
        self._cache.clear()

    def count(self, table: str) -> int:
        return self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ---------- internos ----------
    @staticmethod
    def _fact(r) -> Fact:
        return Fact(r["id"], r["text"], r["source"], r["created"], r["updated"])

    @staticmethod
    def _turn(r) -> TurnRec:
        return TurnRec(r["id"], r["session_id"], r["ts"], r["user_text"], r["reply_text"])

    def _index(self, table: str, model: str) -> dict:
        key = (table, model)
        if key not in self._cache:
            sid = "session_id" if table == "turns" else "COALESCE(session_id, -1)"
            rows = self.db.execute(
                f"SELECT id, {sid} AS sid, embedding FROM {table} "
                f"WHERE embed_model=? AND embedding IS NOT NULL ORDER BY id", (model,)).fetchall()
            mat = (np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                   if rows else None)
            self._cache[key] = {"ids": [r["id"] for r in rows], "sess": [r["sid"] for r in rows], "mat": mat}
        return self._cache[key]

    def _append(self, table: str, model: str, rid: int, sid: int, vec: np.ndarray) -> None:
        idx = self._cache.get((table, model))
        if idx is None:
            return  # se cargará completo en la próxima búsqueda
        v = np.asarray(vec, dtype=np.float32)[None, :]
        idx["mat"] = v if idx["mat"] is None else np.vstack([idx["mat"], v])
        idx["ids"].append(rid)
        idx["sess"].append(sid)

    def _search_vec(self, table: str, qvec, model: str, k: int, min_score: float,
                    exclude_session: int | None = None) -> list[tuple[float, int]]:
        idx = self._index(table, model)
        if idx["mat"] is None:
            return []
        scores = idx["mat"] @ np.asarray(qvec, dtype=np.float32)
        if exclude_session is not None:
            scores = np.where(np.array(idx["sess"]) == exclude_session, -1.0, scores)
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), idx["ids"][i]) for i in order if scores[i] >= min_score]

    @staticmethod
    def _search_lex(qtext: str, rows: list[tuple[int, str]], k: int) -> list[tuple[float, int]]:
        q = set(content_words(qtext))
        if not q:
            return []
        scored = [(len(q & set(content_words(t))) / len(q), rid) for rid, t in rows]
        return sorted((x for x in scored if x[0] >= LEXICAL_MIN), reverse=True)[:k]
