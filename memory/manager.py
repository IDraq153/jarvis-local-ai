"""Memoria persistente de JARVIS: recuerda entre sesiones.

Tres capas, todas en un SQLite local:
  - hechos: datos duraderos sobre Dario ("recuerda que...", o extraídos al cerrar la sesión);
  - recuerdos: cada turno (pregunta + respuesta) con su vector, para evocar charlas pasadas;
  - resumen de la última sesión.
En cada turno se buscan los más parecidos a lo que dice Dario y se añaden al prompt de
sistema (sin tocar el historial). La memoria nunca debe romper el chat: si falla el
embedder se avisa una vez y se busca por palabras.
"""
import time
from dataclasses import dataclass, fields

from llm import LLMError

from .commands import parse_memory_command
from .embeddings import EmbedError, Embedder
from .extract import build_messages, parse_extraction
from .store import Fact, MemoryStore
from .text import content_words, looks_secret


@dataclass
class MemoryConfig:
    top_facts: int = 5
    top_episodes: int = 2
    min_fact_score: float = 0.45      # coseno mínimo para evocar un hecho
    min_episode_score: float = 0.55   # más exigente: un recuerdo ajeno confunde más que ayuda
    dedupe_score: float = 0.90        # por encima, un hecho nuevo actualiza al antiguo
    forget_score: float = 0.60        # parecido mínimo para borrar con "olvida ..."
    max_block_chars: int = 1200       # ~300 tokens del contexto de 4096
    episode_chars: int = 600
    summarize_on_exit: bool = True
    min_turns_to_summarize: int = 2

    @classmethod
    def from_dict(cls, d: dict | None) -> "MemoryConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def _ago(then: float, now: float) -> str:
    days = int((now - then) // 86400)
    if days <= 0:
        return "hoy"
    return "ayer" if days == 1 else f"hace {days} días"


class Memory:
    def __init__(self, store: MemoryStore, embedder: Embedder, cfg: MemoryConfig | None = None,
                 clock=time.time, warn=print):
        self.store, self.embedder = store, embedder
        self.cfg = cfg or MemoryConfig()
        self.clock, self.warn = clock, warn
        self.session_id: int | None = None
        self.last_block = ""
        self.last_retrieval_s = 0.0
        self._warned = False
        self._prev_summary: tuple[str, float] | None = None

    # ---------- ciclo de vida ----------
    def start_session(self) -> dict:
        self.session_id = self.store.start_session(self.clock())
        self._prev_summary = self.store.last_summary(self.session_id)
        self.reindex()
        return self.stats()

    def stats(self) -> dict:
        return {"facts": self.store.count("facts"), "turns": self.store.count("turns"),
                "sessions": self.store.count("sessions") - (1 if self.session_id else 0)}

    def warmup(self) -> float | None:
        """Carga el modelo de embeddings (la primera llamada es lenta). Segundos, o None si falla."""
        t0 = time.perf_counter()
        return time.perf_counter() - t0 if self._embed(["hola"]) is not None else None

    def reindex(self) -> int:
        """Embebe lo que no tiene vector o lo tiene de otro modelo. Devuelve cuántas filas."""
        n = 0
        for table in ("facts", "turns"):
            pending = self.store.pending_embeddings(table, self.embedder.model)
            for i in range(0, len(pending), 16):
                chunk = pending[i:i + 16]
                vecs = self._embed([t for _, t in chunk])
                if vecs is None:
                    return n
                for (rid, _), v in zip(chunk, vecs):
                    self.store.set_embedding(table, rid, v, self.embedder.model)
                    n += 1
        return n

    # ---------- por turno ----------
    def augment(self, messages: list[dict], query: str) -> list[dict]:
        """Copia de `messages` con el bloque de memoria añadido al prompt de sistema."""
        t0 = time.perf_counter()
        try:
            block = self._build_block(query)
        except Exception as e:  # la memoria nunca debe tumbar la conversación
            self._warn_once(f"[memoria] error al buscar: {e}")
            block = ""
        self.last_retrieval_s = time.perf_counter() - t0
        self.last_block = block
        if not block:
            return messages
        first = dict(messages[0], content=messages[0]["content"] + "\n\n" + block)
        return [first] + list(messages[1:])

    def record_turn(self, user: str, reply: str) -> None:
        if self.session_id is None or not user.strip() or not reply.strip():
            return
        try:
            text = f"{user}\n{reply}"[:self.cfg.episode_chars]
            vecs = self._embed([text])
            vec = vecs[0] if vecs is not None else None
            self.store.add_turn(self.session_id, user, reply, vec, self.embedder.model, self.clock())
        except Exception as e:
            self._warn_once(f"[memoria] no pude guardar el turno: {e}")

    # ---------- hechos ----------
    def remember(self, text: str, source: str = "explicit") -> tuple[str, Fact | None]:
        text = " ".join(text.split())
        if not text:
            return "empty", None
        if looks_secret(text):
            return "secret", None
        vecs = self._embed([text])
        vec = vecs[0] if vecs is not None else None
        return self.store.add_fact(text, vec, self.embedder.model, source, self.session_id,
                                   self.cfg.dedupe_score, self.clock())

    def facts(self) -> list[Fact]:
        return self.store.all_facts()

    def forget_id(self, fid: int) -> bool:
        return self.store.delete_fact(fid)

    def forget_matching(self, text: str) -> Fact | None:
        vecs = self._embed([text])
        hits = self.store.search_facts(vecs[0] if vecs is not None else None, text,
                                       self.embedder.model, 1, self.cfg.forget_score)
        if not hits:
            return None
        fact = hits[0][1]
        self.store.delete_fact(fact.id)
        return fact

    def handle_command(self, text: str) -> str | None:
        """Si `text` es una orden de memoria, la ejecuta y devuelve lo que JARVIS responde."""
        cmd = parse_memory_command(text)
        if cmd is None:
            return None
        kind, arg = cmd
        if kind == "remember":
            status, _ = self.remember(arg)
            return {"added": "Anotado, Dario.", "updated": "Actualizado, Dario.",
                    "secret": "Prefiero no guardar contraseñas ni números de documento o de tarjeta.",
                    "empty": "No entendí qué debo recordar."}[status]
        if kind == "forget":
            fact = self.forget_matching(arg)
            return f"Olvidado: {fact.text}" if fact else "No encuentro nada parecido en mi memoria."
        facts = self.facts()
        if not facts:
            return "Todavía no sé nada sobre ti, Dario."
        recent = "; ".join(f.text for f in facts[:3])
        if len(facts) == 1:
            return f"Sé una cosa sobre ti: {recent}."
        return f"Sé {len(facts)} cosas sobre ti. Las más recientes: {recent}."

    # ---------- cierre ----------
    def close_session(self, llm) -> dict | None:
        """Resume la sesión y extrae hechos con el LLM. Devuelve {resumen, nuevos, actualizados} o None."""
        if self.session_id is None:
            return None
        sid, self.session_id = self.session_id, None
        turns = self.store.session_turns(sid)
        summary, result = "", None
        if self.cfg.summarize_on_exit and len(turns) >= self.cfg.min_turns_to_summarize:
            try:
                raw = llm.chat(build_messages(turns), temperature=0.1)
                summary, facts = parse_extraction(raw)
                added = updated = 0
                self.session_id = sid  # para etiquetar los hechos con su sesión
                for f in facts:
                    status, _ = self.remember(f, source="auto")
                    added += status == "added"
                    updated += status == "updated"
                self.session_id = None
                result = {"resumen": summary, "nuevos": added, "actualizados": updated}
            except LLMError as e:
                self.warn(f"[memoria] no pude resumir la sesión: {e}")
        self.store.end_session(sid, summary or None, self.clock())
        return result

    # ---------- internos ----------
    def _embed(self, texts: list[str]):
        try:
            return self.embedder.embed(texts)
        except EmbedError as e:
            self._warn_once(f"[memoria] sin embeddings ({e}); busco solo por palabras.")
            return None

    def _warn_once(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            self.warn(msg)

    def _build_block(self, query: str) -> str:
        has_prev = self._prev_summary is not None
        if len(content_words(query)) < 2 and not has_prev:
            return ""
        now = self.clock()
        facts, episodes = [], []
        if len(content_words(query)) >= 2:
            vecs = self._embed([query])
            qv = vecs[0] if vecs is not None else None
            model = self.embedder.model
            facts = [f for _, f in self.store.search_facts(qv, query, model, self.cfg.top_facts,
                                                          self.cfg.min_fact_score)]
            episodes = [t for _, t in self.store.search_turns(qv, query, model, self.cfg.top_episodes,
                                                              self.cfg.min_episode_score, self.session_id)]
        parts, budget = [], self.cfg.max_block_chars
        if facts:
            lines = []
            for f in facts:
                if budget - len(f.text) < 0:
                    break
                lines.append(f"- {f.text}")
                budget -= len(f.text) + 3
            if lines:
                parts.append("Lo que sabes de Dario (texto en primera persona = palabras de Dario):\n"
                             + "\n".join(lines))
        if episodes:
            lines = []
            for t in episodes:
                snippet = f"({_ago(t.ts, now)}) Dario: {t.user_text[:150]} / JARVIS: {t.reply_text[:150]}"
                if budget - len(snippet) < 0:
                    break
                lines.append(f"- {snippet}")
                budget -= len(snippet) + 3
            if lines:
                parts.append("Conversaciones pasadas relacionadas:\n" + "\n".join(lines))
        if has_prev:
            text, ended = self._prev_summary
            text = text[:min(300, max(budget, 0))]
            if text:
                parts.append(f"Última sesión ({_ago(ended, now)}): {text}")
        if not parts:
            return ""
        return ("MEMORIA (úsala solo si viene al caso; no la recites ni digas que la consultaste):\n"
                + "\n\n".join(parts))
