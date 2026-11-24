"""Pegamento entre el LLM, el selector y el ejecutor de herramientas (Fase 6.5).

El bucle de voz y el chat de texto usan el mismo `ToolRuntime`:

    pre = rt.before(text)                     # ¿había una escritura pendiente? ("sí" / "no" / otra cosa)
    if pre.outcome: ...                       # era la respuesta a la confirmación: decir pre.outcome.text y listo
    messages, schemas = rt.prepare(messages, text)   # añade la guía y elige las 4 herramientas más parecidas
    for ev in llm.stream_events(messages, tools=schemas): ...   # texto (str) o ToolCall
    out = rt.execute(call, text)              # lectura: respuesta; escritura: propuesta que espera el "sí"

Si algo falla (embeddings caídos, datos ilegibles) la conversación sigue sin herramientas: nunca rompe el chat.
"""
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .prompt import tool_guide
from .unigym.executor import Outcome, UniGymTools


@dataclass
class Pre:
    outcome: Outcome | None = None   # respuesta a la confirmación pendiente (consumed=True): no hay que llamar al LLM
    prefix: str = ""                 # "Descarté lo que estaba pendiente." cuando la frase era otra orden


class ToolRuntime:
    def __init__(self, tools: UniGymTools, selector, today=None, reply_mode: str = "template", warn=print):
        self.tools, self.selector = tools, selector
        self._today = today
        self.reply_mode = reply_mode
        self.warn = warn
        self._warned = False
        self.last_select_s = 0.0
        self.last_selected: list[str] = []

    # ------------------------------ antes del LLM ------------------------------
    def before(self, text: str) -> Pre:
        out = self.tools.answer(text)
        if out is None:
            return Pre()
        if out.consumed:
            return Pre(outcome=out)
        return Pre(prefix=out.text)

    def prepare(self, messages: list[dict], text: str):
        """(mensajes con la guía de herramientas, esquemas) o (mensajes, None) si no hay nada que ofrecer."""
        t0 = time.perf_counter()
        try:
            specs = self.selector.select(text)
        except Exception as e:  # embeddings caídos: se sigue sin herramientas
            if not self._warned:
                self.warn(f"[herramientas] sin selección este turno: {e}")
                self._warned = True
            specs = []
        self.last_select_s = time.perf_counter() - t0
        self.last_selected = [s.name for s in specs]
        if not specs:
            return messages, None
        guide = tool_guide(self._today() if self._today else None)
        first = dict(messages[0], content=messages[0]["content"] + "\n\n" + guide)
        return [first] + list(messages[1:]), [s.ollama_schema() for s in specs]

    # ------------------------------ tras el LLM ------------------------------
    def execute(self, call, heard: str) -> Outcome:
        try:
            return self.tools.run(call.name, call.args, heard)
        except Exception as e:  # una herramienta rota no debe tumbar la conversación
            return Outcome(f"No pude usar esa herramienta: {e}", "error", getattr(call, "name", ""), ok=False)

    @property
    def has_pending(self) -> bool:
        return self.tools.has_pending

    def plain_turn(self) -> None:
        """Turno respondido por el LLM sin herramienta: cambió el tema, se olvida el curso de contexto."""
        self.tools.forget_context()

    def remember_direct(self, history: list[dict], text: str, out: Outcome) -> None:
        """Una lectura resuelta sin LLM («redes», «cómo voy en iot») también va al historial, para que el LLM sepa de qué
        curso se habla ahora (si no, en el turno siguiente pasaba el curso de una charla anterior)."""
        from llm import ToolCall
        if out.tool and out.kind == "answer" and out.ok:
            history.append({"role": "user", "content": text})
            self.remember(history, ToolCall(out.tool, {}), out)

    @staticmethod
    def remember(history: list[dict], call, out: Outcome) -> None:
        """Deja en el historial del LLM solo lo que no se puede imitar mal.

        Bug real (24-nov): al guardar «¿Marco como hecha «EDA2»?» como mensaje del asistente, el LLM lo copió como TEXTO en
        el turno siguiente, dejó de llamar a la herramienta y llegó a decir «se marca como listo» sin guardar nada.
        Por eso: una lectura se guarda como llamada + resultado en el formato nativo de herramientas; las preguntas,
        confirmaciones y errores NO se guardan (la pregunta del usuario se quita; el diálogo lo lleva Python).
        `history[-1]` es el mensaje del usuario recién añadido.
        """
        from .unigym.readers import READERS
        if out.kind == "answer" and out.ok or (out.kind == "ask" and out.tool in READERS):
            history += [{"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": call.name, "arguments": dict(call.args or {})}}]},
                        {"role": "tool", "tool_name": call.name, "content": out.text}]
        elif history and history[-1].get("role") == "user":
            history.pop()


_CLAIM = re.compile(r"\b(se (marca|agrega|guarda|registra|anota|completa)\w*|marqu[eé]|agreg[uú]?[eé]|guard[eé]|registr[eé]|anot[eé]|"
                    r"(est[aá]n?|queda[n]?|qued[oó]) (listo|lista|marcad\w+|guardad\w+|agregad\w+|registrad\w+))\b", re.I)
NOT_SAVED = " Ojo: no guardé nada. Pídemelo de nuevo."


_DATA = re.compile(r"\b(promedio|llevas|nota|notas|parcial|examen|evaluaci\w+|ejercicio|proyecto|necesitas|sacaste)\b", re.I)
NOT_FROM_DATA = " Ojo: eso no lo saqué de tus datos, puede estar mal. Pregúntame de nuevo nombrando el curso."


def invents_data(reply: str, heard: str = "") -> bool:
    """Texto del LLM (sin herramienta) que da notas/promedios con números que Dario NO dijo: no viene de UniGym."""
    nums = set(re.findall(r"\d+(?:[.,]\d+)?", reply or "")) - set(re.findall(r"\d+(?:[.,]\d+)?", heard or ""))
    return bool(nums) and len(_DATA.findall(reply or "")) >= 2


_FAKE_Q = re.compile(r"¿\s*(marco|marcar|agrego|agregar|guardo|guardar|registro|registrar|anoto|anotar|confirmas?)\b", re.I)


def known_context(heard: str, history: list[dict], last: int = 8) -> str:
    """Lo dicho + los resultados recientes de herramientas: números que vienen de ahí NO son inventados."""
    return heard + " " + " ".join(m.get("content", "") for m in history[-last:] if m.get("role") == "tool")


def claims_unbacked_write(reply: str) -> bool:
    """El LLM respondió con TEXTO diciendo que hizo un cambio, o pidiendo confirmar uno, sin llamar a ninguna herramienta:
    es falso, nada se guardó ni hay nada pendiente."""
    return bool(_CLAIM.search(reply or "") or _FAKE_Q.search(reply or ""))


def _connect_semantic_courses(embedder) -> None:
    """Cursos nuevos se reconocen solos: si el nombre no coincide por letras («iot», «la de finanzas»), se compara por
    significado con bge-m3 contra los nombres reales de los cursos (embeddings de los nombres en caché)."""
    from .unigym import names
    cache: dict = {}

    def semantic(query: str, labels: list[str]):
        key = tuple(labels)
        if key not in cache:
            cache.clear()
            cache[key] = embedder.embed(list(labels))
        q = embedder.embed([query])[0]
        sims = sorted(((float(v @ q), i) for i, v in enumerate(cache[key])), reverse=True)
        return sims[0][1], sims[0][0], (sims[1][0] if len(sims) > 1 else 0.0)
    names.SEMANTIC_FALLBACK = semantic


def load_tools(cfg: dict, root, embedder, warn=print) -> ToolRuntime | None:
    """Crea el runtime a partir de `tools:` y `unigym:` de config.yaml. None si está desactivado o no hay datos."""
    tcfg = cfg.get("tools") or {}
    if not tcfg.get("enabled", True):
        return None
    from .selector import ToolSelector
    from .unigym import SPECS
    from .unigym.audit import AuditLog
    from .unigym.confirm import ConfirmationGate
    from .unigym.repo import JsonFileRepository

    root = Path(root)
    ucfg = cfg.get("unigym") or {}
    repo = JsonFileRepository(root / ucfg.get("data_path", "data/unigym.json"), keep=ucfg.get("backups_keep", 30))
    if not repo.exists():
        warn("[herramientas] no hay data/unigym.json; impórtalo con scripts/unigym_import.py. Sigo sin herramientas.")
        return None
    mem_db = (cfg.get("memory") or {}).get("db_path", "data/jarvis_memory.db")
    audit = AuditLog(root / mem_db)
    gate = ConfirmationGate(ttl_s=tcfg.get("confirm_ttl_s", 60))
    unigym = UniGymTools(repo, audit, gate=gate)
    _connect_semantic_courses(embedder)
    sel = tcfg.get("selector") or {}
    selector = ToolSelector(SPECS, embedder, top_k=sel.get("top_k", 4), min_score=sel.get("min_score", 0.50))
    return ToolRuntime(unigym, selector, reply_mode=tcfg.get("reply_mode", "template"), warn=warn)


def load_routines(cfg: dict, root, runtime: "ToolRuntime | None" = None):
    """RoutineRunner a partir de `routines.casa` de config.yaml (comparte repo y registro con las herramientas, si existen)."""
    from .routines import RoutineRunner
    rcfg = ((cfg.get("routines") or {}).get("casa")) or {}
    if not rcfg.get("enabled", True):
        return None
    repo = audit = None
    if runtime is not None:
        repo, audit = runtime.tools.repo, runtime.tools.audit
    return RoutineRunner(rcfg, repo=repo, audit=audit)
