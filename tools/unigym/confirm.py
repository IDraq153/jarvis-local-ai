"""Confirmación por voz de las escrituras (Fase 6.4): propuesta pendiente + interpretación de sí / no.

Regla de seguridad: solo un SÍ claro y corto confirma. Un no, cualquier otra cosa, un cambio de tema o dejar
pasar el tiempo cancelan. Ante la duda, no se escribe.
"""
import time
from dataclasses import dataclass

from memory.text import strip_accents

_CORE_YES = {"si", "claro", "dale", "ok", "okay", "okey", "confirmo", "confirmado", "correcto", "afirmativo",
             "adelante", "hazlo", "exacto", "perfecto", "vale", "guardalo", "agregalo", "anotalo", "registralo",
             "marcalo", "acuerdo", "listo"}
_YES_FILLER = {"por", "favor", "jarvis", "gracias", "es", "asi", "de", "eso", "muy", "bien", "que", "y"}
_NO = {"no", "cancela", "cancelar", "cancelalo", "olvidalo", "negativo", "detente", "alto", "nada", "nunca", "mejor"}
MAX_YES_WORDS = 5


def classify_answer(text: str) -> str:
    """'yes' solo si TODA la frase es una afirmación corta; 'no' si hay una negación; si no, 'other'."""
    s = strip_accents(text or "").lower()
    toks = [t.strip(".,;:!?¡¿\"'()") for t in s.split()]
    toks = [t for t in toks if t]
    if not toks:
        return "other"
    if any(t in _NO for t in toks):
        return "no"
    if len(toks) <= MAX_YES_WORDS and all(t in _CORE_YES or t in _YES_FILLER for t in toks) \
            and any(t in _CORE_YES for t in toks):
        return "yes"
    return "other"


@dataclass
class Pending:
    proposal: object
    created: float
    heard: str = ""
    revision: int = 0
    audit_id: int | None = None
    args: dict | None = None         # argumentos con los que se preparó (para corregir la fecha sin empezar de cero)


@dataclass
class SlotPending:
    """JARVIS preguntó un dato que faltaba (título, curso, fecha...) y espera la respuesta de Dario."""
    tool: str
    args: dict
    slot: str
    created: float
    heard: str = ""
    attempts: int = 1


class ConfirmationGate:
    def __init__(self, ttl_s: float = 60.0, clock=time.monotonic):
        self.ttl_s = ttl_s
        self._clock = clock
        self.pending: Pending | None = None
        self.slot: SlotPending | None = None

    def set_slot(self, tool: str, args: dict, slot: str, heard: str = "", attempts: int = 1) -> None:
        self.pending = None
        self.slot = SlotPending(tool, dict(args), slot, self._clock(), heard, attempts)

    def slot_expired(self) -> bool:
        return bool(self.slot) and self._clock() - self.slot.created > self.ttl_s

    @property
    def active(self) -> bool:
        return self.pending is not None or self.slot is not None

    def set(self, proposal, heard: str = "", revision: int = 0, audit_id=None, args=None) -> None:
        self.slot = None
        self.pending = Pending(proposal, self._clock(), heard, revision, audit_id, dict(args) if args else None)

    def clear(self) -> Pending | None:
        p, self.pending, self.slot = self.pending, None, None
        return p

    def expired(self) -> bool:
        return bool(self.pending) and self._clock() - self.pending.created > self.ttl_s
