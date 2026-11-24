"""Fechas habladas -> fecha ISO, resueltas en Python (el LLM nunca calcula fechas).

Entiende lo que Whisper suele devolver: "hoy", "mañana", "pasado mañana", "el viernes",
"el próximo lunes", "el 28", "28 de noviembre", "28/11", "2026-11-28", "veintiocho de noviembre".
Cuando algo es dudoso (p. ej. "el viernes" dicho un viernes) no adivina: devuelve `ambiguous`
con la pregunta que JARVIS debe hacer.
"""
import datetime as dt
import re
from dataclasses import dataclass

from memory.text import strip_accents

DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre"]
_WEEKDAY = {strip_accents(d): i for i, d in enumerate(DAYS)}
_MONTH = {strip_accents(m): i + 1 for i, m in enumerate(MONTHS)} | {"setiembre": 9}
_UNITS = {"uno": 1, "un": 1, "primero": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
          "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
          "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
          "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
          "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30}
_FILLER = {"el", "la", "para", "este", "esta", "del", "dia", "de", "en", "al", "a", "pa"}


class DateError(ValueError):
    """La expresión no se entiende o no es válida; el mensaje se puede decir en voz alta."""


@dataclass
class DateResult:
    iso: str
    ambiguous: str | None = None   # pregunta para Dario si no hay forma segura de decidir
    defaulted: bool = False        # no dijo fecha: se usó hoy


def _tokens(expr: str) -> list[str]:
    s = strip_accents(expr).lower()
    s = re.sub(r"[¿?¡!,;:\"'()]", " ", s)
    s = re.sub(r"(?<=\d)\.(?=\D|$)", " ", s)
    return s.split()


def _num(tok: str) -> int | None:
    if tok.isdigit():
        return int(tok)
    return _UNITS.get(tok)


def _build(y: int, m: int, d: int) -> dt.date:
    try:
        return dt.date(y, m, d)
    except ValueError:
        raise DateError(f"El {d} de {MONTHS[m - 1]} no existe.") from None


def resolve_date(expr, today: dt.date | None = None, *, same_weekday: str = "ask",
                 allow_past: bool = True) -> DateResult:
    """Convierte lo que dijo Dario en una fecha.

    same_weekday: qué hacer con "el viernes" dicho un viernes: "ask" (preguntar), "today" o "next".
    allow_past: False para fechas límite de tareas (una fecha pasada suele ser un error del STT).
    """
    today = today or dt.date.today()
    if expr is None or not str(expr).strip():
        return DateResult(today.isoformat(), defaulted=True)
    raw = str(expr).strip()
    toks = _tokens(raw)
    if not toks:
        return DateResult(today.isoformat(), defaulted=True)
    words = [t for t in toks if t not in _FILLER] or toks
    text = " ".join(words)
    res = _resolve(text, words, raw, today, same_weekday)
    if not allow_past and dt.date.fromisoformat(res.iso) < today:
        raise DateError(f"{spoken_date(res.iso, today)} ya pasó. ¿Qué fecha quieres?")
    return res


def _resolve(text, words, raw, today, same_weekday) -> DateResult:
    iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw.strip())
    if iso:
        return DateResult(_build(*map(int, iso.groups())).isoformat())
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", text)
    if m:
        d, mo, y = int(m[1]), int(m[2]), m[3]
        year = (int(y) + (2000 if len(y) == 2 else 0)) if y else None
        return DateResult(_next_or_exact(d, mo, year, today))
    if text in ("hoy", "ahora", "hoy dia"):
        return DateResult(today.isoformat())
    if text == "manana":
        return DateResult((today + dt.timedelta(days=1)).isoformat())
    if text in ("pasado manana", "pasadomanana"):
        return DateResult((today + dt.timedelta(days=2)).isoformat())
    if text == "ayer":
        return DateResult((today - dt.timedelta(days=1)).isoformat())
    # día de la semana: "viernes", "proximo viernes", "viernes que viene"
    wd = [w for w in words if w in _WEEKDAY]
    if len(wd) == 1:
        target = _WEEKDAY[wd[0]]
        nxt = any(w in ("proximo", "proxima", "siguiente", "viene") for w in words)
        delta = (target - today.weekday()) % 7
        if delta == 0:
            if nxt or same_weekday == "next":
                delta = 7
            elif same_weekday == "ask":
                return DateResult(today.isoformat(), ambiguous=f"Hoy ya es {DAYS[target]}. ¿Hoy o el {DAYS[target]} de la próxima semana?")
        elif nxt and delta <= 0:
            delta += 7
        return DateResult((today + dt.timedelta(days=delta)).isoformat())
    # "28 de noviembre [de 2026]" / "veintiocho de noviembre" / "noviembre 28"
    mo_idx = [i for i, w in enumerate(words) if w in _MONTH]
    if len(mo_idx) == 1:
        i = mo_idx[0]
        nums = [n for n in (_num(w) for w in words) if n is not None]
        day = next((n for n in nums if 1 <= n <= 31), None)
        year = next((n for n in nums if n >= 2000), None)
        if day is None:
            raise DateError("No entendí el día del mes.")
        return DateResult(_next_or_exact(day, _MONTH[words[i]], year, today))
    # "el 28"
    if len(words) == 1 and _num(words[0]) is not None and 1 <= _num(words[0]) <= 31:
        d = _num(words[0])
        for k in range(0, 14):  # el próximo mes que tenga ese día
            y, mo = today.year + (today.month - 1 + k) // 12, (today.month - 1 + k) % 12 + 1
            try:
                cand = dt.date(y, mo, d)
            except ValueError:
                continue
            if cand >= today:
                return DateResult(cand.isoformat())
    raise DateError(f"No entendí la fecha «{raw}».")


def _next_or_exact(d: int, mo: int, year: int | None, today: dt.date) -> str:
    if year is not None:
        return _build(year, mo, d).isoformat()
    cand = _build(today.year, mo, d)
    if cand < today - dt.timedelta(days=7):   # "28 de febrero" dicho en noviembre: el del año que viene
        cand = _build(today.year + 1, mo, d)
    return cand.isoformat()


def spoken_date(iso: str, today: dt.date | None = None) -> str:
    """'hoy', 'mañana', 'ayer' o 'viernes 28 de noviembre' (para decirlo y confirmarlo)."""
    today = today or dt.date.today()
    d = dt.date.fromisoformat(iso)
    diff = (d - today).days
    if diff == 0:
        return "hoy"
    if diff == 1:
        return "mañana"
    if diff == -1:
        return "ayer"
    return f"{DAYS[d.weekday()]} {d.day} de {MONTHS[d.month - 1]}"


_NEXT = {"proximo", "proxima", "siguiente"}


def find_date_phrase(text: str, today: dt.date | None = None) -> str | None:
    """Fecha dicha dentro de una frase más larga («no, mejor para este sábado probablemente» -> «este sábado»).

    Solo reconoce día de la semana, mes, «mañana», «hoy», «pasado mañana» y dd/mm; no números sueltos («tengo 3 tareas»).
    """
    toks = _tokens(text or "")
    key = [i for i, t in enumerate(toks) if t in _WEEKDAY or t in _MONTH or t in ("manana", "hoy")
           or re.fullmatch(r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?", t)]
    if not key:
        return None
    today = today or dt.date.today()
    for size in range(1, 6):
        for start in range(max(0, key[0] - size + 1), min(len(toks) - size, key[-1]) + 1):
            if not any(start <= k < start + size for k in key):
                continue
            window = toks[start:start + size]
            if window[0] in _FILLER and size > 1:
                continue
            try:
                resolve_date(" ".join(window), today, same_weekday="ask")
            except DateError:
                continue
            if start > 0 and toks[start - 1] in _NEXT | {"pasado"}:
                window = [toks[start - 1]] + window
            return " ".join(window)
    return None
