"""Herramientas de ESCRITURA de UniGym (Fase 6.4): PREPARAN una propuesta, nunca guardan solas.

`prepare_*` valida lo que dijo Dario (el STT es ruidoso) y devuelve:
- `Proposal`: la pregunta de confirmación ("¿Agrego la tarea «EDA2: informe» para el viernes 27 de noviembre?"),
  el texto de éxito y `apply(estado)`, que modifica el estado igual que lo haría la web (mismos campos y formatos);
- o un `Result` si falta/está mal un dato (pregunta aclaratoria) o ya estaba hecho (nada que confirmar).
`apply` se ejecuta dentro del mutador del repositorio, sobre el estado más reciente, y busca por id: si lo que se
quería cambiar desapareció entre la pregunta y el sí, falla sin guardar.
"""
import datetime as dt
import random
import re
import string
import time
from dataclasses import dataclass
from typing import Callable

from memory.text import strip_accents

from . import names
from .dates import DateError, resolve_date, spoken_date
from .readers import Result, _course_or_ask, _join, _s, fmt
from .repo import UniGymDataError
from .stats import js_round, num

TASK_TITLE_MAX = 120
WEIGHT_RANGE = (30.0, 250.0)       # kg; más estricto que la web (que acepta hasta 300): un 7 o un 700 es un error del STT
LIMITS = {"steps": 100_000, "sleep": 16.0, "water": 40}
BACKFILL_DAYS = 7                  # cuántos días atrás se puede registrar algo hablando
_ALNUM36 = string.digits + string.ascii_lowercase


@dataclass
class Proposal:
    tool: str
    question: str
    done: str
    apply: Callable[[dict], None]
    data: dict


def new_id(prefix: str, now_ms: int | None = None) -> str:
    """Mismo formato que uid() de la web: prefijo_ + tiempo en base 36 + 5 caracteres al azar."""
    n, out = int(now_ms if now_ms is not None else time.time() * 1000), ""
    while n:
        n, r = divmod(n, 36)
        out = _ALNUM36[r] + out
    return f"{prefix}_{out}{''.join(random.choices(_ALNUM36, k=5))}"


def to_number(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _day(args, key, today, *, past_days=BACKFILL_DAYS):
    """(iso, None) o (None, Result con la pregunta). Sin fecha: hoy. No acepta futuro ni fechas muy viejas."""
    try:
        r = resolve_date(_s(args, key), today, same_weekday="today")
    except DateError as e:
        return None, Result.ask(str(e))
    d = dt.date.fromisoformat(r.iso)
    if d > today:
        return None, Result.ask("Solo puedo registrar hoy o días pasados. ¿Qué día es?")
    if (today - d).days > past_days:
        return None, Result.ask(f"Eso es de hace más de {past_days} días. Dime la fecha otra vez por favor.")
    return r.iso, None


# ------------------------------- crear_tarea -------------------------------

def _ask(question: str, slot: str) -> Result:
    """Pregunta aclaratoria que además dice QUÉ dato falta: la respuesta siguiente de Dario lo rellena (executor)."""
    r = Result.ask(question)
    r.data = {"slot": slot}
    return r


_NADA = {"sin fecha", "sin curso", "ninguna", "ninguno", "no", "sin plazo", "sin dia", "sin", "personal", "nada"}


def is_none_answer(text) -> bool:
    return " ".join(strip_accents(str(text or "")).lower().replace(",", " ").replace(".", " ").split()) in _NADA


_LEAD = re.compile(r"^(?:(?:una|la|mi|esa|nueva)\s+)*(?:tarea|pendiente)\s*(?:de|para|sobre|:)?\s*", re.I)


def _mentions_course(titulo: str, course: dict, query: str | None) -> bool:
    """¿El título ya nombra al curso? (por siglas, por nombre completo o por una palabra significativa)."""
    t_compact = names.compact(titulo)
    toks = names.tokens(titulo)
    units = set(toks) | {a + b for a, b in zip(toks, toks[1:])}          # "eda 2" -> "eda2"
    acr = names.acronym(course.get("name", ""))
    if acr and acr in units:                                              # siglas: solo como palabra entera ("rc" no está en "parcial")
        return True
    if any(c and len(c) >= 4 and c in t_compact for c in (names.compact(course.get("name", "")), names.compact(query or ""))):
        return True
    words = {w for w in names.tokens(course.get("name", "")) if len(w) >= 4 and w not in names._STOP}
    return bool(words & set(names.tokens(titulo)))


def _course_label(course: dict, query: str | None) -> str:
    q = names.compact(query or "")
    if q and q == names.acronym(course.get("name", "")):
        return q.upper()                          # dijo "EDA 2" -> "EDA2"
    return course.get("name", "?")


def prepare_crear_tarea(state, args, today, now_ms=None):
    titulo = _s(args, "titulo")
    curso_q = _s(args, "curso")
    course = None
    if curso_q and not is_none_answer(curso_q):
        course, ask = _course_or_ask(state, curso_q)
        if ask:
            return _ask(ask.text, "curso")
    if titulo:
        titulo = _LEAD.sub("", titulo).strip(" .,:;")
    if titulo and course and names.compact(titulo) in (names.compact(course.get("name", "")), names.acronym(course.get("name", ""))):
        titulo = ""                                # "tarea de EDA2" no es un título: es solo el curso
    if not titulo and not course:
        return _ask("¿Qué tarea agrego?", "titulo")
    if titulo and len(titulo) > TASK_TITLE_MAX:
        return _ask("Ese título es muy largo. ¿Me lo dices más corto?", "titulo")
    if titulo and not course and not args.get("sin_curso") and not is_none_answer(curso_q):
        return _ask("¿Es de algún curso? Dime cuál, o «ninguno».", "curso")
    fecha_q = _s(args, "fecha")
    if not titulo and course and not fecha_q:       # «agrega una tarea para el curso redes»: ¿cuál tarea?
        return _ask(f"¿Qué tarea de {_course_label(course, curso_q)} agrego?", "titulo")
    if not fecha_q:
        return _ask("¿Para qué día es? Puedes decir «mañana», «el viernes» o «sin fecha».", "fecha")
    label = _course_label(course, curso_q) if course else None
    if course and titulo:
        title = titulo if _mentions_course(titulo, course, curso_q) else f"{label}: {titulo}"
    else:
        title = titulo or f"Tarea de {label}"
    due, due_say = "", ""
    if fecha_q and not is_none_answer(fecha_q):
        try:
            r = resolve_date(fecha_q, today, same_weekday="ask", allow_past=False)
        except DateError as e:
            return _ask(str(e), "fecha")
        if r.ambiguous:
            return _ask(r.ambiguous, "fecha")
        due, due_say = r.iso, spoken_date(r.iso, today)
    prio = names.compact(_s(args, "prioridad") or "")
    priority = prio if prio in ("alta", "media", "baja") else "media"
    category = "Universidad" if course else "Personal"
    key = names.compact(title)
    if any(isinstance(t, dict) and not t.get("done") and names.compact(t.get("title", "")) == key and (t.get("due") or "") == due
           for t in state.get("tasks", [])):
        return Result(f"Ya tienes pendiente la tarea «{title}»" + (f" para {due_say}" if due else "") + ". No la dupliqué.")

    def apply(s):
        s["tasks"].append({"id": new_id("tar", now_ms), "title": title, "category": category, "priority": priority,
                           "due": due, "done": False, "createdAt": int(now_ms if now_ms is not None else time.time() * 1000)})
    return Proposal("crear_tarea", f"¿Agrego la tarea «{title}»" + (f" para {due_say}" if due else "") + "?",
                    "Listo, tarea agregada.", apply, {"title": title, "due": due, "category": category})


# ----------------------------- completar_tarea -----------------------------

def prepare_completar_tarea(state, args, today, now_ms=None):
    q = _s(args, "titulo")
    pending = [t.get("title", "?") for t in state.get("tasks", []) if isinstance(t, dict) and not t.get("done")]
    if not pending:
        return Result("No tienes tareas pendientes.")
    listed = _join(pending[:5]) + (f" y {len(pending) - 5} más" if len(pending) > 5 else "")
    if not q:
        return _ask(f"¿Cuál marco como hecha? Tienes {listed}.", "titulo")
    r = names.resolve_task(_LEAD.sub("", q).strip() or q, state)
    if r.ambiguous:
        return _ask("¿Cuál de estas: " + _join([m.item.get("title", "?") for m in r.candidates]) + "?", "titulo")
    if not r.found:
        return _ask(f"No encontré «{q}» entre tus pendientes. Tienes {listed}. ¿Cuál?", "titulo")
    task = r.item

    def apply(s):
        t = next((x for x in s["tasks"] if isinstance(x, dict) and x.get("id") == task.get("id")), None)
        if t is None:
            raise UniGymDataError("Esa tarea ya no existe.")
        t["done"] = True
    return Proposal("completar_tarea", f"¿Marco como hecha «{task.get('title')}»?", "Listo, tarea completada.", apply,
                    {"id": task.get("id"), "title": task.get("title")})


# ------------------------------ marcar_habito ------------------------------

def prepare_marcar_habito(state, args, today, now_ms=None):
    q = _s(args, "habito")
    if not q:
        return Result.ask("¿Qué hábito marco?")
    r = names.resolve_habit(q, state)
    habits = [h.get("name", "?") for h in state.get("habits", []) if isinstance(h, dict)]
    if r.ambiguous:
        return Result.ask("¿Cuál de estos: " + _join([m.item.get("name", "?") for m in r.candidates]) + "?")
    if not r.found:
        return Result.ask(f"No encontré el hábito «{q}»." + (f" Tienes {_join(habits)}." if habits else " No tienes hábitos registrados."))
    iso, ask = _day(args, "fecha", today)
    if ask:
        return ask
    habit, label = r.item, spoken_date(iso, today)
    if (habit.get("log") or {}).get(iso):
        return Result(f"«{habit.get('name')}» ya estaba marcado {label}.")

    def apply(s):
        h = next((x for x in s["habits"] if isinstance(x, dict) and x.get("id") == habit.get("id")), None)
        if h is None:
            raise UniGymDataError("Ese hábito ya no existe.")
        h["log"][iso] = True
    return Proposal("marcar_habito", f"¿Marco «{habit.get('name')}» como cumplido {label}?", "Listo, hábito marcado.", apply,
                    {"habit": habit.get("name"), "date": iso})


# ------------------------------ registrar_peso ------------------------------

def prepare_registrar_peso(state, args, today, now_ms=None):
    kg = to_number((args or {}).get("kg"))
    if kg is None:
        return Result.ask("No entendí el peso. ¿Cuántos kilos?")
    if not WEIGHT_RANGE[0] <= kg <= WEIGHT_RANGE[1]:
        return Result.ask(f"{fmt(kg, 1)} kilos no parece correcto. Dime tu peso otra vez.")
    kg = js_round(kg, 1)
    iso, ask = _day(args, "fecha", today, past_days=30)
    if ask:
        return ask
    label = spoken_date(iso, today)
    prev = next((w for w in state.get("gym", {}).get("bodyWeight", []) if isinstance(w, dict) and w.get("date") == iso), None)
    if prev and prev.get("kg") == kg:
        return Result(f"Ya tenías {fmt(kg, 1)} kilos registrados {label}.")

    def apply(s):
        existing = next((w for w in s["gym"]["bodyWeight"] if isinstance(w, dict) and w.get("date") == iso), None)
        if existing:
            existing["kg"] = kg
        else:
            s["gym"]["bodyWeight"].append({"id": new_id("pc", now_ms), "date": iso, "kg": kg})
    extra = f" Antes había {fmt(prev.get('kg'), 1)}." if prev else ""
    return Proposal("registrar_peso", f"¿Registro {fmt(kg, 1)} kilos {label}?{extra}", "Listo, peso registrado.", apply,
                    {"kg": kg, "date": iso})


# ----------------------------- registrar_diario -----------------------------

_FIELDS = {"pasos": "steps", "paso": "steps", "steps": "steps", "sueno": "sleep", "dormi": "sleep", "sleep": "sleep",
           "agua": "water", "vasos": "water", "vaso": "water", "water": "water"}
_FIELD_NAME = {"steps": "pasos", "sleep": "horas de sueño", "water": "vasos de agua"}
_ADD = {"sumar", "suma", "agregar", "mas", "anadir", "add"}
_SET = {"fijar", "fija", "establecer", "poner", "total", "set"}


def prepare_registrar_diario(state, args, today, now_ms=None):
    field = _FIELDS.get(strip_accents(_s(args, "campo") or "").lower())
    if not field:
        return Result.ask("¿Qué registro: pasos, sueño o agua?")
    v = to_number((args or {}).get("valor"))
    if v is None or v < 0:
        return Result.ask("No entendí la cantidad. ¿Cuánto es?")
    mode_raw = strip_accents(_s(args, "modo") or "").lower()
    mode = "sumar" if mode_raw in _ADD else "fijar" if mode_raw in _SET else ("sumar" if field == "water" else "fijar")
    if field == "sleep":
        v = js_round(v, 1)
    else:
        if abs(v - round(v)) > 0.01:
            return Result.ask(f"Los {_FIELD_NAME[field]} van en números enteros. ¿Cuántos son?")
        v = int(round(v))
    iso, ask = _day(args, "fecha", today)
    if ask:
        return ask
    label = spoken_date(iso, today)
    cur = (state.get("dailyLogs", {}).get(iso) or {}).get(field)
    cur = num(cur, 0.0) if cur is not None else 0.0
    total = cur + v if mode == "sumar" else v
    if total > LIMITS[field] or v > LIMITS[field]:
        return Result.ask(f"{fmt(total, 1)} {_FIELD_NAME[field]} parece demasiado. Dime la cantidad otra vez.")

    def apply(s):
        log = s["dailyLogs"].setdefault(iso, {"steps": None, "sleep": None, "water": 0, "pomodoros": 0})
        base = num(log.get(field), 0.0) if log.get(field) is not None else 0.0
        new = base + v if mode == "sumar" else v
        if new > LIMITS[field]:
            raise UniGymDataError(f"Eso sumaría {fmt(new, 1)} {_FIELD_NAME[field]}.")
        log[field] = js_round(new, 1) if field == "sleep" else int(new)
    if mode == "sumar":
        q = f"¿Sumo {fmt(v, 1)} {_FIELD_NAME[field]} {label}? Quedarías en {fmt(total, 1)}."
    else:
        q = f"¿Registro {fmt(v, 1)} {_FIELD_NAME[field]} {label}?"
    return Proposal("registrar_diario", q, "Listo, registrado.", apply, {"field": field, "value": v, "mode": mode, "date": iso})


# ------------------------------ registrar_nota ------------------------------

def prepare_registrar_nota(state, args, today, now_ms=None):
    course, ask = _course_or_ask(state, _s(args, "curso"))
    if ask:
        return ask
    grade = to_number((args or {}).get("nota"))
    if grade is None:
        return Result.ask("No entendí la nota. ¿Cuánto sacaste?")
    if not 0 <= grade <= 20:
        return Result.ask(f"{fmt(grade)} no es una nota válida; va de 0 a 20. Dímela otra vez.")
    grade = js_round(grade, 2)
    q = _s(args, "evaluacion")
    evals = [e for e in course.get("evals", []) if isinstance(e, dict)]
    listing = _join([e.get("name", "?") for e in evals]) or "ninguna"
    if not q:
        return Result.ask(f"¿En qué evaluación de {course['name']}? Tienes {listing}.")
    from .readers import find_eval
    ev = find_eval(course, q)                              # «el último examen», «el ejercicio 3», «la práctica»
    if ev is None:
        r = names.resolve_eval(q, course)
        if r.ambiguous:
            return Result.ask("¿Cuál de estas: " + _join([m.item.get("name", "?") for m in r.candidates]) + "?")
        if not r.found:
            return Result.ask(f"No encontré la evaluación «{q}» en {course['name']}. Tienes {listing}.")
        ev = r.item
    old = ev.get("grade") if ev.get("status") == "calificada" else None
    if old is not None and abs(old - grade) < 1e-9:
        return Result(f"{ev.get('name')} de {course['name']} ya tenía {fmt(grade)}.")

    def apply(s):
        c = next((x for x in s["courses"] if isinstance(x, dict) and x.get("id") == course.get("id")), None)
        e = next((x for x in (c or {}).get("evals", []) if isinstance(x, dict) and x.get("id") == ev.get("id")), None)
        if e is None:
            raise UniGymDataError("Esa evaluación ya no existe.")
        e["grade"], e["status"] = grade, "calificada"
    extra = f" Reemplaza el {fmt(old)}." if old is not None else ""
    return Proposal("registrar_nota", f"¿Anoto {fmt(grade)} en {ev.get('name')} de {course['name']}?{extra}", "Listo, nota registrada.",
                    apply, {"course": course["name"], "eval": ev.get("name"), "grade": grade})


WRITERS = {f.__name__.removeprefix("prepare_"): f for f in (
    prepare_crear_tarea, prepare_completar_tarea, prepare_marcar_habito, prepare_registrar_peso,
    prepare_registrar_diario, prepare_registrar_nota)}
