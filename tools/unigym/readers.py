"""Herramientas de LECTURA de UniGym (Fase 6.2): estado + argumentos crudos -> frase lista para decir.

Cada lector recibe `(state, args, today)` y devuelve un `Result`: `text` se puede leer en voz alta tal
cual (plantilla de Python, sin segunda vuelta del LLM) y `data` conserva los datos estructurados por si
se prefiere que el LLM redacte. Nunca modifican el estado. Los argumentos vienen del LLM y pueden ser
basura: aquí se validan; si algo no se entiende, `needs` trae la pregunta que JARVIS debe hacer.
"""
import datetime as dt
import re
from dataclasses import dataclass, field

from memory.text import strip_accents

from . import names
from .dates import DateError, resolve_date, spoken_date
from .stats import (is_graded, course_stats, cycle_stats, habit_streak, js_round, blocks_for_day, num,
                    routine_for_day, weekday_index)

MAX_SPOKEN_ITEMS = 5
_PRIORITY_RANK = {"alta": 0, "media": 1, "baja": 2}


@dataclass
class Result:
    text: str
    data: dict = field(default_factory=dict)
    ok: bool = True
    needs: str | None = None    # pregunta aclaratoria; si existe, `text` es esa pregunta

    @classmethod
    def ask(cls, question: str) -> "Result":
        return cls(text=question, ok=False, needs=question)


# --------------------------------- utilidades ---------------------------------

def fmt(x, decimals: int = 2) -> str:
    """Número para decirlo en español: 15 -> '15', 14.285 -> '14,29'."""
    if x is None:
        return "sin dato"
    r = js_round(float(x), decimals)
    return str(int(r)) if r == int(r) else str(r).replace(".", ",")


def _s(args, key):
    v = (args or {}).get(key)
    return str(v).strip() if v is not None and str(v).strip() else None


def _date(args, key, today, same_weekday="today"):
    return resolve_date(_s(args, key), today, same_weekday=same_weekday)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _course_or_ask(state, query):
    """(curso, None) o (None, Result con la pregunta)."""
    courses = [c for c in state.get("courses", []) if isinstance(c, dict)]
    if not courses:
        return None, Result.ask("Todavía no tienes cursos registrados en UniGym.")
    r = names.resolve_course(query, state)
    if r.found:
        return r.item, None
    listed = _join([c.get("name", "?") for c in courses])
    if r.ambiguous:
        opts = _join([m.item.get("name", "?") for m in r.candidates])
        return None, Result.ask(f"¿A cuál curso te refieres: {opts}?")
    if not query:
        return None, Result.ask(f"¿De qué curso? Tienes {listed}.")
    return None, Result.ask(f"No encontré un curso llamado «{query}». Tienes {listed}.")


def _due_phrase(due: str, today: dt.date) -> str:
    d = dt.date.fromisoformat(due)
    if d < today:
        return f"vencida {spoken_date(due, today)}"
    return f"para {spoken_date(due, today)}"


def _valid_iso(v) -> bool:
    try:
        return isinstance(v, str) and dt.date.fromisoformat(v).isoformat() == v
    except ValueError:
        return False


# ----------------------------------- lectores -----------------------------------

def consultar_agenda(state, args, today) -> Result:
    try:
        day = dt.date.fromisoformat(_date(args, "fecha", today).iso)
    except DateError as e:
        return Result.ask(str(e))
    iso = day.isoformat()
    label = spoken_date(iso, today)
    blocks = [b for b in blocks_for_day(state, weekday_index(day)) if b.get("type") != "libre"]
    evals = [(c.get("name", "?"), e) for c in state.get("courses", []) if isinstance(c, dict)
             for e in c.get("evals", []) if isinstance(e, dict) and e.get("date") == iso and e.get("status") != "calificada"]
    tasks = [t for t in state.get("tasks", []) if isinstance(t, dict) and not t.get("done") and t.get("due") == iso]
    overdue = [t for t in state.get("tasks", []) if isinstance(t, dict) and not t.get("done")
               and _valid_iso(t.get("due")) and t["due"] < iso] if day == today else []
    parts = []
    if blocks:
        spoken = [f"{b.get('title', 'bloque')} de {b.get('start', '?')} a {b.get('end', '?')}"
                  + (f" en {b['room']}" if b.get("room") else "") for b in blocks]
        parts.append(f"Tienes {_plural(len(blocks), 'bloque', 'bloques')} {label}: " + "; ".join(spoken) + ".")
    if evals:
        parts.append("Evaluación " + _join([f"{e.get('name', '?')} de {c}" for c, e in evals]) + ".")
    if tasks:
        parts.append(f"Tareas para {label}: " + _join([t["title"] for t in tasks]) + ".")
    if overdue:
        parts.append(f"Y tienes {_plural(len(overdue), 'tarea vencida', 'tareas vencidas')}.")
    if not parts:
        parts.append(f"No tienes clases ni tareas para {label}.")
    return Result(" ".join(parts), {"fecha": iso, "bloques": [b.get("title") for b in blocks],
                                    "tareas": [t["title"] for t in tasks], "vencidas": len(overdue),
                                    "evaluaciones": [e.get("name") for _, e in evals]})


def _task_mentions(task: dict, course: dict) -> bool:
    title = names.compact(task.get("title", ""))
    cands = {names.acronym(course.get("name", "")), names.compact(course.get("name", ""))}
    cid = str(course.get("id", ""))
    cands.add(names.compact(cid.split("_", 1)[-1]))
    return any(c and c in title for c in cands)


def tareas_pendientes(state, args, today) -> Result:
    tasks = [t for t in state.get("tasks", []) if isinstance(t, dict) and not t.get("done")]
    curso = None
    if _s(args, "curso"):
        curso, ask = _course_or_ask(state, _s(args, "curso"))
        if ask:
            return ask
        tasks = [t for t in tasks if _task_mentions(t, curso)]
    tasks.sort(key=lambda t: (t.get("due") if _valid_iso(t.get("due")) else "9999",
                              _PRIORITY_RANK.get(t.get("priority"), 1), num(t.get("createdAt"))))
    where = f" de {curso['name']}" if curso else ""
    if not tasks:
        return Result(f"No tienes tareas pendientes{where}.", {"total": 0})
    shown = tasks[:MAX_SPOKEN_ITEMS]
    spoken = [t.get("title", "?") + (f", {_due_phrase(t['due'], today)}" if _valid_iso(t.get("due")) else "") for t in shown]
    more = f" Y {len(tasks) - len(shown)} más." if len(tasks) > len(shown) else ""
    return Result(f"Tienes {_plural(len(tasks), 'tarea pendiente', 'tareas pendientes')}{where}: "
                  + "; ".join(spoken) + "." + more, {"total": len(tasks), "titulos": [t.get("title") for t in shown]})


def rutina_gym(state, args, today) -> Result:
    try:
        day = dt.date.fromisoformat(_date(args, "fecha", today).iso)
    except DateError as e:
        return Result.ask(str(e))
    label = spoken_date(day.isoformat(), today)
    routine = routine_for_day(state, weekday_index(day))
    active = state.get("gym", {}).get("activeSession")
    if not routine:
        return Result(f"{label.capitalize()} descansas del gym.", {"rutina": None})
    exercises = [e for e in routine.get("exercises", []) if isinstance(e, dict)]
    spoken = []
    for e in exercises[:6]:
        lo, hi = e.get("repMin"), e.get("repMax")
        reps = f"{lo}" if lo == hi or hi is None else f"{lo} a {hi}"
        spoken.append(f"{e.get('name', '?')} {e.get('sets', '?')} por {reps}")
    text = f"{label.capitalize()} te toca {routine.get('name', 'tu rutina')}"
    text += f" ({routine['focus']})" if routine.get("focus") else ""
    text += ": " + "; ".join(spoken) + "." if spoken else ", sin ejercicios cargados."
    if len(exercises) > 6:
        text += f" Y {len(exercises) - 6} ejercicios más."
    if active and day == today:
        text += " Ya tienes un entrenamiento en curso."
    return Result(text, {"rutina": routine.get("name"), "ejercicios": len(exercises)})


def notas_curso(state, args, today) -> Result:
    if _s(args, "curso"):
        curso, ask = _course_or_ask(state, _s(args, "curso"))
        if ask:
            return ask
        s = course_stats(curso, today)
        if not curso.get("evals"):
            return Result(f"{curso['name']} todavía no tiene evaluaciones registradas.", {"curso": curso["name"]})
        if s["average"] is None:
            return Result(f"En {curso['name']} aún no tienes notas; falta el {fmt(s['pending_weight'])} por ciento.",
                          {"curso": curso["name"]})
        notas = [f"{e.get('name', '?')} {fmt(e['grade'])}" for e in s["graded"]]
        text = (f"En {curso['name']} llevas {fmt(s['average'])} de promedio. Ya diste: " + "; ".join(notas) + ". ")
        if s["finished"]:
            text += f"Terminaste con {fmt(s['points'])}: {s['status_label'].lower()}."
        else:
            falta = _join([f"{e.get('name', '?')} ({fmt(num(e.get('weight')))}%)" for e in s["pending"]])
            text += f"Te falta {falta}. Estado: {s['status_label'].lower()}."
            if s["needed_to_pass"] is not None and 0 < s["needed_to_pass"] <= 20:
                text += f" Para aprobar necesitas promediar {fmt(s['needed_to_pass'])} en lo que falta."
        return Result(text, {"curso": curso["name"], "promedio": s["average"], "estado": s["status_label"],
                             "notas": {e.get("name"): e["grade"] for e in s["graded"]}})
    cs = cycle_stats(state.get("courses", []), today)
    if not cs["rows"]:
        return Result("Todavía no tienes cursos registrados en UniGym.")
    graded = [(c, s) for c, s in cs["rows"] if s["average"] is not None]
    if not graded:
        return Result("Aún no tienes notas registradas en ningún curso.")
    spoken = [f"{c.get('name', '?')} {fmt(s['average'])}" for c, s in graded]
    return Result(f"Tu promedio del ciclo es {fmt(cs['average'])}. Por curso: " + "; ".join(spoken) + ".",
                  {"promedio_ciclo": cs["average"], "cursos": len(graded)})


def nota_necesaria(state, args, today) -> Result:
    curso, ask = _course_or_ask(state, _s(args, "curso"))
    if ask:
        return ask
    s = course_stats(curso, today)
    raw_goal = (args or {}).get("meta")
    to_pass = raw_goal is None or str(raw_goal).strip() == ""
    if to_pass:                    # sin meta: "¿qué nota necesito para aprobar?" -> nota mínima del curso
        goal = s["min_grade"]
    else:
        goal = num(raw_goal, default=-1)
    if not 0 <= goal <= 20:
        return Result.ask("¿Qué nota final quieres alcanzar? Debe estar entre 0 y 20.")
    name = curso["name"]
    if s["pending_weight"] <= 0:
        return Result(f"En {name} ya no quedan evaluaciones; terminaste con {fmt(s['points'])}.", {"curso": name})
    needed = s["needed_for"](goal)
    pend = [e for e in s["pending"]]
    falta = _join([f"{e.get('name', '?')} ({fmt(num(e.get('weight')))}%)" for e in pend])
    base = {"curso": name, "meta": goal, "necesaria": needed, "peso_pendiente": s["pending_weight"], "para_aprobar": to_pass}
    meta_txt = f"aprobar {name} ({fmt(goal)})" if to_pass else f"llegar a {fmt(goal)} en {name}"
    if needed <= 0:
        return Result(f"Ya tienes asegurado {'aprobar' if to_pass else fmt(goal)} en {name}: aunque saques 0 en lo que falta.", base)
    if needed > 20:
        return Result(f"En {name} ya no alcanza para {'aprobar' if to_pass else fmt(goal)}: aun sacando 20 en todo lo que falta "
                      f"llegarías a {fmt(s['max_final'])}.", base)
    solo = len(pend) == 1
    how = f"sacar {fmt(needed)} en {pend[0].get('name', 'la evaluación pendiente')}" if solo else \
        f"promediar {fmt(needed)} en lo que falta: {falta}"
    return Result(f"Para {meta_txt} necesitas {how}. Hoy llevas {fmt(s['points'])} puntos "
                  f"y falta el {fmt(s['pending_weight'])} por ciento.", base)


def progreso_diario(state, args, today) -> Result:
    try:
        iso = _date(args, "fecha", today).iso
    except DateError as e:
        return Result.ask(str(e))
    log = (state.get("dailyLogs") or {}).get(iso) or {}
    goals = state.get("settings", {}).get("goals") or {}
    label = spoken_date(iso, today)

    def line(name, key, unit, goal_key):
        v, g = log.get(key), goals.get(goal_key)
        u = f" {unit}" if unit else ""
        if v is None:
            return f"{name}: sin registrar"
        return f"{name}: {fmt(v, 1)}{u}" + (f" de {fmt(g, 1)}" if g else "")
    parts = [line("Pasos", "steps", "", "steps"), line("Sueño", "sleep", "horas", "sleep"),
             line("Agua", "water", "vasos", "water")]
    return Result(f"Registro de {label}: " + "; ".join(parts) + ".", {"fecha": iso, "log": dict(log)})


def estado_habitos(state, args, today) -> Result:
    habits = [h for h in state.get("habits", []) if isinstance(h, dict)]
    if not habits:
        return Result("Todavía no tienes hábitos registrados.")
    iso = today.isoformat()
    done = [h for h in habits if (h.get("log") or {}).get(iso)]
    todo = [h for h in habits if h not in done]
    text = f"Hoy llevas {len(done)} de {len(habits)} hábitos."
    if todo:
        text += " Te faltan: " + _join([h.get("name", "?") for h in todo]) + "."
    else:
        text += " ¡Los completaste todos!"
    best = max(habits, key=lambda h: habit_streak(h, today))
    streak = habit_streak(best, today)
    if streak >= 2:
        text += f" Tu mejor racha es {best.get('name', '?')}, {streak} días."
    return Result(text, {"hechos": len(done), "total": len(habits), "faltan": [h.get("name") for h in todo]})


_TYPE_WORDS = {"examen": "examen", "examenes": "examen", "proyecto": "proyecto", "proyectos": "proyecto",
               "practica": "practica", "practicas": "practica", "informe": "informe", "informes": "informe",
               "ejercicio": "ejercicio", "ejercicios": "ejercicio", "trabajo": "trabajo", "trabajos": "trabajo"}


def _plain(text) -> str:
    """minúsculas, sin tildes ni símbolos (names.compact quita «examen» como palabra vacía, aquí no sirve)."""
    return "".join(ch for ch in strip_accents(str(text or "")).lower() if ch.isalnum())


MAX_TOPICS_SPOKEN = 4   # por voz: lo principal y se ofrece el resto («¿sigo?»)
MAX_PLAN_SPOKEN = 3
BRIEF_WORDS = 9


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("true", "1", "si", "sí", "yes", "verdadero")


_ORDINALS = {"primer": 1, "primero": 1, "primera": 1, "segundo": 2, "segunda": 2, "tercer": 3, "tercero": 3, "tercera": 3,
             "cuarto": 4, "cuarta": 4, "quinto": 5, "quinta": 5, "sexto": 6, "sexta": 6}
_NUMS = {"uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6}
_KINDS = {"examen": "examen", "examenes": "examen", "parcial": "examen", "proyecto": "proyecto", "proyectos": "proyecto",
          "practica": "practica", "practicas": "practica", "informe": "informe", "informes": "informe",
          "ejercicio": "ejercicio", "ejercicios": "ejercicio", "trabajo": "trabajo", "trabajos": "trabajo",
          "sustentacion": "sustentacion", "evaluacion": "", "evaluaciones": "", "eva": ""}
_KIND_SAY = {"examen": "exámenes", "proyecto": "proyectos", "practica": "prácticas", "informe": "informes",
             "ejercicio": "ejercicios", "trabajo": "trabajos", "sustentacion": "sustentaciones"}


def eval_spec(text) -> tuple:
    """«el tercer ejercicio individual» -> ("ejercicio", 3, False); «la evaluación 2» -> ("", 2, False);
    «el último examen» -> ("examen", None, True). Tipo None = no lo dijo. El número solo cuenta como ordinal o
    pegado al tipo («evaluación 2»), así «estructura de datos 2» no se toma por la evaluación 2."""
    toks = [t for t in re.split(r"[^a-z0-9]+", strip_accents(str(text or "")).lower()) if t]
    kind, n, last, k_at = None, None, False, None
    for i, t in enumerate(toks):
        if t in _KINDS and kind is None:
            kind, k_at = _KINDS[t], i
        if t in _ORDINALS and n is None:
            n = _ORDINALS[t]
        if t in ("ultimo", "ultima", "final"):
            last = True
    if n is None and k_at is not None:
        for t in toks[k_at + 1:k_at + 3]:
            if t.isdigit() and int(t) <= 20:
                n = int(t)
                break
            if t in _NUMS:
                n = _NUMS[t]
                break
    return kind, n, last


def _ev_number(e) -> int | None:
    m = re.findall(r"\d+", strip_accents(str(e.get("name", ""))))
    return int(m[-1]) if m else None


def _list_evals(evals) -> str:
    return _join([e.get("name", "?") for e in evals])


def temas_examen(state, args, today) -> Result:
    """Temas (o plan de estudio) de una evaluación. Entiende tipo («examen», «ejercicio»...), número u ordinal
    («el tercero», «evaluación 2») y «último». Si lo pedido no existe, no dice solo «no hay»: cuenta qué evaluaciones
    tiene el curso y pregunta de cuál (la respuesta siguiente se resuelve sin el LLM)."""
    courses = [c for c in state.get("courses", []) if isinstance(c, dict)]
    curso = None
    if _s(args, "curso"):
        curso, ask = _course_or_ask(state, _s(args, "curso"))
        if ask:
            return ask
        courses = [curso]
    pool = [dict(e, _c=c) for c in courses for e in c.get("evals", []) if isinstance(e, dict)]
    if not pool:
        return Result("No tengo evaluaciones cargadas" + (f" de {curso['name']}." if curso else "."))
    # Lo que DIJO Dario manda sobre lo que pasó el LLM (el LLM cambió «ejercicio individual» por «examen»).
    h_kind, h_n, h_last = eval_spec((args or {}).get("_heard"))
    q_kind, q_n, q_last = eval_spec(_s(args, "evaluacion"))
    kind = h_kind if h_kind is not None else q_kind
    n = h_n if h_n is not None else q_n
    last = h_last or q_last or (_plain(_s(args, "cual")) in ("ultimo", "ultima", "final") and n is None)
    heard_txt = (args or {}).get("_heard")
    plan = wants_plan(heard_txt) if heard_txt else _truthy((args or {}).get("plan"))   # lo dicho manda: el LLM pone plan=true de más
    query = _s(args, "evaluacion")
    ev = None
    if kind is None and n is None and query and _plain(query) not in ("", "ultimo", "proximo"):
        r = names.resolve(query, pool, lambda e: [e.get("name", ""), f"{e.get('name', '')} {e['_c'].get('name', '')}"])
        if r.found:
            ev = r.item
        else:
            where = f"En {curso['name']} tengo" if curso else "Tengo"
            opts = curso["evals"] if curso else [m.item for m in r.candidates]
            return _ask_slot(f"No encontré «{query}». {where} {_list_evals(opts)}. ¿Cuál?", "evaluacion")
    if ev is None:
        cands = pool if not kind else [e for e in pool if kind in _plain(e.get("name", ""))]
        if not cands:
            what = _KIND_SAY.get(kind, "evaluaciones de ese tipo")
            if not curso:
                return _ask_slot(f"No encuentro {what}. ¿De qué curso?", "curso")
            return _ask_slot(f"En {curso['name']} no hay {what}. Tiene {_list_evals(curso['evals'])}. ¿De cuál quieres "
                             f"{'el plan' if plan else 'los temas'}?", "evaluacion")
        if n is not None:
            exact = [e for e in cands if _ev_number(e) == n]
            ev = exact[0] if len(exact) == 1 else (cands[n - 1] if not exact and 0 < n <= len(cands) and curso else None)
            if ev is None:
                where = curso["name"] if curso else "tus cursos"
                return _ask_slot(f"No encontré esa evaluación en {where}. Tiene {_list_evals(cands)}. ¿Cuál?", "evaluacion")
        else:
            dated = sorted((e for e in cands if _valid_iso(e.get("date"))), key=lambda e: e["date"])
            upcoming = [e for e in dated if e["date"] >= today.isoformat() and e.get("status") != "calificada"]
            if last:
                ev = dated[-1] if dated else cands[-1]
            elif upcoming:
                ev = upcoming[0]
            elif dated:
                ev = dated[-1]
            else:
                return _ask_slot(f"Tengo {_list_evals(cands)}. ¿De cuál?", "evaluacion")
    curso_name = ev["_c"].get("name", "?")
    when = f" ({spoken_date(ev['date'], today)})" if _valid_iso(ev.get("date")) else ""
    head = f"{ev.get('name', 'la evaluación')} de {curso_name}{when}"
    topics = [str(t) for t in (ev.get("topics") or []) if str(t).strip()]
    guide = [str(g) for g in (ev.get("guide") or []) if str(g).strip()]
    base = {"curso": curso_name, "evaluacion": ev.get("name"), "temas": topics, "plan": guide}
    full = bool((args or {}).get("_todo"))               # «sí» a «¿sigo?»: el resto, sin recortar
    if plan:
        if not guide:
            extra = " ¿Te digo los temas que entran?" if topics else ""
            return Result(f"Para {head} no tengo un plan de estudio cargado.{extra}", base)
        if full:
            rest = guide[MAX_PLAN_SPOKEN:] or guide
            return Result("Sigo: " + "; ".join(_item(g) for g in rest) + ".", base)
        shown = guide[:MAX_PLAN_SPOKEN]
        more = len(guide) > len(shown)
        base["more"] = more
        return Result(f"Plan de estudio para {ev.get('name', 'esa evaluación')}: " + "; ".join(brief(g) for g in shown) + "."
                      + (f" Hay {len(guide) - len(shown)} pasos más, ¿sigo?" if more else ""), base)
    if not topics:
        extra = " Pero sí tengo un plan de estudio, ¿te lo digo?" if guide else ""
        return Result(f"Para {head} no tengo temas cargados.{extra}", base)
    if full:
        rest = topics[MAX_TOPICS_SPOKEN:] or topics
        return Result("Los demás temas: " + "; ".join(_item(t) for t in rest) + ".", base)
    shown = topics[:MAX_TOPICS_SPOKEN]
    more = len(topics) > len(shown)
    base["more"] = more
    tail = f" Y {len(topics) - len(shown)} más, ¿te los digo?" if more else (" ¿Quieres el plan de estudio?" if guide else "")
    return Result(f"En {head} entran: " + "; ".join(brief(t) for t in shown) + "." + tail, base)


def brief(text: str, words: int = BRIEF_WORDS) -> str:
    """Versión corta para decir en voz alta: hasta los dos puntos / paréntesis / punto y coma, y como mucho `words` palabras.
    «Ejecuta paso a paso, a mano, Prim, Kruskal, Dijkstra y Ford-Fulkerson sobre un grafo de 6 a 8 nodos.» ->
    «Ejecuta paso a paso, a mano, Prim, Kruskal, Dijkstra y Ford-Fulkerson»."""
    t = _item(text)
    t = re.split(r"\s*[:;(]\s*|\.\s", t, maxsplit=1)[0].strip()
    ws = t.split()
    if len(ws) > words:
        cut = " ".join(ws[:words])
        t = re.sub(r"\s+(y|o|de|del|la|el|los|las|a|en|con|para|por|que|sobre)$", "", cut.rstrip(",")).rstrip(",")
    return t


_PLAN_WORDS = ("plan", "estudi", "prepar", "repas")


def wants_plan(heard) -> bool:
    """«dame un plan», «cómo estudio», «para prepararme» -> plan de estudio; «quiero saber la evaluación» -> no."""
    t = strip_accents(str(heard or "")).lower()
    return any(w in t for w in _PLAN_WORDS)


def _item(text: str) -> str:
    return str(text).strip().rstrip(".;: ")


def evaluaciones_proximas(state, args, today) -> Result:
    """Las evaluaciones que se acercan (fecha, peso), de un curso o de todos."""
    courses = [c for c in state.get("courses", []) if isinstance(c, dict)]
    curso = None
    if _s(args, "curso"):
        curso, ask = _course_or_ask(state, _s(args, "curso"))
        if ask:
            return ask
        courses = [curso]
    iso = today.isoformat()
    up = sorted((dict(e, _c=c) for c in courses for e in c.get("evals", []) if isinstance(e, dict)
                 and _valid_iso(e.get("date")) and e["date"] >= iso and e.get("status") != "calificada"),
                key=lambda e: e["date"])
    data = {"curso": curso["name"] if curso else None, "evaluaciones": [e.get("name") for e in up]}
    if not up:
        return Result((f"En {curso['name']} no" if curso else "No") + " te quedan evaluaciones pendientes con fecha.", data)
    shown = up[:6]

    def one(e):
        w = num(e.get("weight"))
        peso = f", vale {fmt(w)} por ciento" if w else ""
        return f"{e.get('name', '?')}{'' if curso else ' de ' + e['_c'].get('name', '?')}, {spoken_date(e['date'], today)}{peso}"
    more = f" Y {len(up) - len(shown)} más." if len(up) > len(shown) else ""
    where = f"En {curso['name']} se" if curso else "Se"
    return Result(f"{where} acercan: " + "; ".join(one(e) for e in shown) + "." + more
                  + " ¿Quieres los temas de alguna?", data)


_NUMW = {"cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
         "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
         "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20}
_NUM_RE = r"(\d{1,2}(?:[.,]\d+)?|" + "|".join(sorted(_NUMW, key=len, reverse=True)) + r")"
_PAIR = re.compile(_NUM_RE + r"\s+(?:en|de|para|al|a)\s+(.+?)(?=\s+y\s+(?:(?:saco|sacar|sacando|un|una)\s+)?" + _NUM_RE
                   + r"\b|\s*,\s*|\s*;\s*|$)")
_REST = ("todo", "todas", "todos", "lo que falta", "el resto", "lo demas", "las demas", "los demas", "cada una", "ambas", "ambos")


def _to_grade(tok: str):
    t = tok.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return float(_NUMW[t]) if t in _NUMW else None


def find_eval(course: dict, text: str):
    """La evaluación del curso que nombra `text` («el proyecto», «el ejercicio individual 3», «la EVA 2»), o None."""
    evals = [e for e in course.get("evals", []) if isinstance(e, dict)]
    kind, n, last = eval_spec(text)
    cands = evals if not kind else [e for e in evals if kind in _plain(e.get("name", ""))]
    if last and n is None and cands:                     # «el último (examen)»: el último por fecha
        dated = sorted((e for e in cands if _valid_iso(e.get("date"))), key=lambda e: e["date"])
        return dated[-1] if dated else cands[-1]
    if n is not None:
        exact = [e for e in cands if _ev_number(e) == n]
        if len(exact) == 1:
            return exact[0]
        if not exact and 0 < n <= len(cands):
            return cands[n - 1]
        return None
    if kind:
        pending = [e for e in cands if not is_graded(e)]
        if len(cands) == 1:
            return cands[0]
        if len(pending) == 1:
            return pending[0]
    r = names.resolve(text, evals, lambda e: [e.get("name", "")])
    return r.item if r.found else None


def simular_notas(state, args, today) -> Result:
    """«¿Y si saco 15 en el proyecto y 15 en el ejercicio 3?» -> nota final (o lo que faltaría), con la fórmula de la web."""
    curso, ask = _course_or_ask(state, _s(args, "curso"))
    if ask:
        return ask
    said = strip_accents(str((args or {}).get("_heard") or _s(args, "supuestos") or "")).lower()
    pairs = _PAIR.findall(said)
    if not pairs:                                        # «si saco en el último de redes 15»: nota después
        pairs = [(m.group(2), m.group(1)) for m in re.finditer(
            r"\b(?:en|de)\s+((?:el|la|los|las)\s+.+?)\s+" + _NUM_RE + r"\b", said)]
    if not pairs:
        return _ask_slot(f"Dime qué nota supones y en qué evaluación de {curso['name']}, por ejemplo «15 en el proyecto».", "supuestos")
    s = course_stats(curso, today)
    hyp, notes, bad = {}, [], []
    for raw_n, target, *_ in pairs:
        g = _to_grade(raw_n)
        target = target.strip(" .?!¿¡")
        if g is None or not 0 <= g <= 20:
            bad.append(raw_n)
            continue
        if any(w in target for w in _REST):
            for e in s["pending"]:
                hyp.setdefault(e.get("id"), (e, g))
            continue
        e = find_eval(curso, target)
        if e is None:
            bad.append(target)
        elif is_graded(e):
            notes.append(f"{e.get('name')} ya está calificada con {fmt(e['grade'])}, uso esa nota")
        else:
            hyp[e.get("id")] = (e, g)
    if not hyp and notes:
        return Result(" ".join(n + "." for n in notes) + " Dime una nota para algo que aún te falte: "
                      + _join([e.get("name", "?") for e in s["pending"]]) + ".", {"curso": curso["name"]})
    if bad and not hyp:
        return _ask_slot(f"No ubiqué «{bad[0]}» en {curso['name']}. Tiene {_list_evals(curso['evals'])}. ¿Cuál?", "supuestos")
    points = s["points"] + sum(g * num(e.get("weight")) / 100 for e, g in hyp.values())
    rest = [e for e in s["pending"] if e.get("id") not in hyp]
    rest_w = sum(num(e.get("weight")) for e in rest)
    supuesto = _join([f"{fmt(g)} en {e.get('name')}" for e, g in hyp.values()])
    pre = (" ".join(n + "." for n in notes) + " ") if notes else ""
    data = {"curso": curso["name"], "final": None, "puntos": points}
    if not rest:
        ok = points >= s["min_grade"]
        data["final"] = points
        return Result(f"{pre}Si sacas {supuesto}, terminarías {curso['name']} con {fmt(points)}: "
                      f"{'apruebas' if ok else 'no llegas a aprobar'} (mínimo {fmt(s['min_grade'])}).", data)
    need = (s["min_grade"] - points) / (rest_w / 100) if rest_w > 0 else None
    falta = _join([f"{e.get('name')} ({fmt(num(e.get('weight')))}%)" for e in rest])
    tail = ("ya tendrías aprobado" if need is not None and need <= 0 else
            f"para aprobar necesitarías {fmt(need)} en {falta}" if need is not None and need <= 20 else
            f"aun con 20 en {falta} no llegarías a aprobar")
    return Result(f"{pre}Si sacas {supuesto} en {curso['name']}, llevarías {fmt(points)} puntos y aún faltaría {falta}: {tail}.", data)


_ORD_LEVEL = {"primer": 1, "primero": 1, "primera": 1, "segundo": 2, "segunda": 2, "tercer": 3, "tercero": 3, "tercera": 3,
              "cuarto": 4, "cuarta": 4, "quinto": 5, "quinta": 5, "sexto": 6, "sexta": 6, "septimo": 7, "octavo": 8,
              "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8}


def _levels_said(q: str) -> list[int]:
    """«nivel 4 y 5», «cuarto y quinto ciclo», «niveles 1, 2 y 3» -> [4, 5] ... (en el orden dicho, sin repetir)."""
    out = []
    word = r"(?:\d{1,2}|" + "|".join(sorted(_ORD_LEVEL, key=len, reverse=True)) + r")\b"
    for m in re.finditer(r"\b(?:nivel|niveles|ciclo|ciclos)\s+(" + word + r"(?:\s*(?:,|y|e|con|vs|contra|al|a)?\s*" + word + r")*)", q):
        for t in re.findall(r"\d{1,2}|[a-z]+", m.group(1)):
            n = int(t) if t.isdigit() else _ORD_LEVEL.get(t)
            if n and n < 20 and n not in out:
                out.append(n)
    for m in re.finditer(r"\b(primer|segundo|tercer|cuarto|quinto|sexto)\b(?=[a-z ,]*\b(?:ciclo|nivel)s?\b)", q):
        n = _ORD_LEVEL[m.group(1)]
        if n not in out:
            out.append(n)
    if not out and re.search(r"\b(?:nivel|niveles|ciclo|ciclos)\b", q):       # «nivel de récord académico 4 y 5»
        for t in re.findall(r"\b\d{1,2}\b", re.sub(r"\d{4}\s*-\s*[12]", " ", q)):
            if 0 < int(t) < 20 and int(t) not in out:
                out.append(int(t))
    return out


def _pct(new: float, old: float) -> str:
    if not old:
        return ""
    p = (new - old) / old * 100
    return f"{fmt(abs(js_round(p, 1)), 1)} por ciento"


def _change(new: float, old: float) -> str:
    """«subió 1,2 por ciento» en palabras simples."""
    if abs(new - old) < 0.005:
        return "se mantuvo igual"
    return f"{'subió' if new > old else 'bajó'} un {_pct(new, old)}"


def _cycle_by_level(cycles, n):
    return next((c for c in cycles if num(c.get("level")) == n), None) or (cycles[n - 1] if 0 < n <= len(cycles) else None)


def record_academico(state, args, today) -> Result:
    """Récord académico explicado en palabras simples: «promedio del ciclo» (PPP) y «promedio acumulado» (PPA), cuánto
    subiste o bajaste en por ciento, y qué significa. Un ciclo, una comparación entre dos, todos, o el resumen."""
    rec = state.get("record") or {}
    cycles = sorted((c for c in rec.get("cycles", []) if isinstance(c, dict) and c.get("cycle")), key=lambda c: str(c["cycle"]))
    if not cycles:
        return Result("Todavía no tienes tu récord académico cargado en UniGym.")
    q = strip_accents(str(_s(args, "ciclo") or "") + " " + str((args or {}).get("_heard") or "")).lower()
    lv = lambda c: f"nivel {c['level']}" if c.get("level") else str(c["cycle"])  # noqa: E731

    if re.search(r"\b(todos|cada|ciclo por ciclo|historial)\b", q):
        parts = [f"{lv(c)}: {fmt(num(c.get('ppp')), 1)}" for c in cycles]
        first, last = cycles[0], cycles[-1]
        return Result("Tu promedio de cada ciclo fue " + "; ".join(parts) + f". Tu acumulado {_change(num(last.get('ppa')), num(first.get('ppa')))}"
                      f" desde el primero: de {fmt(num(first.get('ppa')), 1)} a {fmt(num(last.get('ppa')), 1)}.",
                      {"ciclos": len(cycles)})

    picked = []
    for m in re.finditer(r"\b(\d{4})\s*-\s*([12])\b", q):
        c = next((c for c in cycles if str(c["cycle"]) == f"{m.group(1)}-{m.group(2)}"), None)
        if c and c not in picked:
            picked.append(c)
    for n in _levels_said(q):
        c = _cycle_by_level(cycles, n)
        if c is None:
            return Result(f"No tengo el nivel {n} en tu récord. Tengo del nivel {cycles[0].get('level', 1)} al "
                          f"{cycles[-1].get('level', len(cycles))}.")
        if c not in picked:
            picked.append(c)

    if len(picked) >= 2:                                     # comparación entre dos ciclos
        a, b = sorted(picked[:2], key=lambda c: str(c["cycle"]))
        pa, pb, aa, ab = (num(a.get("ppp")), num(b.get("ppp")), num(a.get("ppa")), num(b.get("ppa")))
        text = (f"Del {lv(a)} al {lv(b)}, tu promedio del ciclo {_change(pb, pa)}, de {fmt(pa, 1)} a {fmt(pb, 1)}, "
                f"y tu promedio acumulado {_change(ab, aa)}.")
        ca, cb = num(a.get("credits")), num(b.get("credits"))
        if pb >= pa and cb > ca:
            text += f" Y eso llevando más créditos ({fmt(cb)} contra {fmt(ca)}): rendiste mejor con más carga."
        elif pb < pa and cb > ca:
            text += f" Bajó un poco, pero llevaste más créditos ({fmt(cb)} contra {fmt(ca)})."
        text += (" En resumen: vas mejorando." if pb > pa + 0.005 else
                 " En resumen: te mantuviste estable." if abs(pb - pa) <= 0.005 else
                 " En resumen: bajaste un poco, pero sigues por encima de tu acumulado." if pb > ab else
                 " En resumen: ese ciclo te costó más; el siguiente lo puede recuperar.")
        return Result(text, {"ciclos": [a["cycle"], b["cycle"]]})

    if len(picked) == 1:                                     # un ciclo, explicado
        c = picked[0]
        i = cycles.index(c)
        ppp, ppa = num(c.get("ppp")), num(c.get("ppa"))
        text = f"En el {lv(c)} ({c['cycle']}) tu promedio del ciclo fue {fmt(ppp, 1)} y tu acumulado quedó en {fmt(ppa, 1)}."
        if i > 0:
            prev = cycles[i - 1]
            text += (f" Frente al {lv(prev)}, tu promedio del ciclo {_change(ppp, num(prev.get('ppp')))}"
                     f" y tu acumulado {_change(ppa, num(prev.get('ppa')))}.")
        text += " Fue un ciclo que te subió el acumulado." if ppp > ppa + 0.005 else \
            " Ese ciclo te bajó un poco el acumulado." if ppp < ppa - 0.005 else ""
        return Result(text, {"ciclo": c["cycle"]})

    last, first = cycles[-1], cycles[0]                      # resumen general
    ppa, ppp = num(last.get("ppa")), num(last.get("ppp"))
    streak = 0
    for i in range(len(cycles) - 1, 0, -1):
        if num(cycles[i].get("ppa")) > num(cycles[i - 1].get("ppa")):
            streak += 1
        else:
            break
    rank = f" Estás en el {rec['ranking'].lower()}." if rec.get("ranking") else ""
    text = f"Tu promedio acumulado es {fmt(ppa, 1)}.{rank}"
    if len(cycles) >= 2:
        text += (f" Subió en todos tus ciclos: un {_pct(ppa, num(first.get('ppa')))} desde el primero." if streak == len(cycles) - 1 else
                 f" Lleva {streak} {'ciclo' if streak == 1 else 'ciclos'} seguidos subiendo." if streak else
                 " En el último ciclo no subió.")
    data = {"ppa": ppa, "ppp": ppp, "ciclo": last["cycle"], "ciclos": len(cycles)}
    cs = cycle_stats(state.get("courses", []), today)
    if cs["average"] is not None and cs["credits_counted"]:
        acc = sum(num(c.get("credits")) for c in cycles)
        credits = sum(num(c.get("credits")) for c in state.get("courses", []) if isinstance(c, dict))
        if acc > 0 and credits > 0:
            proj = (ppa * acc + cs["average"] * credits) / (acc + credits)
            data.update(ppp_actual=cs["average"], ppa_proyectado=proj)
            text += (f" Si terminas este ciclo como vas ({fmt(cs['average'], 1)}), tu acumulado "
                     + ("seguiría subiendo." if proj > ppa + 0.005 else "bajaría un poco." if proj < ppa - 0.005
                        else "se mantendría."))
    return Result(text + " Puedo compararte dos niveles, por ejemplo «nivel 4 y 5».", data)


def lista_cursos(state, args, today) -> Result:
    """Los cursos registrados en UniGym, tal cual están en los datos (nunca de memoria ni inventados)."""
    courses = [c for c in state.get("courses", []) if isinstance(c, dict) and c.get("name")]
    if not courses:
        return Result.ask("Todavía no tienes cursos registrados en UniGym.")
    names_ = [c["name"] for c in courses]
    return Result(f"Tienes {_plural(len(courses), 'curso', 'cursos')}: {_join(names_)}.",
                  {"cursos": names_})


def _ask_slot(question: str, slot: str) -> Result:
    r = Result.ask(question)
    r.data = {"slot": slot}
    return r


READERS = {f.__name__: f for f in (consultar_agenda, tareas_pendientes, rutina_gym, notas_curso,
                                   nota_necesaria, progreso_diario, estado_habitos, temas_examen,
                                   evaluaciones_proximas, simular_notas, record_academico, lista_cursos)}
