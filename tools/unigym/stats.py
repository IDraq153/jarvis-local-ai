"""Cálculos de UniGym portados de la web a Python (misma lógica que js/modules/courses.js y schedule.js).

Hay una prueba de paridad (tests/test_unigym_stats.py) con valores calculados por el JS real
sobre datos sintéticos; si cambias la fórmula en la web, esa prueba debe cambiar también.
"""
import datetime as dt
import math

SAFE_LIMIT = 12   # igual que courses.js: nota necesaria <= 12 = seguro
RISK_LIMIT = 16   # <= 16 = en riesgo; más = crítico
DEFAULT_MIN_GRADE = 10.5
DEFAULT_TARGET_GRADE = 15.0


def js_round(value: float, decimals: int = 2) -> float:
    """Math.round de JS (mitades hacia arriba), no el redondeo bancario de Python."""
    f = 10 ** decimals
    return math.floor((value + 2.220446049250313e-16) * f + 0.5) / f


def num(value, default=0.0) -> float:
    """Number(x) || 0 de JS: texto numérico -> número; basura -> default."""
    try:
        n = float(str(value).strip().replace(",", ".")) if not isinstance(value, (int, float)) else float(value)
    except (TypeError, ValueError):
        return default
    return n if math.isfinite(n) else default


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_graded(e: dict) -> bool:
    return e.get("status") == "calificada" and finite(e.get("grade"))


def tone_for_needed(needed) -> str:
    if needed is None:
        return "neutral"
    if needed <= SAFE_LIMIT:
        return "ok"
    if needed <= RISK_LIMIT:
        return "warn"
    return "bad"


def course_stats(course: dict, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    evals = [e for e in course.get("evals", []) if isinstance(e, dict)]
    graded = [e for e in evals if is_graded(e)]
    pending = [e for e in evals if not is_graded(e)]
    total_w = js_round(sum(num(e.get("weight")) for e in evals), 2)
    graded_w = js_round(sum(num(e.get("weight")) for e in graded), 2)
    pending_w = js_round(sum(num(e.get("weight")) for e in pending), 2)
    points = sum(e["grade"] * num(e.get("weight")) / 100 for e in graded)
    average = points / graded_w * 100 if graded_w > 0 else None
    mg = course.get("minGrade")
    min_grade = mg if finite(mg) else DEFAULT_MIN_GRADE

    def needed_for(goal):
        return (goal - points) / (pending_w / 100) if pending_w > 0 else None

    needed_pass = needed_for(min_grade)
    max_final = points + 20 * pending_w / 100
    if not evals:
        status = ("neutral", "Sin evaluaciones")
    elif pending_w <= 0:
        status = ("ok", "Aprobado") if points >= min_grade else ("bad", "Desaprobado")
    elif graded_w == 0:
        status = ("neutral", "Sin notas aún")
    elif needed_pass <= 0:
        status = ("ok", "Aprobado asegurado")
    elif needed_pass > 20:
        status = ("bad", "Ya no alcanza")
    else:
        key = tone_for_needed(needed_pass)
        status = (key, {"ok": "Seguro", "warn": "En riesgo", "bad": "Crítico"}[key])

    upcoming = []
    for e in pending:
        d = e.get("date")
        try:
            if d and (dt.date.fromisoformat(d) - today).days >= 0:
                upcoming.append(e)
        except (TypeError, ValueError):
            pass
    next_eval = sorted(upcoming, key=lambda e: e["date"])[0] if upcoming else None
    return {"total_weight": total_w, "graded_weight": graded_w, "pending_weight": pending_w, "points": points,
            "average": average, "min_grade": min_grade, "needed_for": needed_for, "needed_to_pass": needed_pass,
            "max_final": max_final, "status_key": status[0], "status_label": status[1], "next_eval": next_eval,
            "graded_count": len(graded), "pending_count": len(pending),
            "finished": bool(evals) and pending_w <= 0, "pending": pending, "graded": graded}


def cycle_stats(courses: list, today: dt.date | None = None) -> dict:
    rows = [(c, course_stats(c, today)) for c in courses if isinstance(c, dict)]
    with_grades = [(c, s) for c, s in rows if s["average"] is not None]
    credits = sum(num(c.get("credits")) for c, _ in with_grades)
    avg = sum(s["average"] * num(c.get("credits")) for c, s in with_grades) / credits if credits > 0 else None
    return {"average": avg, "credits_counted": credits, "rows": rows}


def habit_streak(habit: dict, today: dt.date) -> int:
    log = habit.get("log") or {}
    cursor = today if log.get(today.isoformat()) else today - dt.timedelta(days=1)
    streak = 0
    while log.get(cursor.isoformat()):
        streak += 1
        cursor -= dt.timedelta(days=1)
    return streak


def time_to_minutes(hhmm):
    try:
        h, m = str(hhmm).split(":")
        h, m = int(h), int(m)
    except ValueError:
        return float("nan")
    return float("nan") if h > 24 or m > 59 or not 0 <= m else h * 60 + m


def blocks_for_day(state: dict, weekday: int) -> list:
    """Bloques del horario de un día (1 = lunes ... 7 = domingo) por hora de inicio."""
    blocks = [b for b in state.get("schedule", {}).get("blocks", []) if isinstance(b, dict) and b.get("day") == weekday]
    return sorted(blocks, key=lambda b: (math.inf if math.isnan(time_to_minutes(b.get("start"))) else time_to_minutes(b.get("start"))))


def routine_for_day(state: dict, weekday: int):
    return next((r for r in state.get("gym", {}).get("routines", [])
                 if isinstance(r, dict) and weekday in r.get("days", [])), None)


def weekday_index(d: dt.date) -> int:
    return d.isoweekday()   # lunes = 1 ... domingo = 7, igual que utils.js
