"""Compara tools/unigym/stats.py con el JS real de la web (js/modules/courses.js) sobre datos sintéticos.

Uso: python scripts/check_stats_parity.py [--web RUTA_A_uni-gym-tracker] [--dump]
Requiere Node. Extrae `isGraded`, `toneForNeeded` y `courseStats` del archivo de la web (sin importar
el módulo entero, que toca el DOM), los ejecuta y compara. Con --dump imprime los valores de JS
(así se generó tests/golden/course_stats.json, que usa la prueba unitaria sin necesitar Node).
Nada personal: los cursos de prueba son inventados.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FUTURE, PAST = "2099-01-01", "2000-01-01"
GOALS = [10.5, 12, 15, 18]


def ev(i, w, grade=None, date=None, status=None):
    return {"id": f"e{i}", "name": f"EVA {i}", "weight": w, "grade": grade, "date": date or "",
            "status": status or ("calificada" if grade is not None else "pendiente")}


COURSES = [
    {"id": "c0", "evals": []},
    {"id": "c1", "evals": [ev(1, 25), ev(2, 25), ev(3, 50, date=FUTURE)]},
    {"id": "c2", "evals": [ev(1, 20, 14), ev(2, 30, 16), ev(3, 25, date=FUTURE), ev(4, 25, date="2098-06-01")]},
    {"id": "c3", "evals": [ev(1, 40, 15), ev(2, 60, 12)]},
    {"id": "c4", "evals": [ev(1, 40, 8), ev(2, 60, 10)]},
    {"id": "c5", "evals": [ev(1, 60, 4), ev(2, 20, date=FUTURE), ev(3, 20, date=PAST)]},
    {"id": "c6", "evals": [ev(1, 70, 18), ev(2, 30, date=FUTURE)]},
    {"id": "c7", "minGrade": 13, "evals": [ev(1, 33.33, 13.5), ev(2, 33.33, 11), ev(3, 33.34, date=FUTURE)]},
    {"id": "c8", "evals": [ev(1, "25", 15), {"id": "e2", "name": "EVA 2", "weight": "75", "grade": "15", "status": "calificada"}]},
    {"id": "c9", "minGrade": 14, "evals": [ev(1, 10, 20), ev(2, 15, 0), ev(3, 75, date=FUTURE)]},
    {"id": "c10", "evals": [ev(1, 30, 12.25), ev(2, 30, 13.75), ev(3, 40, date=PAST)]},
]

JS_HARNESS = """
import {{ sum, round, daysBetween, todayISO }} from '{utils}';
const SAFE_LIMIT = 12; const RISK_LIMIT = 16;
{code}
const courses = {courses};
const goals = {goals};
const out = courses.map((c) => {{
  const s = courseStats(c);
  return {{ id: c.id, totalWeight: s.totalWeight, gradedWeight: s.gradedWeight, pendingWeight: s.pendingWeight,
    points: s.points, average: s.average, minGrade: s.minGrade, neededToPass: s.neededToPass, maxFinal: s.maxFinal,
    statusKey: s.status.key, statusLabel: s.status.label, nextEval: s.nextEval ? s.nextEval.id : null,
    gradedCount: s.gradedCount, pendingCount: s.pendingCount, finished: s.finished,
    needed: goals.map((g) => s.neededFor(g)) }};
}});
console.log(JSON.stringify(out));
"""


def extract(text: str, name: str) -> str:
    m = re.search(rf"(?:export )?function {name}\(.*?\n\}}\n", text, re.S)
    if not m:
        raise SystemExit(f"No encontré la función {name} en courses.js")
    return m.group(0).replace("export function", "function")


def run_js(web: Path):
    src = (web / "js/modules/courses.js").read_text(encoding="utf-8")
    code = "\n".join(extract(src, n) for n in ("isGraded", "toneForNeeded", "courseStats"))
    utils = (web / "js/core/utils.js").resolve().as_uri()
    script = JS_HARNESS.format(utils=utils, code=code, courses=json.dumps(COURSES), goals=json.dumps(GOALS))
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "harness.mjs"
        f.write_text(script, encoding="utf-8")
        out = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
    if out.returncode:
        raise SystemExit(out.stderr)
    return json.loads(out.stdout)


def py_results():
    import datetime as dt
    from tools.unigym.stats import course_stats
    res = []
    for c in COURSES:
        s = course_stats(c, dt.date.today())
        res.append({"id": c["id"], "totalWeight": s["total_weight"], "gradedWeight": s["graded_weight"],
                    "pendingWeight": s["pending_weight"], "points": s["points"], "average": s["average"],
                    "minGrade": s["min_grade"], "neededToPass": s["needed_to_pass"], "maxFinal": s["max_final"],
                    "statusKey": s["status_key"], "statusLabel": s["status_label"],
                    "nextEval": s["next_eval"]["id"] if s["next_eval"] else None,
                    "gradedCount": s["graded_count"], "pendingCount": s["pending_count"], "finished": s["finished"],
                    "needed": [s["needed_for"](g) for g in GOALS]})
    return res


def same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 1e-9
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default=str(ROOT.parent / "uni-gym-tracker"))
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    js = run_js(Path(a.web))
    if a.dump:
        print(json.dumps({"courses": COURSES, "goals": GOALS, "expected": js}, ensure_ascii=False, indent=1))
        return
    bad = 0
    for j, p in zip(js, py_results()):
        for k in j:
            if not same(j[k], p[k]):
                bad += 1
                print(f"DIFERENCIA en {j['id']}.{k}: JS={j[k]!r} Python={p[k]!r}")
    print("Paridad OK" if not bad else f"{bad} diferencias")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
