"""Pruebas de stats (paridad con el JS de la web), fechas habladas y búsqueda de nombres (Fase 6.2)."""
import datetime as dt
import json
import unittest
from pathlib import Path

from tools.unigym.dates import DateError, resolve_date, spoken_date
from tools.unigym.names import acronym, resolve_course, resolve_eval, resolve_habit, resolve_task
from tools.unigym.stats import (blocks_for_day, course_stats, habit_streak, js_round, routine_for_day, num)

GOLDEN = json.loads((Path(__file__).parent / "golden" / "course_stats.json").read_text(encoding="utf-8"))
TUE = dt.date(2026, 11, 24)   # martes


class StatsParity(unittest.TestCase):
    def test_matches_values_computed_by_the_real_js(self):
        """Los esperados salen de ejecutar courseStats de la web (scripts/check_stats_parity.py --dump)."""
        self.assertGreaterEqual(len(GOLDEN["courses"]), 10)
        for course, exp in zip(GOLDEN["courses"], GOLDEN["expected"]):
            s = course_stats(course, TUE)
            got = {"totalWeight": s["total_weight"], "gradedWeight": s["graded_weight"],
                   "pendingWeight": s["pending_weight"], "points": s["points"], "average": s["average"],
                   "minGrade": s["min_grade"], "neededToPass": s["needed_to_pass"], "maxFinal": s["max_final"],
                   "statusKey": s["status_key"], "statusLabel": s["status_label"],
                   "nextEval": s["next_eval"]["id"] if s["next_eval"] else None,
                   "gradedCount": s["graded_count"], "pendingCount": s["pending_count"], "finished": s["finished"],
                   "needed": [s["needed_for"](g) for g in GOLDEN["goals"]]}
            for k, v in exp.items():
                if k == "id":
                    continue
                with self.subTest(course=course["id"], campo=k):
                    if isinstance(v, list):
                        for a, b in zip(got[k], v):
                            self.assertTrue((a is None and b is None) or abs(a - b) < 1e-9, (k, a, b))
                    elif isinstance(v, float):
                        self.assertAlmostEqual(got[k], v, places=9)
                    else:
                        self.assertEqual(got[k], v)

    def test_js_round_rounds_halves_up(self):
        self.assertEqual(js_round(2.5, 0), 3)
        self.assertEqual(js_round(0.125, 2), 0.13)
        self.assertEqual(js_round(14.285, 2), 14.29)

    def test_num_mimics_number_or_zero(self):
        self.assertEqual((num("25"), num("2,5"), num(None), num("x"), num(7)), (25, 2.5, 0, 0, 7))

    def test_habit_streak_counts_back_from_today_or_yesterday(self):
        log = {"2026-11-24": True, "2026-11-23": True, "2026-11-22": True, "2026-11-20": True}
        self.assertEqual(habit_streak({"log": log}, TUE), 3)
        del log["2026-11-24"]
        self.assertEqual(habit_streak({"log": log}, TUE), 2)   # hoy aún no marcado: cuenta desde ayer
        self.assertEqual(habit_streak({"log": {}}, TUE), 0)

    def test_blocks_and_routine_for_day(self):
        state = {"schedule": {"blocks": [{"day": 2, "start": "10:00", "title": "B"}, {"day": 2, "start": "08:00", "title": "A"},
                                          {"day": 3, "start": "07:00", "title": "C"}]},
                 "gym": {"routines": [{"name": "Pecho", "days": [2, 5]}, {"name": "Pierna", "days": [3]}]}}
        self.assertEqual([b["title"] for b in blocks_for_day(state, 2)], ["A", "B"])
        self.assertEqual(routine_for_day(state, 5)["name"], "Pecho")
        self.assertIsNone(routine_for_day(state, 7))


class DateTests(unittest.TestCase):
    def iso(self, expr, **kw):
        return resolve_date(expr, TUE, **kw).iso

    def test_relative_days(self):
        self.assertEqual(self.iso(None), "2026-11-24")
        self.assertTrue(resolve_date("", TUE).defaulted)
        self.assertEqual(self.iso("hoy"), "2026-11-24")
        self.assertEqual(self.iso("mañana"), "2026-11-25")
        self.assertEqual(self.iso("Pasado mañana"), "2026-11-26")
        self.assertEqual(self.iso("ayer"), "2026-11-23")

    def test_weekdays_hito(self):
        self.assertEqual(self.iso("el viernes"), "2026-11-27")      # el caso del hito
        self.assertEqual(self.iso("viernes"), "2026-11-27")
        self.assertEqual(self.iso("para el lunes"), "2026-11-30")
        self.assertEqual(self.iso("el sábado"), "2026-11-28")
        self.assertEqual(self.iso("el próximo viernes"), "2026-11-27")
        self.assertEqual(self.iso("el miércoles que viene"), "2026-11-25")

    def test_same_weekday_is_asked_not_guessed(self):
        r = resolve_date("el martes", TUE)
        self.assertIsNotNone(r.ambiguous)
        self.assertEqual(self.iso("el martes", same_weekday="today"), "2026-11-24")
        self.assertEqual(self.iso("el martes", same_weekday="next"), "2026-12-01")
        self.assertEqual(self.iso("el próximo martes"), "2026-12-01")

    def test_day_of_month_and_explicit_dates(self):
        self.assertEqual(self.iso("el 28"), "2026-11-28")
        self.assertEqual(self.iso("el 3"), "2026-12-03")            # 3 de este mes ya pasó: el siguiente
        self.assertEqual(self.iso("28 de noviembre"), "2026-11-28")
        self.assertEqual(self.iso("veintiocho de noviembre"), "2026-11-28")
        self.assertEqual(self.iso("1 de diciembre de 2026"), "2026-12-01")
        self.assertEqual(self.iso("28/11"), "2026-11-28")
        self.assertEqual(self.iso("2/3/2027"), "2027-03-02")
        self.assertEqual(self.iso("2026-12-01"), "2026-12-01")
        self.assertEqual(self.iso("15 de febrero"), "2027-02-15")   # febrero ya pasó este año

    def test_invalid_dates_raise_speakable_errors(self):
        for bad in ("31 de noviembre", "30 de febrero", "la semana que viene", "blablabla", "32"):
            with self.assertRaises(DateError, msg=bad):
                resolve_date(bad, TUE)

    def test_past_dates_rejected_for_deadlines(self):
        with self.assertRaises(DateError):
            resolve_date("ayer", TUE, allow_past=False)
        self.assertEqual(resolve_date("hoy", TUE, allow_past=False).iso, "2026-11-24")

    def test_spoken_date(self):
        self.assertEqual(spoken_date("2026-11-24", TUE), "hoy")
        self.assertEqual(spoken_date("2026-11-25", TUE), "mañana")
        self.assertEqual(spoken_date("2026-11-27", TUE), "viernes 27 de noviembre")


STATE = {"courses": [
    {"id": "cur_eda2", "name": "Estructura de Datos y Algoritmos 2", "evals": [
        {"id": "e1", "name": "EVA 1"}, {"id": "e2", "name": "EVA 2"}, {"id": "e3", "name": "Examen final"}]},
    {"id": "cur_redes", "name": "Redes de Computadoras", "evals": []},
    {"id": "cur_gestfin", "name": "Gestión Financiera", "evals": []},
    {"id": "cur_estad", "name": "Estadística", "evals": []},
    {"id": "cur_prog1", "name": "Programación 1", "evals": []},
    {"id": "cur_prog2", "name": "Programación 2", "evals": []}],
    "tasks": [{"id": "t1", "title": "Informe de redes", "done": False}, {"id": "t2", "title": "Práctica de EDA2", "done": False},
              {"id": "t3", "title": "Informe antiguo", "done": True}],
    "habits": [{"id": "h1", "name": "Tomar agua", "log": {}}, {"id": "h2", "name": "Leer", "log": {}}]}


class NameTests(unittest.TestCase):
    def course(self, q):
        r = resolve_course(q, STATE)
        return r.item["id"] if r.found else ("ambiguo" if r.ambiguous else None)

    def test_stt_variants_of_the_same_course(self):
        for q in ("EDA2", "EDA 2", "eda dos", "Eda II", "estructura de datos y algoritmos 2"):
            self.assertEqual(self.course(q), "cur_eda2", q)
        self.assertEqual(self.course("redes"), "cur_redes")
        self.assertEqual(self.course("gestion financiera"), "cur_gestfin")
        self.assertEqual(self.course("GESTFIN"), "cur_gestfin")
        self.assertEqual(self.course("estadistica"), "cur_estad")

    def test_unknown_course_is_none_not_a_guess(self):
        self.assertIsNone(self.course("fisica cuantica"))
        self.assertIsNone(self.course(""))
        self.assertIsNone(self.course(None))

    def test_close_candidates_are_ambiguous(self):
        self.assertEqual(self.course("programación"), "ambiguo")
        self.assertEqual(self.course("programación 2"), "cur_prog2")

    def test_acronym(self):
        self.assertEqual(acronym("Estructura de Datos y Algoritmos 2"), "eda2")

    def test_eval_task_habit(self):
        eda = STATE["courses"][0]
        self.assertEqual(resolve_eval("eva dos", eda).item["id"], "e2")
        self.assertEqual(resolve_eval("el examen final", eda).item["id"], "e3")
        self.assertEqual(resolve_task("informe de redes", STATE).item["id"], "t1")
        self.assertEqual(resolve_task("la práctica", STATE).item["id"], "t2")
        self.assertFalse(resolve_task("informe antiguo", STATE).found)     # ya hecha: no se busca entre pendientes
        self.assertEqual(resolve_habit("agua", STATE).item["id"], "h1")


if __name__ == "__main__":
    unittest.main()


class FraseConFecha(unittest.TestCase):
    T = dt.date(2026, 11, 24)

    def test_encuentra_la_fecha(self):
        from tools.unigym.dates import find_date_phrase as f
        self.assertEqual(f("no aun tengo tiempo para este sabado probablemente", self.T), "sabado")
        self.assertEqual(f("sí, estudiaré el sábado", self.T), "sabado")
        self.assertEqual(f("no, mejor el 2 de diciembre", self.T), "2 de diciembre")
        self.assertEqual(f("mejor pasado mañana", self.T), "pasado manana")
        self.assertEqual(f("para el 28/11", self.T), "28/11")
        self.assertEqual(f("el próximo miércoles", self.T), "proximo miercoles")

    def test_no_inventa_fechas(self):
        from tools.unigym.dates import find_date_phrase as f
        for t in ("no gracias", "tengo 3 tareas", "sí", "", "cancela eso", "el 28"):
            self.assertIsNone(f(t, self.T), t)


class CursoConRelleno(unittest.TestCase):
    def test_limpia_la_frase(self):
        from tools.unigym import names
        self.assertEqual(names.clean_course_query("agrega una tarea para el curso redes"), "redes")
        from tests.test_unigym_readers import state
        r = names.resolve_course("agrega una tarea para el curso redes", state())
        self.assertTrue(r.found)
        self.assertEqual(r.item["name"], "Redes de Computadoras")


class CursosAutomaticos(unittest.TestCase):
    def state(self):
        return {"courses": [{"id": "a", "name": "Internet de las Cosas"}, {"id": "b", "name": "Programación Web"},
                            {"id": "c", "name": "Gestión Financiera"}, {"id": "d", "name": "Base de Datos II"}]}

    def test_abreviaturas(self):
        from tools.unigym import names
        for q, want in (("iot", "Internet de las Cosas"), ("progra web", "Programación Web"), ("finanzas", "Gestión Financiera"),
                        ("bd 2", "Base de Datos II")):
            r = names.resolve_course(q, self.state())
            self.assertTrue(r.found, q)
            self.assertEqual(r.item["name"], want, q)

    def test_siglas_cortas_mal_oidas(self):
        from tools.unigym import names
        st = {"courses": [{"id": "x", "name": "AEA"}, {"id": "y", "name": "Redes de Computadoras"}]}
        for q in ("aaa", "aéa", "AEA"):
            self.assertEqual(names.resolve_course(q, st).item["name"], "AEA", q)
        st["courses"].append({"id": "z", "name": "AAA"})                 # dos a una letra de distancia: no adivina
        self.assertFalse(names.resolve_course("aba", st).found)

    def test_respaldo_semantico(self):
        from tools.unigym import names
        old = names.SEMANTIC_FALLBACK
        try:
            names.SEMANTIC_FALLBACK = lambda q, labels: (labels.index("Gestión Financiera"), 0.7, 0.4)
            self.assertEqual(names.resolve_course("la de plata y bancos", self.state()).item["name"], "Gestión Financiera")
            names.SEMANTIC_FALLBACK = lambda q, labels: (0, 0.62, 0.60)                    # sin margen: no adivina
            self.assertFalse(names.resolve_course("algo raro", self.state()).found)
        finally:
            names.SEMANTIC_FALLBACK = old
