"""Pruebas de las escrituras de UniGym (Fase 6.4): validación del STT, propuestas y mutaciones. Datos inventados."""
import copy
import datetime as dt
import unittest

from tests.test_unigym_readers import state as base_state
from tools.unigym.readers import Result
from tools.unigym.repo import UniGymDataError, normalize_state, validate_state
from tools.unigym.writers import WRITERS, Proposal, new_id, to_number

TUE = dt.date(2026, 11, 24)


def prep(name, args, st=None, today=TUE):
    return WRITERS[name](st if st is not None else base_state(), args, today, 1_700_000_000_000)


def applied(prop, st=None):
    st = normalize_state(copy.deepcopy(st if st is not None else base_state()))
    prop.apply(st)
    validate_state(st)           # lo que escribe JARVIS siempre debe seguir siendo válido para la web
    return st


class CrearTarea(unittest.TestCase):
    def test_hito_tarea_de_eda2_para_el_viernes(self):
        p = prep("crear_tarea", {"titulo": "tarea de EDA2", "curso": "EDA2", "fecha": "viernes"})
        self.assertIsInstance(p, Proposal)
        self.assertEqual(p.question, "¿Agrego la tarea «Tarea de EDA2» para viernes 27 de noviembre?")
        task = applied(p)["tasks"][-1]
        self.assertEqual((task["title"], task["due"], task["category"], task["priority"], task["done"]),
                         ("Tarea de EDA2", "2026-11-27", "Universidad", "media", False))
        self.assertTrue(task["id"].startswith("tar_"))
        self.assertEqual(task["createdAt"], 1_700_000_000_000)

    def test_titulo_con_curso_y_prioridad(self):
        p = prep("crear_tarea", {"titulo": "informe final", "curso": "eda dos", "fecha": "mañana", "prioridad": "alta"})
        self.assertIn("«EDA2: informe final»", p.question)
        self.assertEqual(applied(p)["tasks"][-1]["priority"], "alta")

    def test_curso_por_nombre_largo_no_se_repite_en_el_titulo(self):
        p = prep("crear_tarea", {"titulo": "práctica de redes", "curso": "redes", "fecha": "sin fecha"})
        self.assertIn("«práctica de redes»", p.question)
        self.assertNotIn(" para ", p.question)
        self.assertEqual(applied(p)["tasks"][-1]["due"], "")

    def test_sin_curso_es_personal(self):
        p = prep("crear_tarea", {"titulo": "comprar cuaderno", "fecha": "el 28", "curso": "ninguno"})
        self.assertEqual(applied(p)["tasks"][-1]["category"], "Personal")
        self.assertIn("sábado 28 de noviembre", p.question)

    def test_pregunta_si_falta_o_no_se_entiende(self):
        for args in ({}, {"fecha": "viernes"}, {"curso": "física", "titulo": "x"}, {"titulo": "y" * 200},
                     {"titulo": "algo", "fecha": "ayer"}, {"titulo": "algo", "fecha": "31 de noviembre"},
                     {"titulo": "algo", "fecha": "martes"}):          # "martes" dicho un martes: ambiguo
            r = prep("crear_tarea", args)
            self.assertIsInstance(r, Result, args)
            self.assertFalse(r.ok, args)
            self.assertTrue(r.needs, args)

    def test_no_duplica(self):
        r = prep("crear_tarea", {"titulo": "Práctica de EDA2", "curso": "eda2", "fecha": "hoy"})
        self.assertIsInstance(r, Result)
        self.assertTrue(r.ok)
        self.assertIn("No la dupliqué", r.text)

    def test_argumentos_basura(self):
        for args in ({"titulo": 5, "curso": None, "fecha": 3}, {"titulo": [], "curso": {}, "fecha": object()}, {"prioridad": 7}):
            r = prep("crear_tarea", args, st={})
            self.assertTrue(isinstance(r, (Proposal, Result)), args)


class CompletarTarea(unittest.TestCase):
    def test_encuentra_por_parecido_y_marca_hecha(self):
        p = prep("completar_tarea", {"titulo": "el informe de redes"})
        self.assertEqual(p.question, "¿Marco como hecha «Informe de redes»?")
        st = applied(p)
        self.assertTrue(next(t for t in st["tasks"] if t["id"] == "t1")["done"])
        self.assertFalse(next(t for t in st["tasks"] if t["id"] == "t2")["done"])

    def test_desconocida_ambigua_y_ya_hecha_preguntan(self):
        self.assertFalse(prep("completar_tarea", {"titulo": "limpiar el cuarto"}).ok)
        self.assertFalse(prep("completar_tarea", {"titulo": "ya hecha"}).ok)        # solo busca entre pendientes
        self.assertFalse(prep("completar_tarea", {}).ok)
        st = base_state()
        st["tasks"] = [{"id": "a", "title": "Informe de redes 1", "done": False}, {"id": "b", "title": "Informe de redes 2", "done": False}]
        r = prep("completar_tarea", {"titulo": "informe de redes"}, st)
        self.assertTrue(r.needs and "¿Cuál" in r.text)

    def test_si_la_tarea_desaparecio_no_inventa(self):
        p = prep("completar_tarea", {"titulo": "informe de redes"})
        st = base_state()
        st["tasks"] = []
        with self.assertRaises(UniGymDataError):
            applied(p, st)


class Habito(unittest.TestCase):
    def test_marca_hoy(self):
        p = prep("marcar_habito", {"habito": "leer"})
        self.assertEqual(p.question, "¿Marco «Leer» como cumplido hoy?")
        self.assertTrue(next(h for h in applied(p)["habits"] if h["id"] == "h2")["log"]["2026-11-24"])

    def test_ya_marcado_no_pide_confirmacion(self):
        r = prep("marcar_habito", {"habito": "agua"})
        self.assertIsInstance(r, Result)
        self.assertIn("ya estaba marcado", r.text)

    def test_ayer_si_futuro_no(self):
        self.assertIn("ayer", prep("marcar_habito", {"habito": "leer", "fecha": "ayer"}).question)
        self.assertFalse(prep("marcar_habito", {"habito": "leer", "fecha": "mañana"}).ok)
        self.assertFalse(prep("marcar_habito", {"habito": "leer", "fecha": "2026-10-01"}).ok)
        self.assertFalse(prep("marcar_habito", {"habito": "meditar"}).ok)


class Peso(unittest.TestCase):
    def test_registra_con_coma_o_punto(self):
        for raw in (70.5, "70.5", "70,5"):
            p = prep("registrar_peso", {"kg": raw})
            self.assertEqual(p.question, "¿Registro 70,5 kilos hoy?")
        w = applied(p)["gym"]["bodyWeight"][-1]
        self.assertEqual((w["kg"], w["date"]), (70.5, "2026-11-24"))
        self.assertTrue(w["id"].startswith("pc_"))

    def test_rangos_y_basura(self):
        for bad in (7.05, 705, 0, -70, "mucho", None, True):
            self.assertFalse(prep("registrar_peso", {"kg": bad}).ok, bad)

    def test_reemplaza_el_del_dia(self):
        st = base_state()
        st["gym"]["bodyWeight"] = [{"id": "pc_1", "date": "2026-11-24", "kg": 71.0}]
        p = prep("registrar_peso", {"kg": 70.5}, st)
        self.assertIn("Antes había 71", p.question)
        wl = applied(p, st)["gym"]["bodyWeight"]
        self.assertEqual([(w["id"], w["kg"]) for w in wl], [("pc_1", 70.5)])
        self.assertIn("Ya tenías", prep("registrar_peso", {"kg": 71}, st).text)


class Diario(unittest.TestCase):
    def test_agua_suma_por_defecto(self):
        p = prep("registrar_diario", {"campo": "agua", "valor": 1})
        self.assertEqual(p.question, "¿Sumo 1 vasos de agua hoy? Quedarías en 4.")
        self.assertEqual(applied(p)["dailyLogs"]["2026-11-24"]["water"], 4)

    def test_pasos_y_sueno_fijan(self):
        self.assertEqual(applied(prep("registrar_diario", {"campo": "pasos", "valor": 9000}))["dailyLogs"]["2026-11-24"]["steps"], 9000)
        p = prep("registrar_diario", {"campo": "sueño", "valor": "7,5"})
        self.assertEqual(applied(p)["dailyLogs"]["2026-11-24"]["sleep"], 7.5)

    def test_dia_nuevo_crea_registro_con_forma_de_la_web(self):
        st = applied(prep("registrar_diario", {"campo": "agua", "valor": 2, "fecha": "ayer"}))
        self.assertEqual(st["dailyLogs"]["2026-11-23"], {"steps": None, "sleep": None, "water": 2, "pomodoros": 0})

    def test_modo_explicito(self):
        p = prep("registrar_diario", {"campo": "agua", "valor": 6, "modo": "fijar"})
        self.assertEqual(applied(p)["dailyLogs"]["2026-11-24"]["water"], 6)
        p = prep("registrar_diario", {"campo": "pasos", "valor": 1000, "modo": "sumar"})
        self.assertEqual(applied(p)["dailyLogs"]["2026-11-24"]["steps"], 6200)

    def test_rechaza_lo_absurdo(self):
        for args in ({"campo": "agua", "valor": 60}, {"campo": "sueno", "valor": 30}, {"campo": "pasos", "valor": 500000},
                     {"campo": "pasos", "valor": 100.5}, {"campo": "calorias", "valor": 3}, {"campo": "agua", "valor": "x"},
                     {"campo": "agua", "valor": -1}, {"campo": "agua", "valor": 1, "fecha": "mañana"}, {}):
            self.assertFalse(prep("registrar_diario", args).ok, args)

    def test_suma_que_se_pasa_del_limite(self):
        self.assertFalse(prep("registrar_diario", {"campo": "agua", "valor": 38}).ok)   # 3 + 38 > 40


class Nota(unittest.TestCase):
    def test_registra_en_evaluacion_pendiente(self):
        p = prep("registrar_nota", {"curso": "redes", "evaluacion": "eva dos", "nota": 16})
        self.assertEqual(p.question, "¿Anoto 16 en EVA 2 de Redes de Computadoras?")
        e = next(e for e in applied(p)["courses"][1]["evals"] if e["id"] == "r2")
        self.assertEqual((e["grade"], e["status"]), (16, "calificada"))

    def test_reemplazo_y_sin_cambio(self):
        p = prep("registrar_nota", {"curso": "eda2", "evaluacion": "eva 1", "nota": 15.5})
        self.assertIn("Reemplaza el 14", p.question)
        self.assertIn("ya tenía 14", prep("registrar_nota", {"curso": "eda2", "evaluacion": "eva 1", "nota": 14}).text)

    def test_validaciones(self):
        for args in ({"curso": "redes", "evaluacion": "eva 2", "nota": 25}, {"curso": "redes", "evaluacion": "eva 2", "nota": -1},
                     {"curso": "redes", "evaluacion": "eva 2"}, {"curso": "redes", "nota": 15},
                     {"curso": "redes", "evaluacion": "laboratorio", "nota": 15}, {"curso": "física", "evaluacion": "eva 1", "nota": 15},
                     {"curso": "vacío", "evaluacion": "eva 1", "nota": 15}):
            r = prep("registrar_nota", args)
            self.assertIsInstance(r, Result, args)
            self.assertFalse(r.ok, args)

    def test_pregunta_lista_las_evaluaciones(self):
        self.assertIn("EVA 1, EVA 2 y EVA 3", prep("registrar_nota", {"curso": "redes", "evaluacion": "laboratorio 9", "nota": 15}).text)


class Utilidades(unittest.TestCase):
    def test_new_id_tiene_formato_de_la_web(self):
        a, b = new_id("tar", 1_700_000_000_000), new_id("tar", 1_700_000_000_000)
        self.assertRegex(a, r"^tar_[0-9a-z]{13}$")
        self.assertNotEqual(a, b)

    def test_to_number(self):
        self.assertEqual((to_number("70,5"), to_number(3), to_number(None), to_number(True), to_number("x")), (70.5, 3.0, None, None, None))

    def test_ningun_prepare_modifica_el_estado(self):
        for name, args in {"crear_tarea": {"titulo": "algo", "fecha": "viernes"}, "completar_tarea": {"titulo": "informe"},
                           "marcar_habito": {"habito": "leer"}, "registrar_peso": {"kg": 70},
                           "registrar_diario": {"campo": "agua", "valor": 1}, "registrar_nota": {"curso": "redes", "evaluacion": "eva 2", "nota": 16}}.items():
            st = base_state()
            before = copy.deepcopy(st)
            prep(name, args, st)
            self.assertEqual(st, before, name)


if __name__ == "__main__":
    unittest.main()
