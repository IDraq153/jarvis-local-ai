"""Pruebas de las herramientas de lectura de UniGym (Fase 6.2) con un estado inventado."""
import copy
import datetime as dt
import unittest

from tools.unigym.readers import READERS, fmt
from tools.unigym.specs import SPECS
from tools import READ

TUE = dt.date(2026, 11, 24)   # martes = día 2


def state():
    return {
        "settings": {"goals": {"steps": 8000, "sleep": 8, "water": 8}},
        "courses": [
            {"id": "cur_eda2", "name": "Estructura de Datos y Algoritmos 2", "credits": 4, "targetGrade": 15, "minGrade": 10.5,
             "evals": [{"id": "e1", "name": "EVA 1", "weight": 30, "grade": 14, "status": "calificada", "date": "2026-10-01"},
                       {"id": "e2", "name": "EVA 2", "weight": 30, "grade": 16, "status": "calificada", "date": "2026-11-01"},
                       {"id": "e3", "name": "Examen final", "weight": 40, "status": "pendiente", "date": "2026-12-10"}]},
            {"id": "cur_redes", "name": "Redes de Computadoras", "credits": 3, "targetGrade": 15,
             "evals": [{"id": "r1", "name": "EVA 1", "weight": 50, "grade": 12, "status": "calificada"},
                       {"id": "r2", "name": "EVA 2", "weight": 25, "status": "pendiente", "date": "2026-11-24"},
                       {"id": "r3", "name": "EVA 3", "weight": 25, "status": "pendiente"}]},
            {"id": "cur_vacio", "name": "Curso Vacío", "credits": 2, "evals": []},
        ],
        "schedule": {"blocks": [
            {"id": "b1", "title": "Redes", "type": "clase", "day": 2, "start": "10:00", "end": "12:00", "room": "A-204"},
            {"id": "b2", "title": "EDA2", "type": "clase", "day": 2, "start": "08:00", "end": "10:00", "room": ""},
            {"id": "b3", "title": "Libre", "type": "libre", "day": 2, "start": "13:00", "end": "14:00"},
            {"id": "b4", "title": "Estadística", "type": "clase", "day": 3, "start": "08:00", "end": "09:00"}]},
        "tasks": [
            {"id": "t1", "title": "Informe de redes", "priority": "alta", "due": "2026-11-27", "done": False, "createdAt": 1},
            {"id": "t2", "title": "Práctica de EDA2", "priority": "media", "due": "2026-11-24", "done": False, "createdAt": 2},
            {"id": "t3", "title": "Leer capítulo", "priority": "baja", "due": "", "done": False, "createdAt": 3},
            {"id": "t4", "title": "Entrega vieja de EDA2", "priority": "alta", "due": "2026-11-20", "done": False, "createdAt": 0},
            {"id": "t5", "title": "Ya hecha", "priority": "alta", "due": "2026-11-24", "done": True, "createdAt": 4}],
        "gym": {"routines": [{"id": "r1", "name": "Pecho y tríceps", "focus": "empuje", "days": [2, 5],
                              "exercises": [{"name": "Press banca", "sets": 4, "repMin": 8, "repMax": 10},
                                            {"name": "Fondos", "sets": 3, "repMin": 12, "repMax": 12}]}],
                "sessions": [], "activeSession": None, "bodyWeight": []},
        "dailyLogs": {"2026-11-24": {"steps": 5200, "sleep": 7.5, "water": 3}},
        "habits": [{"id": "h1", "name": "Agua", "log": {"2026-11-24": True, "2026-11-23": True, "2026-11-22": True}},
                   {"id": "h2", "name": "Leer", "log": {}}],
    }


def run(name, args=None, st=None, today=TUE):
    return READERS[name](st if st is not None else state(), args or {}, today)


class Agenda(unittest.TestCase):
    def test_que_tengo_hoy(self):   # el caso del hito
        r = run("consultar_agenda")
        self.assertTrue(r.ok)
        self.assertIn("2 bloques hoy", r.text)
        self.assertLess(r.text.index("EDA2 de 08:00"), r.text.index("Redes de 10:00"))   # ordenados por hora
        self.assertIn("Redes de 10:00 a 12:00 en A-204", r.text)
        self.assertNotIn("Libre", r.text)                                               # los bloques libres no se dicen
        self.assertIn("Evaluación EVA 2 de Redes de Computadoras", r.text)
        self.assertIn("Tareas para hoy: Práctica de EDA2", r.text)
        self.assertIn("1 tarea vencida", r.text)
        self.assertNotIn("Ya hecha", r.text)

    def test_manana_y_dia_vacio(self):
        self.assertIn("Estadística de 08:00 a 09:00", run("consultar_agenda", {"fecha": "mañana"}).text)
        self.assertIn("No tienes clases ni tareas para el domingo", run("consultar_agenda", {"fecha": "el domingo"}).text.replace("para domingo", "para el domingo"))

    def test_fecha_invalida_pregunta(self):
        r = run("consultar_agenda", {"fecha": "31 de noviembre"})
        self.assertFalse(r.ok)
        self.assertIn("no existe", r.needs)

    def test_estado_vacio_no_revienta(self):
        r = run("consultar_agenda", st={})
        self.assertIn("No tienes clases ni tareas", r.text)


class Tareas(unittest.TestCase):
    def test_orden_por_fecha_y_vencidas(self):
        r = run("tareas_pendientes")
        self.assertIn("4 tareas pendientes", r.text)
        self.assertLess(r.text.index("Entrega vieja"), r.text.index("Práctica"))
        self.assertLess(r.text.index("Práctica"), r.text.index("Informe de redes"))
        self.assertIn("vencida", r.text)
        self.assertNotIn("Ya hecha", r.text)
        self.assertTrue(r.text.rstrip(".").endswith("Leer capítulo"))                 # sin fecha, al final

    def test_filtro_por_curso(self):
        r = run("tareas_pendientes", {"curso": "EDA 2"})
        self.assertIn("2 tareas pendientes de Estructura", r.text)
        self.assertNotIn("Informe de redes", r.text)

    def test_curso_desconocido_pregunta(self):
        r = run("tareas_pendientes", {"curso": "física"})
        self.assertFalse(r.ok)
        self.assertIn("No encontré un curso", r.needs)

    def test_sin_tareas(self):
        st = state()
        st["tasks"] = []
        self.assertIn("No tienes tareas pendientes", run("tareas_pendientes", st=st).text)

    def test_corta_la_lista(self):
        st = state()
        st["tasks"] = [{"id": f"t{i}", "title": f"Tarea {i}", "due": "", "done": False, "createdAt": i} for i in range(9)]
        r = run("tareas_pendientes", st=st)
        self.assertIn("Y 4 más", r.text)


class Gym(unittest.TestCase):
    def test_rutina_de_hoy(self):
        r = run("rutina_gym")
        self.assertIn("Hoy te toca Pecho y tríceps (empuje)", r.text)
        self.assertIn("Press banca 4 por 8 a 10", r.text)
        self.assertIn("Fondos 3 por 12", r.text)

    def test_descanso(self):
        self.assertIn("descansas del gym", run("rutina_gym", {"fecha": "mañana"}).text)

    def test_sesion_en_curso(self):
        st = state()
        st["gym"]["activeSession"] = {"entries": []}
        self.assertIn("en curso", run("rutina_gym", st=st).text)


class Notas(unittest.TestCase):
    def test_un_curso(self):
        r = run("notas_curso", {"curso": "EDA2"})
        self.assertIn("llevas 15 de promedio. Ya diste: EVA 1 14; EVA 2 16. Te falta Examen final (40%)", r.text)

    def test_todos(self):
        r = run("notas_curso")
        self.assertIn("promedio del ciclo", r.text)
        self.assertNotIn("Curso Vacío", r.text)

    def test_curso_sin_evaluaciones_y_sin_notas(self):
        self.assertIn("no tiene evaluaciones", run("notas_curso", {"curso": "vacío"}).text)

    def test_numeros_con_coma(self):
        self.assertEqual((fmt(14.285), fmt(15.0), fmt(13.5), fmt(None)), ("14,29", "15", "13,5", "sin dato"))


class NotaNecesaria(unittest.TestCase):
    def test_hito_redes_a_15(self):
        r = run("nota_necesaria", {"curso": "redes", "meta": 15})
        # puntos = 12*50/100 = 6; falta 50%; (15-6)/0.5 = 18
        self.assertAlmostEqual(r.data["necesaria"], 18.0)
        self.assertIn("promediar 18 en lo que falta: EVA 2 (25%) y EVA 3 (25%)", r.text)

    def test_una_sola_pendiente_dice_la_evaluacion(self):
        r = run("nota_necesaria", {"curso": "EDA2", "meta": 17})
        # puntos = 14*.3+16*.3 = 9; (17-9)/.4 = 20
        self.assertIn("sacar 20 en Examen final", r.text)

    def test_sin_meta_es_para_aprobar_con_la_nota_minima_del_curso(self):
        r = run("nota_necesaria", {"curso": "redes"})
        self.assertEqual((r.data["meta"], r.data["para_aprobar"]), (10.5, True))
        # puntos 6; (10,5-6)/0,5 = 9
        self.assertIn("Para aprobar Redes de Computadoras (10,5) necesitas promediar 9 en lo que falta", r.text)
        self.assertFalse(run("nota_necesaria", {"curso": "redes", "meta": 12}).data["para_aprobar"])

    def test_ya_no_alcanza_y_ya_asegurado(self):
        self.assertIn("ya no alcanza", run("nota_necesaria", {"curso": "redes", "meta": 19}).text)
        self.assertIn("aunque saques 0", run("nota_necesaria", {"curso": "EDA2", "meta": 8}).text)
        st = state()
        st["courses"][1]["evals"] = [{"id": "x1", "name": "EVA 1", "weight": 70, "grade": 18, "status": "calificada"},
                                     {"id": "x2", "name": "EVA 2", "weight": 30, "status": "pendiente"}]
        self.assertIn("Ya tienes asegurado aprobar en Redes de Computadoras", run("nota_necesaria", {"curso": "redes"}, st=st).text)

    def test_meta_invalida_o_curso_ausente_pregunta(self):
        self.assertFalse(run("nota_necesaria", {"curso": "redes", "meta": 25}).ok)
        self.assertFalse(run("nota_necesaria", {"curso": "redes", "meta": "mucho"}).ok)
        self.assertFalse(run("nota_necesaria", {}).ok)


class Diario(unittest.TestCase):
    def test_progreso(self):
        r = run("progreso_diario")
        self.assertIn("Pasos: 5200 de 8000", r.text)
        self.assertIn("Sueño: 7,5 horas de 8", r.text)
        self.assertIn("Agua: 3 vasos de 8", r.text)

    def test_dia_sin_registro(self):
        self.assertIn("Pasos: sin registrar", run("progreso_diario", {"fecha": "ayer"}).text)

    def test_habitos(self):
        r = run("estado_habitos")
        self.assertIn("1 de 2 hábitos", r.text)
        self.assertIn("Te faltan: Leer", r.text)
        self.assertIn("mejor racha es Agua, 3 días", r.text)


class Catalogo(unittest.TestCase):
    def test_cada_herramienta_de_lectura_tiene_lector(self):
        reads = {s.name for s in SPECS if s.level == READ}
        self.assertEqual(reads, set(READERS))

    def test_ningun_lector_modifica_el_estado_ni_revienta_con_basura(self):
        for name, fn in READERS.items():
            for args in ({}, {"curso": 5, "fecha": 7, "meta": None}, {"curso": "", "fecha": "???"}):
                st = state()
                before = copy.deepcopy(st)
                r = fn(st, args, TUE)
                self.assertTrue(r.text, name)
                self.assertEqual(st, before, name)
            self.assertTrue(fn({}, {}, TUE).text, name)             # estado vacío


if __name__ == "__main__":
    unittest.main()


class EvalSpec(unittest.TestCase):
    def test_entiende_lo_que_dice_dario(self):
        from tools.unigym.readers import eval_spec as f
        self.assertEqual(f("ese es el tercer ejercicio individual del curso estructura de datos"), ("ejercicio", 3, False))
        self.assertEqual(f("quiero la evaluación 2 del curso de estructura de datos 2"), ("", 2, False))
        self.assertEqual(f("el último examen de estructura de datos 2"), ("examen", None, True))
        self.assertEqual(f("el tercero"), (None, 3, False))
        self.assertEqual(f("ejercicio dos"), ("ejercicio", 2, False))
        self.assertEqual(f("estructura de datos 2"), (None, None, False))     # el 2 del curso no es la evaluación 2


class TemasExamen(unittest.TestCase):
    TODAY = dt.date(2026, 11, 24)

    def st(self):
        st = state()
        for c in st["courses"]:
            for e in c["evals"]:
                if c["name"].startswith("Redes") and e["name"] == "EVA 2":
                    e.update(topics=["Subnetting VLSM", "Rutas estáticas", "VLAN y trunking"], guide=["Repasar VLSM", "Hacer 10 ejercicios"])
                if c["name"].startswith("Estructura") and e["name"] == "Examen final":
                    e.update(topics=[f"Tema {i}" for i in range(1, 12)])
        return st

    def run_(self, **args):
        return READERS["temas_examen"](self.st(), args, self.TODAY)

    def test_el_ultimo_examen_de_un_curso_es_el_final_aunque_no_se_haya_rendido(self):
        r = self.run_(curso="EDA2", evaluacion="examen", cual="ultimo")
        self.assertIn("Examen final de Estructura de Datos y Algoritmos 2", r.text)
        self.assertIn("Tema 1; Tema 2", r.text)

    def test_examen_no_es_proyecto(self):                                  # fallo real de Dario: devolvía el proyecto
        st = self.st()
        eda = st["courses"][0]
        eda["evals"].append({"id": "p", "name": "Proyecto de Estructura", "date": "2026-11-22", "status": "calificada",
                             "topics": ["Lo que pida el enunciado"], "guide": []})
        eda["evals"].append({"id": "x", "name": "Examen parcial 2", "date": "2026-12-15", "status": "pendiente", "topics": ["Grafos"]})
        r = READERS["temas_examen"](st, {"curso": "EDA2", "evaluacion": "examen", "cual": "ultimo"}, self.TODAY)
        self.assertIn("Examen parcial 2", r.text)
        self.assertNotIn("Proyecto", r.text)
        r = READERS["temas_examen"](st, {"curso": "EDA2", "evaluacion": "proyecto", "cual": "ultimo"}, self.TODAY)
        self.assertIn("Proyecto de Estructura", r.text)
        r = READERS["temas_examen"](st, {"curso": "EDA2", "evaluacion": "exámenes"}, self.TODAY)   # sin cual: el próximo examen
        self.assertIn("Examen final", r.text)

    def test_tipo_inexistente(self):
        r = self.run_(curso="redes", evaluacion="proyecto")
        self.assertEqual(r.text, "En Redes de Computadoras no hay proyectos. Tiene EVA 1, EVA 2 y EVA 3. ¿De cuál quieres los temas?")
        self.assertEqual(r.data["slot"], "evaluacion")

    def test_el_proximo_por_defecto(self):
        r = self.run_()
        self.assertTrue(r.ok)
        self.assertIn("Redes de Computadoras", r.text)                      # EVA 2 es hoy: sigue pendiente

    def test_por_curso_y_nombre(self):
        r = self.run_(curso="redes", evaluacion="EVA 2")
        self.assertIn("Subnetting VLSM", r.text)
        r = self.run_(curso="EDA2")
        self.assertIn("Examen final", r.text)
        self.assertIn("Tema 4. Y 7 más, ¿te los digo?", r.text)              # dice 4 y ofrece el resto

    def test_plan_de_estudio(self):
        r = self.run_(curso="redes", evaluacion="EVA 2", plan=True)
        self.assertEqual(r.text.split(": ", 1)[1], "Repasar VLSM; Hacer 10 ejercicios.")

    def test_sin_temas_cargados(self):
        r = self.run_(curso="redes", evaluacion="EVA 3")
        self.assertIn("no tengo temas cargados", r.text)

    def test_evaluacion_inexistente_se_pregunta(self):
        r = self.run_(curso="redes", evaluacion="laboratorio 9")
        self.assertIsNotNone(r.needs)

    def test_no_modifica_el_estado(self):
        st = self.st()
        import copy
        before = copy.deepcopy(st)
        READERS["temas_examen"](st, {"cual": "ultimo"}, self.TODAY)
        self.assertEqual(st, before)
