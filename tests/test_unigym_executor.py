"""Pruebas del ejecutor: confirmación por voz, copia previa, registro y conflictos (Fase 6.4). Datos inventados."""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_unigym_readers import state as base_state
from tools.unigym.audit import AuditLog
from tools.unigym.confirm import ConfirmationGate, classify_answer
from tools.unigym.executor import UniGymTools
from tools.unigym.repo import JsonFileRepository

TUE = dt.date(2026, 11, 24)


class Classify(unittest.TestCase):
    def test_afirmaciones_claras(self):
        for t in ("sí", "Sí.", "si claro", "dale", "OK", "sí, por favor", "confirmo", "hazlo", "sí hazlo jarvis",
                  "correcto", "adelante", "de acuerdo", "guárdalo", "es correcto"):
            self.assertEqual(classify_answer(t), "yes", t)

    def test_negaciones_y_dudas_cancelan(self):
        for t in ("no", "No, gracias", "cancela", "mejor no", "no no", "sí no", "sí pero para el lunes", "olvídalo", "negativo"):
            self.assertIn(classify_answer(t), ("no", "other"), t)
            self.assertNotEqual(classify_answer(t), "yes", t)

    def test_frases_largas_o_ajenas_no_confirman(self):
        for t in ("así es", "", "   ", "¿qué tengo hoy?", "sí claro y además dime qué tengo mañana por favor", "mañana", "el viernes",
                  "gracias", "ya veo", "hola jarvis"):
            self.assertNotEqual(classify_answer(t), "yes", t)


class FlowBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        src = d / "r.json"
        src.write_text(json.dumps({"app": "uni-gym-tracker", "version": 1, "data": base_state()}), encoding="utf-8")
        self.repo = JsonFileRepository.from_backup(src, d / "data" / "unigym.json", keep=10)
        self.audit = AuditLog(":memory:")
        self.now = [1000.0]
        self.gate = ConfirmationGate(ttl_s=60, clock=lambda: self.now[0])
        self.tools = UniGymTools(self.repo, self.audit, today=lambda: TUE, gate=self.gate)
        self.backups = d / "data" / "backups"

    def tearDown(self):
        self._tmp.cleanup()

    def statuses(self):
        return [(r["tool"], r["status"]) for r in self.audit.tail(50)]


class Hito(FlowBase):
    def test_agrega_una_tarea_de_eda2_para_el_viernes(self):
        out = self.tools.run("crear_tarea", {"titulo": "tarea de EDA2", "curso": "EDA2", "fecha": "viernes"},
                             heard="agrega una tarea de EDA2 para el viernes")
        self.assertEqual((out.kind, out.text), ("confirm", "¿Agrego la tarea «Tarea de EDA2» para viernes 27 de noviembre?"))
        self.assertTrue(self.tools.has_pending)
        self.assertEqual(self.repo.read().revision, 1)                     # todavía NO se guardó nada
        self.assertEqual(len(self.repo.read().state["tasks"]), 5)

        done = self.tools.answer("sí")
        self.assertEqual((done.kind, done.text, done.consumed), ("done", "Listo, tarea agregada.", True))
        snap = self.repo.read()
        self.assertEqual((snap.revision, snap.by), (2, "jarvis"))
        self.assertEqual(snap.state["tasks"][-1]["due"], "2026-11-27")
        self.assertEqual(len(list(self.backups.glob("unigym-*-r1.json"))), 1)   # copia previa
        self.assertFalse(self.tools.has_pending)
        self.assertEqual(self.statuses(), [("crear_tarea", "proposed"), ("crear_tarea", "confirmed")])
        last = self.audit.tail(1)[0]
        self.assertEqual((last["revision_before"], last["revision_after"]), (1, 2))

    def test_la_nueva_tarea_aparece_en_la_agenda(self):
        self.tools.run("crear_tarea", {"titulo": "informe", "curso": "EDA2", "fecha": "viernes"})
        self.tools.answer("sí")
        out = self.tools.run("consultar_agenda", {"fecha": "viernes"})
        self.assertIn("EDA2: informe", out.text)

    def test_que_tengo_hoy_es_lectura_libre(self):
        out = self.tools.run("consultar_agenda", {}, heard="¿qué tengo hoy?")
        self.assertEqual(out.kind, "answer")
        self.assertIn("2 bloques hoy", out.text)
        self.assertFalse(self.tools.has_pending)
        self.assertEqual(self.repo.read().revision, 1)
        self.assertEqual(self.statuses(), [("consultar_agenda", "read")])


class DialogoGuiado(FlowBase):
    """Si falta un dato, JARVIS lo pregunta y la frase siguiente lo rellena; nada se guarda hasta el sí final."""

    def ask(self, args, heard=None):
        heard = heard or " ".join(str(v) for v in args.values()) or "crear tarea"
        return self.tools.run("crear_tarea", args, heard=heard)

    def test_solo_crear_tarea_pregunta_titulo_curso_y_fecha(self):
        o = self.ask({})
        self.assertEqual((o.kind, o.text), ("ask", "¿Qué tarea agrego?"))
        self.assertTrue(self.tools.has_pending)
        o = self.tools.answer("estudiar para el parcial")
        self.assertEqual((o.kind, o.text), ("ask", "¿Es de algún curso? Dime cuál, o «ninguno»."))
        o = self.tools.answer("EDA 2")
        self.assertEqual(o.kind, "ask")
        self.assertIn("¿Para qué día es?", o.text)
        o = self.tools.answer("el viernes")
        self.assertEqual((o.kind, o.text), ("confirm", "¿Agrego la tarea «EDA2: estudiar para el parcial» para viernes 27 de noviembre?"))
        self.assertEqual(self.repo.read().revision, 1)                         # aún nada guardado
        self.assertEqual(self.tools.answer("sí").kind, "done")
        t = self.repo.read().state["tasks"][-1]
        self.assertEqual((t["due"], t["category"], t["title"]), ("2026-11-27", "Universidad", "EDA2: estudiar para el parcial"))

    def test_ninguno_y_sin_fecha(self):
        self.ask({"titulo": "comprar cuaderno"})
        o = self.tools.answer("ninguno")
        self.assertIn("¿Para qué día es?", o.text)
        o = self.tools.answer("sin fecha")
        self.assertEqual((o.kind, o.text), ("confirm", "¿Agrego la tarea «comprar cuaderno»?"))
        self.tools.answer("sí")
        t = self.repo.read().state["tasks"][-1]
        self.assertEqual((t["due"], t["category"]), ("", "Personal"))

    def test_con_todo_dicho_no_pregunta_nada(self):                           # el hito sigue directo
        o = self.ask({"titulo": "tarea de EDA2", "curso": "EDA2", "fecha": "viernes"})
        self.assertEqual(o.kind, "confirm")

    def test_fecha_ilegible_se_vuelve_a_preguntar_y_al_final_se_rinde(self):
        self.ask({"titulo": "comprar cuaderno", "curso": "ninguno"})
        for _ in range(2):
            self.assertEqual(self.tools.answer("blablá").kind, "ask")
        o = self.tools.answer("blablá")                                        # a la tercera se rinde
        self.assertEqual(o.kind, "cancelled")
        self.assertFalse(self.tools.has_pending)

    def test_fecha_pasada_se_vuelve_a_preguntar(self):
        self.ask({"titulo": "comprar cuaderno", "curso": "ninguno"})
        o = self.tools.answer("ayer")
        self.assertEqual(o.kind, "ask")
        self.assertTrue(self.tools.has_pending)

    def test_cancelar_en_medio(self):
        self.ask({})
        for frase in ("cancela", "olvídalo", "mejor no"):
            self.ask({})
            o = self.tools.answer(frase)
            self.assertEqual((o.kind, o.consumed), ("cancelled", True), frase)
        self.assertEqual(self.repo.read().revision, 1)

    def test_otra_orden_en_medio_descarta_y_se_procesa(self):
        self.ask({})
        o = self.tools.answer("¿qué tengo hoy?")
        self.assertEqual((o.kind, o.consumed), ("cancelled", False))
        self.assertFalse(self.tools.has_pending)

    def test_una_respuesta_tardia_no_se_toma_como_dato(self):
        self.ask({})
        self.now[0] += 61
        o = self.tools.answer("comprar pan")
        self.assertEqual((o.kind, o.consumed), ("cancelled", False))
        self.assertEqual(self.repo.read().revision, 1)

    def test_no_en_el_titulo_cancela_pero_en_la_fecha_es_sin_fecha(self):
        self.ask({})
        self.assertEqual(self.tools.answer("no").kind, "cancelled")
        self.ask({"titulo": "comprar pan", "curso": "ninguno"})
        self.assertEqual(self.tools.answer("no").kind, "confirm")


class CasoReal(FlowBase):
    """Lo que falló en la prueba de Dario (24-nov): curso con la frase entera, título inventado, correcciones al confirmar."""

    def test_el_curso_llega_con_toda_la_frase(self):
        h = "agrega una tarea para el curso redes"
        o = self.tools.run("crear_tarea", {"curso": h}, heard=h)
        self.assertEqual((o.kind, o.text), ("ask", "¿Qué tarea de Redes de Computadoras agrego?"))
        o = self.tools.answer("estudiar para el parcial")
        self.assertIn("¿Para qué día es?", o.text)
        o = self.tools.answer("el sábado")
        self.assertEqual((o.kind, o.text), ("confirm", "¿Agrego la tarea «Redes de Computadoras: estudiar para el parcial» para sábado 28 de noviembre?"))

    def test_titulo_inventado_por_el_llm_se_pregunta(self):
        h = "agrega una tarea para el curso redes de computadora"
        o = self.tools.run("crear_tarea", {"curso": "redes de computadora", "titulo": "estudiar"}, heard=h)
        self.assertEqual((o.kind, o.text), ("ask", "¿Qué tarea de Redes de Computadoras agrego?"))

    def test_titulo_dicho_si_se_respeta_aunque_cambie_la_forma(self):
        h = "agrega la tarea estudiaré redes para el viernes"
        o = self.tools.run("crear_tarea", {"titulo": "estudiar redes", "curso": "redes", "fecha": "viernes"}, heard=h)
        self.assertEqual(o.kind, "confirm")

    def propose(self):
        h = "agrega una tarea de estudiar redes para el viernes"
        return self.tools.run("crear_tarea", {"titulo": "estudiar redes", "curso": "redes", "fecha": "viernes"}, heard=h)

    def test_no_pero_con_otra_fecha_corrige_y_vuelve_a_proponer(self):
        self.propose()
        o = self.tools.answer("no aún tengo tiempo, para este sábado probablemente")
        self.assertEqual(o.kind, "confirm")
        self.assertIn("sábado 28 de noviembre", o.text)
        self.assertEqual(self.repo.read().revision, 1)
        self.assertEqual(self.tools.answer("sí").kind, "done")
        self.assertEqual(self.repo.read().state["tasks"][-1]["due"], "2026-11-28")

    def test_si_pero_con_otra_fecha_tambien_corrige(self):
        self.propose()
        o = self.tools.answer("sí, estudiaré el sábado")
        self.assertEqual(o.kind, "confirm")
        self.assertIn("sábado 28 de noviembre", o.text)

    def test_un_no_a_secas_sigue_cancelando_y_un_si_a_secas_confirma(self):
        self.propose()
        self.assertEqual(self.tools.answer("no").kind, "cancelled")
        self.propose()
        self.assertEqual(self.tools.answer("sí").kind, "done")

    def test_correccion_a_fecha_pasada_se_pregunta(self):
        self.propose()
        o = self.tools.answer("no, para ayer")
        self.assertNotEqual(o.kind, "done")
        self.assertEqual(self.repo.read().revision, 1)

    def test_otras_escrituras_no_se_corrigen_con_fechas(self):
        self.tools.run("registrar_peso", {"kg": 70.5}, heard="registra que peso 70.5")
        o = self.tools.answer("no, el sábado")
        self.assertEqual(o.kind, "cancelled")


class TemasConTipoOmitido(FlowBase):
    def test_si_dijo_examen_y_el_llm_lo_omite_se_repara(self):
        def add(s):
            for c in s["courses"]:
                for e in c["evals"]:
                    e["topics"] = [f"tema de {e['name']}"]
        self.repo.write(add, by="test")
        o = self.tools.run("temas_examen", {"curso": "EDA2", "cual": "ultimo"}, heard="los temas del último examen de EDA2")
        self.assertIn("Examen final", o.text)


class TemasInteractivo(FlowBase):
    """Casos reales de Dario: EDA2 no tiene exámenes sino ejercicios individuales y un proyecto."""

    def setUp(self):
        super().setUp()

        def shape(s):
            eda = s["courses"][0]
            eda["evals"] = [
                {"id": "e1", "name": "Ejercicio individual 1", "date": "2026-09-20", "weight": 20, "status": "calificada", "grade": 15,
                 "topics": ["Complejidad"], "guide": []},
                {"id": "e2", "name": "Ejercicio individual 2", "date": "2026-10-12", "weight": 20, "status": "calificada", "grade": 14,
                 "topics": [], "guide": []},
                {"id": "p", "name": "Proyecto", "date": "2026-11-22", "weight": 30, "status": "pendiente", "grade": None,
                 "topics": ["Enunciado"], "guide": []},
                {"id": "e3", "name": "Ejercicio individual 3", "date": "2026-11-30", "weight": 30, "status": "pendiente", "grade": None,
                 "topics": ["Grafos bipartitos", "Flujo máximo"], "guide": ["Repasar BFS", "Hacer ejercicios"]}]
        self.repo.write(shape, by="test")

    def test_pide_examen_y_no_hay_pregunta_cual_y_entiende_el_tercero(self):
        o = self.tools.run("temas_examen", {"curso": "EDA2", "evaluacion": "examen", "cual": "ultimo"},
                           heard="qué viene en el último examen de estructura de datos")
        self.assertEqual(o.kind, "ask")
        self.assertIn("no hay exámenes. Tiene Ejercicio individual 1, Ejercicio individual 2, Proyecto y Ejercicio individual 3", o.text)
        o = self.tools.answer("el tercer ejercicio")
        self.assertEqual(o.kind, "answer")
        self.assertIn("Ejercicio individual 3", o.text)
        self.assertIn("Grafos bipartitos; Flujo máximo", o.text)
        self.assertFalse(self.tools.has_pending)

    def test_lo_dicho_manda_sobre_el_examen_del_llm(self):
        o = self.tools.run("temas_examen", {"plan": True, "curso": "EDA2", "evaluacion": "examen"},
                           heard="el tercer ejercicio individual de estructura de datos, dame un plan para estudiar")
        self.assertEqual(o.text, "Plan de estudio para Ejercicio individual 3: Repasar BFS; Hacer ejercicios.")

    def test_evaluacion_2_del_curso_2(self):
        o = self.tools.run("temas_examen", {"curso": "EDA2", "cual": "ultimo"},
                           heard="quiero la evaluación 2 del curso de estructura de datos 2, qué temas vienen")
        self.assertIn("Ejercicio individual 2", o.text)
        self.assertIn("no tengo temas cargados", o.text)

    def test_ejercicios_sin_numero_da_el_proximo(self):
        o = self.tools.run("temas_examen", {"curso": "EDA2", "evaluacion": "ejercicios"}, heard="temas de los ejercicios de EDA2")
        self.assertIn("Ejercicio individual 3", o.text)

    def test_el_plan_se_conserva_tras_preguntar_cual(self):
        self.tools.run("temas_examen", {"curso": "EDA2", "evaluacion": "examen"}, heard="dame el plan del examen de EDA2")
        o = self.tools.answer("el tercero")
        self.assertIn("Plan de estudio para Ejercicio individual 3", o.text)

    # ---- 4.ª ronda (24-nov): frases reales de Dario ----
    def test_evaluaciones_que_se_acercan_aunque_el_llm_elija_nota_necesaria(self):
        o = self.tools.run("nota_necesaria", {"meta": 14, "curso": "EDA2"},
                           heard="quiero saber las evaluaciones de estructura de datos 2 las que se acercan, por favor")
        self.assertEqual(o.tool, "evaluaciones_proximas")
        self.assertIn("se acercan: Ejercicio individual 3, lunes 30 de noviembre, vale 30 por ciento", o.text)
        self.assertNotIn("Proyecto", o.text)                                 # su fecha (22) ya pasó
        o = self.tools.answer("sí")                                           # «¿Quieres los temas de alguna?» -> sí
        self.assertIn("Ejercicio individual 3", o.text)

    def test_seguimiento_sin_repetir_el_curso(self):
        self.tools.run("evaluaciones_proximas", {"curso": "EDA2"}, heard="qué evaluaciones se acercan en EDA2")
        self.tools.offer = None
        o = self.tools.run("consultar_agenda", {"fecha": "hoy"}, heard="el ejercicio individual 3, por favor")
        self.assertEqual(o.tool, "temas_examen")
        self.assertIn("Grafos bipartitos", o.text)

    def test_la_ultima_evaluacion_no_es_el_plan_aunque_el_llm_lo_pida(self):
        o = self.tools.run("temas_examen", {"cual": "ultimo", "curso": "EDA2", "plan": True},
                           heard="quiero saber la última evaluación del curso de estructura de datos 2")
        self.assertTrue(o.text.startswith("En Ejercicio individual 3"))

    def test_la_oferta_no_secuestra_otra_frase(self):
        self.tools.run("evaluaciones_proximas", {"curso": "EDA2"}, heard="qué evaluaciones se acercan")
        self.assertIsNone(self.tools.answer("cuéntame un chiste"))
        self.assertIsNone(self.tools.answer("sí"))                            # la oferta ya caducó

    # ---- 5.ª ronda: simular notas y ver las notas que ya tiene ----
    def test_simula_15_y_15_aunque_el_llm_elija_otra(self):
        o = self.tools.run("nota_necesaria", {"curso": "EDA2"}, heard="yo saco 15 en el proyecto y 15 en el ejercicio individual 3")
        self.assertEqual(o.tool, "simular_notas")
        # ya tiene 15 (20%) y 14 (20%) = 5,8 puntos; + 15*0,30 + 15*0,30 = 14,8
        self.assertEqual(o.text, "Si sacas 15 en Proyecto y 15 en Ejercicio individual 3, terminarías "
                                 "Estructura de Datos y Algoritmos 2 con 14,8: apruebas (mínimo 10,5).")

    def test_simula_parcial_dice_lo_que_falta(self):
        o = self.tools.run("simular_notas", {"curso": "EDA2"}, heard="si saco 8 en el proyecto")
        self.assertIn("en Estructura de Datos y Algoritmos 2, llevarías 8,2 puntos y aún faltaría Ejercicio individual 3 (30%): para aprobar necesitarías 7,67", o.text)

    def test_todo_lo_que_falta_y_numeros_en_palabras(self):
        o = self.tools.run("simular_notas", {"curso": "EDA2"}, heard="y si saco doce en todo lo que falta")
        self.assertIn("terminarías Estructura de Datos y Algoritmos 2 con 13", o.text)

    def test_evaluacion_ya_dada_usa_la_nota_real(self):
        o = self.tools.run("simular_notas", {"curso": "EDA2"}, heard="si saco 20 en el ejercicio individual 1 y 15 en el proyecto")
        self.assertIn("Ejercicio individual 1 ya está calificada con 15, uso esa nota", o.text)

    def test_notas_del_curso_dice_lo_que_ya_dio_y_lo_que_falta(self):
        o = self.tools.run("notas_curso", {"curso": "EDA2"}, heard="cómo voy en EDA2")
        self.assertIn("Ya diste: Ejercicio individual 1 15; Ejercicio individual 2 14.", o.text)
        self.assertIn("Te falta Proyecto (30%) y Ejercicio individual 3 (30%)", o.text)

    # ---- 6.ª ronda: «progra web» tras «¿cómo voy en EDA2?» (el LLM inventó las notas) ----
    def test_seguimiento_con_solo_el_nombre_del_curso(self):
        self.tools.run("notas_curso", {"curso": "EDA2"}, heard="cómo voy en EDA2")
        o = self.tools.answer("redes")
        self.assertEqual((o.tool, o.kind), ("notas_curso", "answer"))
        self.assertIn("En Redes de Computadoras", o.text)
        o = self.tools.answer("¿y en EDA2?")
        self.assertIn("En Estructura de Datos y Algoritmos 2", o.text)

    def test_el_seguimiento_no_secuestra_la_charla(self):
        self.tools.run("notas_curso", {"curso": "EDA2"}, heard="cómo voy en EDA2")
        for t in ("gracias", "cuéntame un chiste", "¿qué tengo hoy?", "agrega una tarea de redes para el viernes"):
            self.assertIsNone(self.tools.answer(t), t)

    def test_sin_lectura_previa_no_hay_seguimiento(self):
        self.assertIsNone(self.tools.answer("redes"))

    # ---- 7.ª ronda: notas de cualquier curso sin LLM, con abreviaturas ----
    def test_como_voy_en_un_curso_sin_llm(self):
        for frase, curso in (("como voy en redes", "Redes de Computadoras"), ("¿qué tal voy en eda 2?", "Estructura de Datos"),
                             ("mis notas de redes", "Redes de Computadoras")):
            o = self.tools.answer(frase)
            self.assertEqual(o.tool, "notas_curso", frase)
            self.assertIn(f"En {curso}", o.text, frase)

    def test_mis_notas_a_secas_son_todas(self):
        self.tools.answer("como voy en redes")
        o = self.tools.answer("mis notas")
        self.assertIn("promedio del ciclo", o.text)

    def test_no_secuestra_otras_frases(self):
        for t in ("cómo me va en la vida", "hola", "si saco 15 en el proyecto cómo voy", "¿qué nota necesito en redes?"):
            self.assertIsNone(self.tools.answer(t), t)

    def test_abreviaturas_para_cursos_nuevos(self):
        def add(s):
            s["courses"].append({"id": "iot", "name": "Internet de las Cosas", "credits": 3, "minGrade": 10.5, "targetGrade": 14,
                                 "evals": [{"id": "l1", "name": "Laboratorio 1", "weight": 50, "status": "calificada", "grade": 13,
                                            "date": "2026-10-01"}]})
        self.repo.write(add, by="test")
        o = self.tools.answer("como voy en iot")
        self.assertIn("En Internet de las Cosas llevas 13 de promedio. Ya diste: Laboratorio 1 13.", o.text)

    # ---- 8.ª ronda: «redes» y luego «si saco 15 en el último examen» (el LLM pasó el curso viejo) ----
    def test_el_curso_viejo_del_llm_se_cambia_por_el_de_la_charla(self):
        self.tools.answer("como voy en EDA2")
        self.tools.answer("redes")
        o = self.tools.run("simular_notas", {"curso": "Estructura de Datos"}, heard="si saco 15 en la última?")
        self.assertIn("en Redes de Computadoras, llevarías", o.text)

    def test_nota_despues_del_nombre(self):
        o = self.tools.run("simular_notas", {}, heard="dije si saco en el último de redes 15 qué pasa")
        self.assertIn("Redes de Computadoras", o.text)
        self.assertIn("Si sacas 15 en", o.text)

    def test_corregir_el_curso_en_la_repregunta(self):
        self.tools.answer("como voy en EDA2")
        o = self.tools.run("simular_notas", {"curso": "EDA2"}, heard="si saco 15 en el laboratorio 9")
        self.assertEqual(o.kind, "ask")
        o = self.tools.answer("de redes, en la eva 3")
        self.assertIn("Redes de Computadoras", o.text)

    # ---- 9.ª ronda: respuestas cortas y «¿sigo?» ----
    def test_temas_largos_se_resumen_y_se_ofrece_el_resto(self):
        def many(s):
            s["courses"][0]["evals"][3]["topics"] = [f"Tema {i}: detalle largo que no hace falta decir" for i in range(1, 8)]
        self.repo.write(many, by="test")
        o = self.tools.run("temas_examen", {"curso": "EDA2"}, heard="temas del ejercicio individual 3")
        self.assertIn("entran: Tema 1; Tema 2; Tema 3; Tema 4. Y 3 más, ¿te los digo?", o.text)
        o = self.tools.answer("sí")
        self.assertEqual(o.text, "Los demás temas: Tema 5: detalle largo que no hace falta decir; Tema 6: detalle largo que no "
                                 "hace falta decir; Tema 7: detalle largo que no hace falta decir.")
        self.assertIsNone(self.tools.answer("sí"))

    def test_brief(self):
        from tools.unigym.readers import brief
        self.assertEqual(brief("Ejecuta paso a paso, a mano, Prim, Kruskal, Dijkstra y Ford-Fulkerson sobre un grafo de 6 a 8 nodos."),
                         "Ejecuta paso a paso, a mano, Prim, Kruskal, Dijkstra")
        self.assertEqual(brief("Es el examen que más pesa del curso (35%) y sin IA: practica a mano."), "Es el examen que más pesa del curso")
        self.assertEqual(brief("Subnetting VLSM"), "Subnetting VLSM")

    # ---- 10.ª ronda: marcar tareas sin LLM, contexto que caduca ----
    def add_task(self, title):
        self.repo.write(lambda s: s["tasks"].append({"id": "t_" + title[:3], "title": title, "category": "Personal",
                                                     "priority": "media", "due": "", "done": False}), by="test")

    def test_marcar_como_listo_sin_llm(self):
        self.add_task("estudiar redes")
        o = self.tools.answer("marcar estudiar redes como listo")
        self.assertEqual((o.tool, o.kind), ("completar_tarea", "confirm"))
        self.assertEqual(self.tools.answer("sí").kind, "done")
        self.assertTrue(next(t for t in self.repo.read().state["tasks"] if t["title"] == "estudiar redes")["done"])

    def test_marca_como_hecha_la_tarea_x(self):
        self.add_task("estudiar redes")
        o = self.tools.answer("marca como hecha la tarea estudiar redes")
        self.assertEqual(o.kind, "confirm")

    def test_marcar_tarea_sin_decir_cual_pregunta_y_lista(self):
        self.add_task("estudiar redes")
        o = self.tools.answer("marcar tarea como lista")
        self.assertTrue(o.text.startswith("¿Cuál marco como hecha? Tienes"), o.text)
        self.assertIn("estudiar redes", o.text)
        o = self.tools.answer("estudiar redes")
        self.assertEqual(o.kind, "confirm")

    def test_no_secuestra(self):
        for t in ("ya terminé de comer", "termina la historia"):
            self.assertIsNone(self.tools.answer(t), t)

    def test_que_tareas_tengo_no_usa_el_curso_viejo(self):
        self.tools.answer("como voy en redes")
        o = self.tools.run("tareas_pendientes", {"curso": "Redes de Computadoras"}, heard="qué tareas tengo pendientes")
        self.assertTrue(o.text.startswith("Tienes"), o.text)
        self.assertNotIn("de Redes", o.text.split(":")[0])

    def test_el_contexto_se_olvida(self):
        self.tools.answer("como voy en redes")
        self.tools.forget_context()
        self.assertIsNone(self.tools.answer("EDA2"))                          # ya no es un seguimiento

    # ---- 11.ª ronda: «ya di el último, saqué 20» y récord académico ----
    def test_saque_nota_propone_registrarla(self):
        self.tools.answer("como voy en EDA2")
        o = self.tools.answer("ya di el último ejercicio, saqué 18")
        self.assertEqual((o.tool, o.kind), ("registrar_nota", "confirm"))
        self.assertEqual(o.text, "¿Anoto 18 en Ejercicio individual 3 de Estructura de Datos y Algoritmos 2?")
        self.assertEqual(self.tools.answer("sí").kind, "done")
        ev = next(e for e in self.repo.read().state["courses"][0]["evals"] if e["id"] == "e3")
        self.assertEqual((ev["grade"], ev["status"]), (18, "calificada"))

    def test_si_saco_no_es_registrar(self):
        self.tools.answer("como voy en EDA2")
        self.assertIsNone(self.tools.answer("si saco 18 en el último ejercicio"))

    def test_record_academico(self):
        def rec(s):
            s["record"] = {"career": "Ing. de Sistemas", "ranking": "Décimo superior", "cycles": [
                {"id": "a", "cycle": "2025-1", "level": 4, "ppa": 14.0, "ppp": 14.0, "credits": 20, "program": "Carrera"},
                {"id": "b", "cycle": "2025-2", "level": 5, "ppa": 14.5, "ppp": 15.0, "credits": 20, "program": "Carrera"}]}
        self.repo.write(rec, by="test")
        o = self.tools.answer("¿cómo va mi récord académico?")
        self.assertEqual(o.tool, "record_academico")
        self.assertTrue(o.text.startswith("Tu promedio acumulado es 14,5. Estás en el décimo superior. "
                                          "Subió en todos tus ciclos: un 3,6 por ciento desde el primero."), o.text)
        self.assertIn("Si terminas este ciclo como vas", o.text)

    def test_record_por_nivel(self):
        def rec(s):
            s["record"] = {"ranking": "Décimo superior", "cycles": [
                {"id": "a", "cycle": "2024-1", "level": 1, "ppa": 13.5, "ppp": 13.5, "credits": 20, "program": "Generales"},
                {"id": "b", "cycle": "2024-2", "level": 2, "ppa": 14.2, "ppp": 15.0, "credits": 22, "program": "Generales"}]}
        self.repo.write(rec, by="test")
        o = self.tools.answer("hablemos de mi récord académico")
        self.assertTrue(o.text.startswith("Tu promedio acumulado es 14,2. Estás en el décimo superior. "
                                          "Subió en todos tus ciclos: un 5,2 por ciento desde el primero."), o.text)
        o = self.tools.answer("aja pero en ciclo nivel 1 como iba")
        self.assertEqual(o.text, "En el nivel 1 (2024-1) tu promedio del ciclo fue 13,5 y tu acumulado quedó en 13,5.")
        o = self.tools.answer("¿y el segundo ciclo?")
        self.assertIn("Frente al nivel 1, tu promedio del ciclo subió un 11,1 por ciento", o.text)
        self.assertIn("Fue un ciclo que te subió el acumulado.", o.text)
        o = self.tools.answer("¿qué puedes decir comparando nivel 1 y 2?")
        self.assertEqual(o.text, "Del nivel 1 al nivel 2, tu promedio del ciclo subió un 11,1 por ciento, de 13,5 a 15, y tu promedio "
                                 "acumulado subió un 5,2 por ciento. Y eso llevando más créditos (22 contra 20): rendiste mejor con "
                                 "más carga. En resumen: vas mejorando.")
        o = self.tools.answer("dime todos los ciclos")
        self.assertEqual(o.text, "Tu promedio de cada ciclo fue nivel 1: 13,5; nivel 2: 15. Tu acumulado subió un 5,2 por ciento "
                                 "desde el primero: de 13,5 a 14,2.")
        self.assertIn("No tengo el nivel 5", self.tools.answer("como me fue en el nivel 5").text)

    def test_otra_pregunta_en_medio_descarta(self):
        self.tools.run("temas_examen", {"curso": "EDA2", "evaluacion": "examen"}, heard="temas del examen de EDA2")
        o = self.tools.answer("¿qué tengo hoy?")
        self.assertFalse(o.consumed)


class NoPierdeTemas(FlowBase):
    def test_una_escritura_conserva_temas_y_plan_de_las_evaluaciones(self):
        def add(s):
            ev = s["courses"][0]["evals"][0]
            ev.update(topics=["Árboles AVL", "Grafos"], guide=["Repasar AVL"], mode="presencial", ia="media")
        self.repo.write(add, by="test")
        self.tools.run("crear_tarea", {"titulo": "comprar pan", "curso": "ninguno", "fecha": "sin fecha"}, heard="comprar pan")
        self.tools.answer("sí")
        ev = self.repo.read().state["courses"][0]["evals"][0]
        self.assertEqual((ev["topics"], ev["guide"], ev["mode"], ev["ia"]), (["Árboles AVL", "Grafos"], ["Repasar AVL"], "presencial", "media"))


class Cancelaciones(FlowBase):
    def propose(self):
        return self.tools.run("registrar_peso", {"kg": 70.5}, heard="registra que peso 70.5")

    def test_no_cancela_sin_escribir(self):
        self.propose()
        out = self.tools.answer("no")
        self.assertEqual((out.kind, out.consumed), ("cancelled", True))
        self.assertEqual(self.repo.read().revision, 1)
        self.assertFalse(self.backups.exists())
        self.assertIsNone(self.tools.answer("sí"))                          # ya no hay nada pendiente

    def test_otra_frase_cancela_y_se_procesa_normal(self):
        self.propose()
        out = self.tools.answer("¿qué tengo mañana?")
        self.assertEqual((out.kind, out.consumed), ("cancelled", False))    # el bucle debe tratarla como una orden nueva
        self.assertEqual(self.repo.read().revision, 1)

    def test_un_si_tardio_no_escribe(self):
        self.propose()
        self.now[0] += 61
        out = self.tools.answer("sí")
        self.assertEqual(out.kind, "cancelled")
        self.assertEqual(self.repo.read().revision, 1)
        self.assertIn("expired", [s for _, s in self.statuses()])

    def test_una_orden_nueva_reemplaza_a_la_pendiente(self):
        self.propose()
        self.tools.run("registrar_diario", {"campo": "agua", "valor": 1})
        self.tools.answer("sí")
        s = self.repo.read().state
        self.assertEqual(s["dailyLogs"]["2026-11-24"]["water"], 4)
        self.assertEqual(s["gym"]["bodyWeight"], [])                        # el peso nunca se guardó

    def test_pregunta_aclaratoria_no_deja_nada_pendiente(self):
        out = self.tools.run("registrar_peso", {"kg": 7})
        self.assertEqual(out.kind, "ask")
        self.assertFalse(self.tools.has_pending)
        self.assertIsNone(self.tools.answer("sí"))
        self.assertEqual(self.statuses(), [("registrar_peso", "ask")])

    def test_lo_que_ya_estaba_hecho_no_pide_confirmacion(self):
        out = self.tools.run("marcar_habito", {"habito": "agua"})
        self.assertEqual(out.kind, "answer")
        self.assertFalse(self.tools.has_pending)


class Robustez(FlowBase):
    def test_herramienta_desconocida_o_argumentos_raros(self):
        self.assertEqual(self.tools.run("borrar_todo", {}).kind, "error")
        self.assertEqual(self.tools.run("consultar_agenda", "no es dict").kind, "answer")
        self.assertEqual(self.tools.run("crear_tarea", None).kind, "ask")
        self.assertEqual(self.repo.read().revision, 1)
        self.assertIn(("borrar_todo", "rejected"), self.statuses())

    def test_si_la_tarea_desaparece_antes_del_si_no_se_guarda(self):
        self.tools.run("completar_tarea", {"titulo": "informe de redes"})
        self.repo.write(lambda s: s["tasks"].clear(), by="web")            # la web la borró mientras tanto
        out = self.tools.answer("sí")
        self.assertEqual((out.kind, out.ok), ("error", False))
        self.assertIn("ya no existe", out.text)
        self.assertEqual(self.repo.read().revision, 2)                      # solo la escritura de la web
        self.assertIn(("completar_tarea", "error"), self.statuses())

    def test_cambios_de_la_web_entre_la_pregunta_y_el_si_se_conservan(self):
        self.tools.run("registrar_diario", {"campo": "agua", "valor": 1})
        self.repo.write(lambda s: s["tasks"].append({"id": "x", "title": "Nueva de la web", "due": "", "done": False}), by="web")
        self.assertEqual(self.tools.answer("sí").kind, "done")
        s = self.repo.read().state
        self.assertTrue(any(t["id"] == "x" for t in s["tasks"]))
        self.assertEqual(s["dailyLogs"]["2026-11-24"]["water"], 4)

    def test_sin_archivo_de_datos_responde_sin_romper(self):
        tools = UniGymTools(JsonFileRepository(Path(self._tmp.name) / "nada.json"), self.audit, today=lambda: TUE)
        out = tools.run("consultar_agenda", {})
        self.assertEqual((out.kind, out.ok), ("error", False))
        self.assertIn("No pude leer", out.text)

    def test_funciona_sin_registro(self):
        tools = UniGymTools(self.repo, None, today=lambda: TUE)
        tools.run("registrar_peso", {"kg": 70})
        self.assertEqual(tools.answer("sí").kind, "done")

    def test_cada_escritura_deja_su_copia_y_el_registro_la_cuenta(self):
        for kg in (70, 70.5, 71):
            self.tools.run("registrar_peso", {"kg": kg}, heard=f"peso {kg}")
            self.tools.answer("sí")
        self.assertEqual(len(list(self.backups.glob("unigym-*.json"))), 3)
        self.assertEqual(self.repo.read().revision, 4)
        self.assertEqual([r["status"] for r in self.audit.tail(50)].count("confirmed"), 3)

    def test_registro_trunca_textos_largos(self):
        self.audit.record("x", "read", heard="a" * 2000)
        self.assertEqual(len(self.audit.tail(1)[0]["heard"]), 500)


if __name__ == "__main__":
    unittest.main()

class ListaDeCursos(FlowBase):
    """Caso real (24-nov): 7 cursos registrados, el LLM dijo 4 e inventó uno; y «aéa»/«AAA» no encontraba AEA."""

    def setUp(self):
        super().setUp()

        def add(s):
            s["courses"].append({"id": "aea", "name": "AEA", "credits": 2, "minGrade": 10.5, "targetGrade": 14,
                                 "evals": [{"id": "e1", "name": "Examen 1", "weight": 100, "status": "pendiente",
                                            "date": "2026-12-05", "topics": ["Tema uno", "Tema dos"]}]})
        self.repo.write(add, by="test")

    def test_lista_todos_los_cursos_reales(self):
        names = [c["name"] for c in self.repo.read().state["courses"]]
        for phrase in ("¿tienes la lista de cursos que tienes registrado ahorita?", "quiero ver todos los cursos de los que estoy inscrito",
                       "qué cursos tengo", "dime mis cursos"):
            o = self.tools.answer(phrase)
            self.assertIsNotNone(o, phrase)
            for n in names:
                self.assertIn(n, o.text, phrase)
            self.assertIn(f"{len(names)} cursos", o.text)

    def test_no_secuestra_otras_frases(self):
        for t in ("mis notas de todos los cursos", "agrega un curso nuevo", "qué tareas tengo en el curso de redes"):
            o = self.tools.answer(t)
            self.assertTrue(o is None or "cursos:" not in o.text, t)

    def test_examen_del_curso_oido_como_aea_o_aaa(self):
        for phrase in ("puedes ver el examen del curso aéa", "el examen del curso llamado AAA"):
            o = self.tools.answer(phrase)
            self.assertIsNotNone(o, phrase)
            self.assertEqual(o.tool, "temas_examen", phrase)
            self.assertIn("tema uno", o.text.lower(), phrase)

    def test_tras_una_pregunta_pendiente_la_lista_sigue_en_python(self):
        self.tools.answer("el examen del curso llamado AAA")
        self.gate.set_slot("temas_examen", {"curso": "AEA"}, "evaluacion", "temas", 1)   # JARVIS preguntó «¿de cuál?»
        o = self.tools.answer("¿Tienes la lista de cursos que tienes registrado ahorita?")
        self.assertEqual(o.tool, "lista_cursos")
        self.assertIn("AEA", o.text)
        self.gate.set_slot("crear_tarea", {"titulo": "algo"}, "fecha", "agrega algo", 1)   # una tarea a medias: se avisa
        o = self.tools.answer("qué cursos tengo")
        self.assertEqual(o.tool, "lista_cursos")
        self.assertTrue(o.text.startswith("Descarté"))


class HabitoDirecto(FlowBase):
    """Bench 24-nov: «ya hice el hábito de leer» el LLM lo felicitaba sin llamar a la herramienta."""

    def test_frases_de_habito_van_directo(self):
        habits = [h["name"] for h in self.repo.read().state["habits"]]
        for phrase in ("Ya hice el hábito de %s", "Ya cumplí con el hábito de %s", "Marca mi hábito de %s", "marca el hábito de %s como hecho"):
            for h in habits[:2]:
                o = self.tools.answer(phrase % h.lower())
                self.assertIsNotNone(o, phrase)
                self.assertEqual(o.tool, "marcar_habito", phrase)

    def test_no_secuestra_preguntas_ni_otras_frases(self):
        for t in ("¿Qué hábitos me faltan hoy?", "¿cómo van mis hábitos?", "ya terminé el informe de redes", "qué es un hábito"):
            o = self.tools.answer(t)
            self.assertTrue(o is None or o.tool != "marcar_habito", t)


class Ronda17(FlowBase):
    """Prueba real (24-nov, 6.6): tarea «Estructura de Datos II: ford» no se encontraba con «EDA2 ford»; la frase entera
    se tomaba como título tras un «¿cuál?»; «comparar nivel 4 y 5» iba al LLM («no tengo acceso a datos»)."""

    def setUp(self):
        super().setUp()

        def add(s):
            s["tasks"].append({"id": "t_ford", "title": "Estructura de Datos II: ford", "category": "Universidad", "priority": "alta",
                               "due": "2026-11-27", "done": False, "createdAt": "2026-11-24"})
        self.repo.write(add, by="test")

    def test_tarea_con_siglas_del_curso(self):
        o = self.tools.answer("marca la tarea EDA2 ford como hecha")
        self.assertEqual(o.kind, "confirm")
        self.assertIn("ford", o.text)

    def test_orden_entera_tras_un_cual(self):
        self.tools.run("completar_tarea", {"titulo": "zzz inexistente"}, heard="marca zzz como hecha")   # pregunta «¿cuál?»
        o = self.tools.answer("marca la tarea ford como hecha")
        self.assertEqual(o.kind, "confirm")
        self.assertIn("ford", o.text)

    def test_comparar_niveles_sin_palabra_de_record(self):
        def rec(st):
            st["record"] = {"career": "x", "ranking": "x", "cycles": [
                {"cycle": f"202{i // 2 + 1}-{i % 2 + 1}", "level": i + 1, "ppp": 15 + i * 0.3, "ppa": 15 + i * 0.2, "credits": 20}
                for i in range(5)]}
        self.repo.write(rec, by="test")
        for phrase in ("comparar nivel 4 y 5", "nivel de record academico 4 y 5"):
            o = self.tools.answer(phrase)
            self.assertIsNotNone(o, phrase)
            self.assertEqual(o.tool, "record_academico", phrase)
            self.assertTrue(o.text.startswith("Del nivel 4 al nivel 5"), (phrase, o.text))

