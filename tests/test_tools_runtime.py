"""Pruebas de la integración herramientas + bucle de voz (Fase 6.5) con LLM, selector y micrófono falsos.
Datos inventados; no tocan data/unigym.json.
"""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.voice_loop import VoiceLoop
from llm import LLMError, ToolCall
from llm.ollama_backend import _parse_tool_call
from stt.base import Transcript
from tests.test_unigym_readers import state as base_state
from tools.runtime import ToolRuntime
from tools.unigym import SPECS, spec_by_name
from tools.unigym.audit import AuditLog
from tools.unigym.confirm import ConfirmationGate
from tools.unigym.executor import UniGymTools
from tools.unigym.repo import JsonFileRepository

TUE = dt.date(2026, 11, 24)
AUDIO = np.ones(1600, dtype=np.float32)


class FakeSelector:
    def __init__(self, names=("consultar_agenda", "crear_tarea", "registrar_peso", "tareas_pendientes"), fail=False):
        self.names, self.fail = names, fail

    def select(self, text):
        if self.fail:
            raise RuntimeError("embeddings caídos")
        return [spec_by_name(n) for n in self.names]


class FakeLLM:
    """Responde con el guion que se le dé: texto, ToolCall o un error."""
    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def stream_events(self, messages, tools=None, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        step = self.script.pop(0) if self.script else ["…"]
        if isinstance(step, Exception):
            raise step
        yield from step

    def stream(self, messages, **kw):
        yield from self.stream_events(messages, None, **kw)


class FakeSpeaker:
    def __init__(self):
        self.said, self.first_audio_delay = [], 0.2

    def feed(self, t): self.said.append(t)
    def finish(self): pass
    def wait(self): pass
    def stop(self): pass


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        src = d / "r.json"
        src.write_text(json.dumps({"app": "uni-gym-tracker", "version": 1, "data": base_state()}), encoding="utf-8")
        self.repo = JsonFileRepository.from_backup(src, d / "unigym.json", keep=10)
        self.audit = AuditLog(":memory:")
        self.unigym = UniGymTools(self.repo, self.audit, today=lambda: TUE, gate=ConfirmationGate(ttl_s=60))
        self.rt = ToolRuntime(self.unigym, FakeSelector(), today=lambda: TUE)
        self.spk = FakeSpeaker()
        self.heard = []

    def tearDown(self):
        self._tmp.cleanup()

    def say(self, text, llm, tools="default", memory=None):
        loop = getattr(self, "loop", None)
        if loop is None:
            self.loop = loop = VoiceLoop(
                listen=lambda: AUDIO, transcribe=lambda a: Transcript(self._text, "es", 1.0, 0.5), llm=llm,
                speaker=self.spk, history=[{"role": "system", "content": "s"}], tail_ms=0, sleep=lambda s: None,
                tools=self.rt if tools == "default" else tools, memory=memory)
        loop.llm = llm
        self._text = text
        return loop.run_turn()


class Hito(Base):
    def test_que_tengo_hoy(self):
        llm = FakeLLM([ToolCall("consultar_agenda", {})])
        t = self.say("¿Qué tengo hoy?", llm)
        self.assertEqual((t.kind, t.tool), ("tool", "consultar_agenda"))
        self.assertIn("2 bloques hoy", t.reply)
        self.assertEqual(self.spk.said[-1], t.reply)
        self.assertEqual(self.repo.read().revision, 1)
        self.assertEqual(len(llm.calls[0]["tools"]), 4)                    # ofrecidas las 4 elegidas
        self.assertIn("Herramientas:", llm.calls[0]["messages"][0]["content"])   # guía solo cuando hay herramientas
        self.assertIn("martes 2026-11-24", llm.calls[0]["messages"][0]["content"])
        h = self.loop.history                                                # lectura: llamada + resultado en formato nativo
        self.assertEqual([m["role"] for m in h], ["system", "user", "assistant", "tool"])
        self.assertEqual(h[2]["tool_calls"][0]["function"]["name"], "consultar_agenda")
        self.assertEqual((h[3]["tool_name"], h[3]["content"]), ("consultar_agenda", t.reply))

    def test_agrega_una_tarea_de_eda2_para_el_viernes_y_el_si(self):
        t = self.say("agrega una tarea de EDA2 para el viernes",
                     FakeLLM([ToolCall("crear_tarea", {"titulo": "tarea de EDA2", "curso": "EDA2", "fecha": "viernes"})]))
        self.assertEqual(t.reply, "¿Agrego la tarea «Tarea de EDA2» para viernes 27 de noviembre?")
        self.assertEqual(self.repo.read().revision, 1)                       # aún no se guardó
        llm2 = FakeLLM()
        t2 = self.say("sí", llm2)                                            # el sí NO pasa por el LLM
        self.assertEqual((t2.kind, t2.reply), ("tool", "Listo, tarea agregada."))
        self.assertEqual(llm2.calls, [])
        snap = self.repo.read()
        self.assertEqual((snap.revision, snap.by, snap.state["tasks"][-1]["due"]), (2, "jarvis", "2026-11-27"))
        self.assertEqual(self.spk.said[-1], "Listo, tarea agregada.")

    def test_no_cancela(self):
        self.say("registra que peso 70.5", FakeLLM([ToolCall("registrar_peso", {"kg": 70.5})]))
        t = self.say("no", FakeLLM())
        self.assertEqual(t.reply, "Cancelado, no guardé nada.")
        self.assertEqual(self.repo.read().revision, 1)


class DialogoPorVoz(Base):
    def test_crear_tarea_pregunta_lo_que_falta_sin_pasar_por_el_llm(self):
        t = self.say("crear tarea", FakeLLM([ToolCall("crear_tarea", {})]))
        self.assertEqual(t.reply, "¿Qué tarea agrego?")
        llm = FakeLLM()
        for frase, esperado in (("estudiar para el parcial", "¿Es de algún curso?"), ("ninguno", "¿Para qué día es?"),
                                ("el viernes", "¿Agrego la tarea «estudiar para el parcial» para viernes 27 de noviembre?")):
            t = self.say(frase, llm)
            self.assertIn(esperado, t.reply)
        self.assertEqual(llm.calls, [])                                       # todo el diálogo fue en Python
        self.assertEqual(self.repo.read().revision, 1)
        self.assertEqual(self.say("sí", llm).reply, "Listo, tarea agregada.")
        self.assertEqual(self.repo.read().revision, 2)


class HistorialSinImitaciones(Base):
    """Bug real (24-nov): el LLM copió «¿Marco como hecha...?» como texto en vez de llamar a la herramienta."""

    def assistant_texts(self):
        return [m["content"] for m in self.loop.history if m["role"] == "assistant" and m["content"]]

    def test_preguntas_y_confirmaciones_no_entran_al_historial(self):
        self.say("agrega una tarea de EDA2 para el viernes",
                 FakeLLM([ToolCall("crear_tarea", {"titulo": "tarea de EDA2", "curso": "EDA2", "fecha": "viernes"})]))
        self.say("sí", FakeLLM())
        self.assertEqual([m["role"] for m in self.loop.history], ["system"])    # ni «¿Agrego...?» ni «Listo, tarea agregada.»

    def test_dialogo_guiado_tampoco(self):
        self.say("crear tarea", FakeLLM([ToolCall("crear_tarea", {})]))
        self.say("estudiar para el parcial", FakeLLM())
        self.assertEqual(self.assistant_texts(), [])

    def test_charla_normal_si_se_guarda(self):
        self.say("hola jarvis", FakeLLM(["Hola, ", "Dario."]))
        self.assertEqual(self.assistant_texts(), ["Hola, Dario."])

    def test_el_siguiente_turno_ve_un_historial_limpio(self):
        self.say("marca como hecha la tarea de EDA2", FakeLLM([ToolCall("completar_tarea", {"titulo": "EDA2"})]))
        llm = FakeLLM([ToolCall("consultar_agenda", {})])
        self.say("¿qué tengo hoy?", llm)
        vistos = [m for m in llm.calls[0]["messages"] if m["role"] == "assistant"]
        self.assertEqual(vistos, [])                                          # nada que el LLM pueda copiar como texto


class AfirmacionFalsa(Base):
    def test_si_el_llm_dice_que_guardo_sin_llamar_a_la_herramienta_se_avisa(self):
        t = self.say("marca la tarea de EDA2", FakeLLM(["Entendido. La tarea «EDA2» se marca como listo."]))
        self.assertEqual(t.kind, "reply")
        self.assertIn("no guardé nada", t.reply)
        self.assertEqual(self.repo.read().revision, 1)

    def test_charla_normal_no_se_toca(self):
        for frase in ("Hola, Dario.", "Claro, te explico cómo funciona el DNS.", "Listo, entendido."):
            self.loop = None
            t = self.say("hola", FakeLLM([frase]))
            self.assertNotIn("no guardé", t.reply, frase)

    def test_sin_herramientas_ofrecidas_no_aplica(self):
        self.rt.selector = FakeSelector(names=())
        t = self.say("cuéntame algo", FakeLLM(["Se guarda en la nube."]))
        self.assertNotIn("no guardé", t.reply)


class ConfirmacionFalsa(Base):
    def test_pregunta_de_confirmacion_inventada_se_avisa_y_no_entra_al_historial(self):
        t = self.say("marcar estudiar redes", FakeLLM(["¿Marcar como listo la tarea «estudiar redes»?"]))
        self.assertIn("no guardé nada", t.reply)
        self.assertEqual([m["role"] for m in self.loop.history], ["system"])


class DatosInventados(Base):
    def test_notas_sin_herramienta_se_avisan(self):
        t = self.say("progra web", FakeLLM(["En Programación Web llevas 10,56 de promedio. Ya diste: Parcial 1 12."]))
        self.assertIn("no lo saqué de tus datos", t.reply)

    def test_si_los_numeros_los_dijo_dario_no_se_avisa(self):
        t = self.say("ya di el examen y saqué 20", FakeLLM(["Bien, un 20 en el examen. Tu nota es excelente."]))
        self.assertNotIn("no lo saqué", t.reply)

    def test_numeros_de_un_resultado_anterior_no_son_inventados(self):
        self.rt.selector = FakeSelector(names=("notas_curso",))
        self.say("como voy en EDA2", FakeLLM([ToolCall("notas_curso", {"curso": "EDA2"})]))
        prev = self.loop.history[-1]["content"]
        import re as _re
        n = _re.findall(r"\d+(?:,\d+)?", prev)[0]
        t = self.say("y entonces?", FakeLLM([f"Tu promedio en EDA2 es {n}, vas bien en la nota del curso."]))
        self.assertNotIn("no lo saqué", t.reply)

    def test_charla_con_numeros_no_se_toca(self):
        for frase in ("Son las 3 de la tarde.", "El puerto 80 es HTTP.", "Hola, Dario."):
            self.loop = None
            self.assertNotIn("no lo saqué", self.say("hola", FakeLLM([frase])).reply, frase)


class RespuestaVacia(Base):
    def test_si_el_llm_no_dice_nada_pide_que_repita(self):
        t = self.say("quiero la evaluación de estructura de datos TOF", FakeLLM([]))
        self.assertEqual(t.reply, "No te entendí bien. ¿Me lo repites?")
        self.assertEqual(self.spk.said[-1], "No te entendí bien. ¿Me lo repites?")


class ContextoDeLecturas(Base):
    def test_una_repregunta_de_lectura_queda_en_formato_nativo(self):
        self.rt.selector = FakeSelector(names=("temas_examen",))
        self.say("temas del proyecto de redes", FakeLLM([ToolCall("temas_examen", {"curso": "redes", "evaluacion": "proyecto"})]))
        roles = [m["role"] for m in self.loop.history]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])     # el LLM ve de qué se hablaba
        self.assertEqual(self.loop.history[2]["content"], "")                # nada de texto que imitar


class OtraOrdenEnLugarDeSi(Base):
    def test_descarta_y_atiende_la_orden_nueva(self):
        self.say("registra que peso 70.5", FakeLLM([ToolCall("registrar_peso", {"kg": 70.5})]))
        llm = FakeLLM([ToolCall("consultar_agenda", {"fecha": "mañana"})])
        t = self.say("¿qué tengo mañana?", llm)
        self.assertEqual(t.kind, "tool")
        self.assertTrue(t.reply.startswith("Descarté lo que estaba pendiente."), t.reply)
        self.assertEqual(self.repo.read().revision, 1)
        self.assertTrue(len(llm.calls) == 1)                                 # sí pasó por el LLM
        self.assertFalse(self.rt.has_pending)

    def test_charla_sin_herramienta_tras_pendiente(self):
        self.say("registra que peso 70.5", FakeLLM([ToolCall("registrar_peso", {"kg": 70.5})]))
        t = self.say("cuéntame un chiste", FakeLLM(["Había una vez ", "un robot."]))
        self.assertEqual(t.kind, "reply")
        self.assertEqual(t.reply, "Descarté lo que estaba pendiente. Había una vez un robot.")


class RespuestasCortas(unittest.TestCase):
    def test_si_y_no_no_son_ruido(self):
        from core.voice_loop import is_noise
        for t in ("sí", "Sí.", "no", "¡No!"):
            self.assertFalse(is_noise(t), t)
        for t in ("a", "eh", "", "¡Suscríbete!"):
            self.assertTrue(is_noise(t), t)


class Charla(Base):
    def test_charla_normal_no_cambia(self):
        t = self.say("hola jarvis", FakeLLM(["Hola, ", "Dario."]))
        self.assertEqual((t.kind, t.reply), ("reply", "Hola, Dario."))

    def test_sin_runtime_no_hay_herramientas(self):
        llm = FakeLLM(["Hola."])
        self.say("hola", llm, tools=None)
        self.assertIsNone(llm.calls[0]["tools"] if llm.calls else None)

    def test_selector_caido_sigue_sin_herramientas(self):
        self.rt.selector = FakeSelector(fail=True)
        self.rt.warn = lambda m: None
        llm = FakeLLM(["Hola."])
        t = self.say("hola", llm)
        self.assertEqual((t.kind, t.reply), ("reply", "Hola."))
        self.assertIsNone(llm.calls[0]["tools"])
        self.assertNotIn("Herramientas:", llm.calls[0]["messages"][0]["content"])

    def test_llm_caido_con_herramientas(self):
        t = self.say("¿qué tengo hoy?", FakeLLM(LLMError("servidor caído")))
        self.assertEqual(t.kind, "error")

    def test_herramienta_con_error_no_tumba(self):
        t = self.say("haz algo", FakeLLM([ToolCall("no_existe", {})]))
        self.assertEqual((t.kind, t.reply), ("tool", "No tengo esa herramienta."))

    def test_numero_inventado_se_pregunta(self):
        t = self.say("anota el informe de redes para el lunes", FakeLLM([ToolCall(
            "registrar_nota", {"curso": "redes", "evaluacion": "informe", "nota": 18})]))
        self.assertEqual(t.kind, "tool")
        self.assertFalse(self.rt.has_pending)
        self.assertEqual(self.repo.read().revision, 1)

    def test_texto_antes_de_la_herramienta_no_se_guarda_como_respuesta(self):
        t = self.say("¿qué tengo hoy?", FakeLLM(["Voy a mirar. ", ToolCall("consultar_agenda", {})]))
        self.assertIn("2 bloques hoy", t.reply)


class Auditoria(Base):
    def test_cada_paso_queda_registrado(self):
        self.say("agrega una tarea de EDA2 para el viernes",
                 FakeLLM([ToolCall("crear_tarea", {"titulo": "tarea de EDA2", "curso": "EDA2", "fecha": "viernes"})]))
        self.say("sí", FakeLLM())
        rows = [(r["tool"], r["status"]) for r in self.audit.tail(10)]
        self.assertEqual(rows, [("crear_tarea", "proposed"), ("crear_tarea", "confirmed")])
        self.assertEqual(self.audit.tail(2)[0]["heard"], "agrega una tarea de EDA2 para el viernes")


class OllamaParse(unittest.TestCase):
    def test_argumentos_dict_o_texto(self):
        a = _parse_tool_call({"function": {"name": "x", "arguments": {"fecha": "hoy"}}})
        b = _parse_tool_call({"function": {"name": "x", "arguments": '{"fecha": "hoy"}'}})
        self.assertEqual((a.name, a.args), ("x", {"fecha": "hoy"}))
        self.assertEqual(b.args, {"fecha": "hoy"})

    def test_basura(self):
        self.assertIsNone(_parse_tool_call({}))
        self.assertIsNone(_parse_tool_call({"function": {"name": ""}}))
        self.assertEqual(_parse_tool_call({"function": {"name": "x", "arguments": "no json"}}).args, {})


if __name__ == "__main__":
    unittest.main()
