"""Pruebas del bucle de voz (Fase 4) sin micrófono, GPU ni modelos.
Ejecutar: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from core.voice_loop import (VoiceLoop, is_echo, is_noise, parse_command, trim_history)
from llm import LLMError
from stt.base import Transcript

AUDIO = np.ones(1600, dtype=np.float32)
SYS = {"role": "system", "content": "s"}


class FakeLLM:
    def __init__(self, chunks=("Hola, ", "Dario."), fail=False):
        self.chunks, self.fail, self.seen = chunks, fail, []

    def stream(self, messages, **kw):
        self.seen.append(list(messages))
        if self.fail:
            raise LLMError("servidor caído")
        yield from self.chunks


class FakeSpeaker:
    def __init__(self, log):
        self.log, self.first_audio_delay = log, 0.3

    def feed(self, t): self.log.append(("feed", t))
    def finish(self): self.log.append(("finish",))
    def wait(self): self.log.append(("wait",))
    def stop(self): self.log.append(("stop",))


def make(text, llm=None, log=None, audio=AUDIO, history=None):
    log = [] if log is None else log
    sleeps = []
    loop = VoiceLoop(listen=lambda: audio, transcribe=lambda a: Transcript(text, "es", 1.0, 0.5),
                     llm=llm or FakeLLM(), speaker=FakeSpeaker(log),
                     history=history if history is not None else [dict(SYS)],
                     tail_ms=400, sleep=sleeps.append)
    return loop, log, sleeps


class TestHelpers(unittest.TestCase):
    def test_comandos(self):
        self.assertEqual(parse_command("¡Salir!"), "exit")
        self.assertEqual(parse_command("Adiós"), "exit")
        self.assertEqual(parse_command("olvida todo"), "reset")
        self.assertIsNone(parse_command("quiero salir a correr"))

    def test_ruido(self):
        for t in ("", "  ", "¡Suscríbete!", "Gracias por ver el video", "a"):
            self.assertTrue(is_noise(t), t)
        self.assertFalse(is_noise("¿qué hora es?"))

    def test_eco(self):
        said = "Claro Dario, la VRAM libre es de unos quinientos megas."
        self.assertTrue(is_echo("la vram libre es de unos quinientos megas", said))
        self.assertFalse(is_echo("abre el navegador por favor", said))
        self.assertFalse(is_echo("hola", said))        # frase corta: no se descarta
        self.assertFalse(is_echo("la vram libre es de unos", ""))

    def test_trim_history(self):
        h = [dict(SYS)]
        for i in range(10):
            h += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
        t = trim_history(h, 3)
        self.assertEqual(len(t), 7)
        self.assertEqual(t[0], SYS)
        self.assertEqual(t[1]["content"], "u7")
        self.assertEqual(trim_history(h[:5], 3), h[:5])
        self.assertEqual(len(h), 21)                   # no modifica el original


class TestVoiceLoop(unittest.TestCase):
    def test_turno_normal(self):
        heard, loop_log = [], []
        loop, log, sleeps = make("¿qué hora es?")
        loop.on_heard = heard.append
        t = loop.run_turn()
        self.assertEqual(t.kind, "reply")
        self.assertEqual(t.reply, "Hola, Dario.")
        self.assertEqual(heard, ["¿qué hora es?"])
        self.assertEqual([h["role"] for h in loop.history], ["system", "user", "assistant"])
        self.assertAlmostEqual(t.first_audio_s, 0.3)
        self.assertGreaterEqual(t.perceived_s, 0.3)
        self.assertEqual(sleeps, [0.4])                 # cola anti-eco tras hablar

    def test_micro_cerrado_hasta_que_calla(self):
        loop, log, _ = make("hola jarvis")
        loop.run_turn()
        names = [e[0] for e in log]
        self.assertEqual(names, ["stop", "feed", "feed", "finish", "wait"])

    def test_silencio_ruido_y_eco(self):
        loop, _, _ = make("x", audio=np.zeros(0, dtype=np.float32))
        self.assertEqual(loop.run_turn().kind, "silence")
        loop, _, _ = make("Suscríbete")
        self.assertEqual(loop.run_turn().kind, "noise")
        loop, _, _ = make("la vram libre es de unos quinientos megas")
        loop.last_reply = "Claro, la VRAM libre es de unos quinientos megas."
        t = loop.run_turn()
        self.assertEqual(t.kind, "echo")
        self.assertEqual(len(loop.history), 1)          # no llegó al LLM

    def test_salir_y_reset(self):
        loop, _, _ = make("salir")
        self.assertEqual(loop.run_turn().kind, "exit")
        h = [dict(SYS), {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        loop, log, _ = make("reinicia", history=h)
        self.assertEqual(loop.run_turn().kind, "reset")
        self.assertEqual(loop.history, [SYS])
        self.assertIn(("feed", "Conversación reiniciada."), log)

    def test_error_del_llm_deja_el_historial_limpio(self):
        loop, log, _ = make("hola jarvis", llm=FakeLLM(fail=True))
        t = loop.run_turn()
        self.assertEqual(t.kind, "error")
        self.assertIn("caído", t.error)
        self.assertEqual(loop.history, [SYS])

    def test_el_llm_solo_ve_los_ultimos_turnos(self):
        h = [dict(SYS)]
        for i in range(10):
            h += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
        llm = FakeLLM()
        loop, _, _ = make("pregunta nueva sobre redes", llm=llm, history=h)
        loop.max_turns = 2
        loop.run_turn()
        self.assertEqual(len(llm.seen[0]), 1 + 4 + 1)  # sistema + 2 intercambios previos + pregunta nueva
        self.assertEqual(llm.seen[0][-1]["content"], "pregunta nueva sobre redes")
        self.assertEqual(len(loop.history), 23)           # el historial completo se conserva


if __name__ == "__main__":
    unittest.main()
