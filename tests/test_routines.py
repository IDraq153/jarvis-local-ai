"""Pruebas de la rutina "Jarvis, estoy en casa" (Fase 6): reconoce la frase, habla, y abre el video DESPUÉS de hablar."""
import datetime as dt
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.voice_loop import VoiceLoop
from stt.base import Transcript
from tests.test_unigym_readers import state as base_state
from tools.routines import PHRASES, RoutineRunner, is_home_phrase, valid_url
from tools.unigym.audit import AuditLog
from tools.unigym.repo import JsonFileRepository

URL = "https://www.youtube.com/watch?v=xMaE6toi4mk&list=RDxMaE6toi4mk&start_radio=1"
TUE = dt.date(2026, 11, 24)


class Frases(unittest.TestCase):
    def test_activan(self):
        for t in ("Jarvis, estoy en casa", "estoy en casa", "Jarvis estoy en casa.", "Ya estoy en casa", "Jarvis, ya llegué",
                  "¡Ya llegué a casa!", "oye Jarvis estoy en casa"):
            self.assertTrue(is_home_phrase(t), t)

    def test_no_activan(self):
        for t in ("estoy en casa de mi tío", "Jarvis, ¿estoy en casa?", "no estoy en casa", "cuando llegue a casa avísame",
                  "estoy en casa estudiando redes", "casa", "", "qué tengo hoy", "estoy en la universidad"):
            self.assertFalse(is_home_phrase(t), t)

    def test_url(self):
        self.assertTrue(valid_url(URL, ["youtube.com", "youtu.be"]))
        self.assertTrue(valid_url("https://music.youtube.com/watch?v=x", ["youtube.com"]))
        for bad in ("http://www.youtube.com/watch?v=x", "https://evil.com/youtube.com", "https://youtube.com.evil.com/x",
                    "file:///C:/x", "javascript:alert(1)", "", "https://notyoutube.com/x"):
            self.assertFalse(valid_url(bad, ["youtube.com", "youtu.be"]), bad)


class Rutina(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        src = d / "r.json"
        src.write_text(json.dumps({"app": "uni-gym-tracker", "version": 1, "data": base_state()}), encoding="utf-8")
        self.repo = JsonFileRepository.from_backup(src, d / "u.json", keep=3)
        self.audit = AuditLog(":memory:")
        self.opened, self.slept = [], []
        self.runner = RoutineRunner({"url": URL}, repo=self.repo, audit=self.audit, today=lambda: TUE,
                                    opener=lambda u: self.opened.append(u) or "firefox", rng=random.Random(1),
                                    sleep=lambda s: self.slept.append(s))

    def tearDown(self):
        self._tmp.cleanup()

    def test_saluda_dice_lo_urgente_y_una_frase(self):
        rr = self.runner.match("Jarvis, estoy en casa")
        self.assertTrue(rr.speech.startswith("Bienvenido, señor."))
        self.assertIn("Lo más urgente:", rr.speech)
        self.assertTrue(any(p in rr.speech for p in PHRASES))
        self.assertEqual(self.opened, [])                      # todavía NO se abrió: primero habla

    def test_abre_firefox_con_la_url_de_config(self):
        rr = self.runner.match("estoy en casa")
        self.assertEqual(rr.after(), "firefox")
        self.assertEqual(self.opened, [URL])
        self.assertEqual([(r["tool"], r["status"]) for r in self.audit.tail(5)],
                         [("estoy_en_casa", "said"), ("estoy_en_casa", "opened")])

    def test_espera_un_margen_solo_si_abre_despues_de_hablar(self):
        self.runner.match("estoy en casa").after()
        self.assertEqual(self.slept, [])                                    # por defecto: a la vez que habla, sin pausa
        r = RoutineRunner({"url": URL, "open_with_speech": False}, opener=lambda u: "firefox", sleep=self.slept.append)
        r.match("estoy en casa").after()
        self.assertEqual(self.slept, [1.0])

    def test_sin_tareas_no_inventa_lo_urgente(self):
        self.repo.write(lambda s: s["tasks"].clear(), by="test")
        rr = self.runner.match("estoy en casa")
        self.assertNotIn("urgente", rr.speech)

    def test_sin_datos_sigue_funcionando(self):
        r = RoutineRunner({"url": URL}, repo=None, opener=lambda u: "firefox")
        self.assertTrue(r.match("estoy en casa").speech.startswith("Bienvenido, señor."))

    def test_url_no_permitida_no_se_abre(self):
        r = RoutineRunner({"url": "https://evil.com/x"}, audit=self.audit, opener=lambda u: self.opened.append(u))
        self.assertEqual(r.match("estoy en casa").after(), "sin video")
        self.assertEqual(self.opened, [])
        self.assertEqual(self.audit.tail(1)[0]["status"], "rejected")

    def test_si_el_navegador_falla_no_tumba(self):
        def boom(u): raise OSError("sin firefox")
        r = RoutineRunner({"url": URL}, audit=self.audit, opener=boom)
        self.assertEqual(r.match("estoy en casa").after(), "error")
        self.assertEqual(self.audit.tail(1)[0]["status"], "error")

    def test_desactivada(self):
        r = RoutineRunner({"url": URL, "enabled": False}, opener=lambda u: None)
        self.assertIsNone(r.match("estoy en casa"))

    def test_otra_frase_no_hace_nada(self):
        self.assertIsNone(self.runner.match("estoy en casa de mi tío"))


class EnElBucle(Rutina):
    def test_el_video_arranca_a_la_vez_que_habla_por_defecto(self):
        log = self._run_loop({"url": URL})
        self.assertEqual([e[0] for e in log], ["open", "feed", "finish", "wait"])

    def test_el_video_se_abre_despues_de_hablar(self):
        log = self._run_loop({"url": URL, "open_with_speech": False})
        self.assertEqual([e[0] for e in log], ["feed", "finish", "wait", "open"])
        self.assertEqual(log[-1], ("open", URL))

    def _run_loop(self, cfg):
        log = []
        self.runner = RoutineRunner(cfg, repo=self.repo, audit=self.audit, today=lambda: TUE, rng=random.Random(1),
                                    sleep=lambda s: None, opener=lambda u: log.append(("open", u)) or "firefox")

        class Spk:
            first_audio_delay = 0.1
            def feed(s, t): log.append(("feed", t))
            def finish(s): log.append(("finish",))
            def wait(s): log.append(("wait",))
            def stop(s): pass

        class LLM:
            def stream(s, m, **k): raise AssertionError("la rutina no debe llamar al LLM")
        loop = VoiceLoop(listen=lambda: np.ones(1600, dtype=np.float32),
                         transcribe=lambda a: Transcript("Jarvis, estoy en casa", "es", 1.0, 0.5), llm=LLM(), speaker=Spk(),
                         history=[{"role": "system", "content": "s"}], tail_ms=0, sleep=lambda s: None, routines=self.runner)
        t = loop.run_turn()
        self.assertEqual((t.kind, t.tool), ("tool", "estoy_en_casa"))
        self.assertEqual(len(loop.history), 1)                              # la rutina no entra al historial del LLM
        return log


if __name__ == "__main__":
    unittest.main()
