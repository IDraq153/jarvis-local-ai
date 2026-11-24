"""Pruebas del módulo stt/ sin micrófono ni modelos.
Ejecutar: python -m unittest discover -s tests -v
"""
import types
import unittest

import numpy as np

from stt import STTError, load_backend
from stt.base import SAMPLE_RATE
from stt.faster_whisper_backend import FasterWhisperBackend
from stt.metrics import normalize, wer
from stt.recorder import FRAME, Endpointer, rms


def frames(seconds, amp):
    n = int(seconds * SAMPLE_RATE / FRAME)
    return [np.full(FRAME, amp, dtype=np.float32) for _ in range(n)]


def run(ep, chunks):
    for f in chunks:
        if ep.feed(f):
            break
    return ep


class TestMetrics(unittest.TestCase):
    def test_normalize_ignora_tildes_y_puntuacion(self):
        self.assertEqual(normalize("¡Hola, Darío!"), ["hola", "dario"])

    def test_wer(self):
        self.assertEqual(wer("abre el navegador", "Abre el navegador."), 0.0)
        self.assertAlmostEqual(wer("abre el navegador", "abre un navegador"), 1 / 3)
        self.assertEqual(wer("hola", ""), 1.0)
        self.assertEqual(wer("", ""), 0.0)


class TestEndpointer(unittest.TestCase):
    def test_frase_completa(self):
        ep = Endpointer(threshold=0.05, silence_ms=300)
        run(ep, frames(0.5, 0.0) + frames(1.0, 0.2) + frames(0.6, 0.0) + frames(1.0, 0.0))
        self.assertTrue(ep.done)
        secs = len(ep.audio()) / SAMPLE_RATE
        self.assertGreater(secs, 1.0)     # voz + preroll + cola de silencio
        self.assertLess(secs, 2.2)

    def test_solo_silencio_no_empieza(self):
        ep = run(Endpointer(threshold=0.05), frames(2.0, 0.001))
        self.assertFalse(ep.started)
        self.assertEqual(len(ep.audio()), 0)

    def test_ruido_breve_se_descarta(self):
        ep = Endpointer(threshold=0.05, silence_ms=300, min_speech_ms=400)
        run(ep, frames(0.1, 0.2) + frames(1.0, 0.0))
        self.assertEqual(len(ep.audio()), 0)

    def test_limite_de_duracion(self):
        ep = run(Endpointer(threshold=0.05, max_seconds=1.0), frames(3.0, 0.2))
        self.assertTrue(ep.done)
        self.assertLessEqual(len(ep.audio()) / SAMPLE_RATE, 1.3)

    def test_rms(self):
        self.assertAlmostEqual(rms(np.full(100, 0.5, dtype=np.float32)), 0.5, places=5)


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, samples, **kw):
        self.kwargs = kw
        segs = iter([types.SimpleNamespace(text=" hola "), types.SimpleNamespace(text="mundo ")])
        return segs, types.SimpleNamespace(language="es")


class TestFasterWhisper(unittest.TestCase):
    def test_transcribe_une_segmentos_y_pasa_opciones(self):
        b = FasterWhisperBackend(model="small", language="es", beam_size=1)
        b._model = FakeModel()
        r = b.transcribe(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
        self.assertEqual(r.text, "hola mundo")
        self.assertEqual(r.language, "es")
        self.assertAlmostEqual(r.audio_seconds, 2.0)
        self.assertEqual(b._model.kwargs["language"], "es")
        self.assertEqual(b._model.kwargs["beam_size"], 1)
        self.assertTrue(b._model.kwargs["vad_filter"])

    def test_idioma_auto(self):
        b = FasterWhisperBackend(language=None)
        self.assertIsNone(b.language)

    def test_load_backend_ignora_claves_del_micro(self):
        b = load_backend({"backend": "faster-whisper", "model": "base", "device": "cpu",
                          "silence_ms": 800, "threshold": None, "device_index": None})
        self.assertEqual(b.model_name, "base")

    def test_backend_desconocido(self):
        with self.assertRaises(STTError):
            load_backend({"backend": "nada"})


if __name__ == "__main__":
    unittest.main()
