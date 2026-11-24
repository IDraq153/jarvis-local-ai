import time
import unittest

import numpy as np

from tts.base import Audio, TTSBackend
from tts.speaker import Speaker
from tts.text import SentenceBuffer, clean_for_speech


class Eco(TTSBackend):
    def __init__(self):
        self.frases = []

    def stream(self, text):
        self.frases.append(text)
        yield Audio(np.zeros(240, dtype=np.float32), 24000)


class TextTests(unittest.TestCase):
    def test_limpia_markdown(self):
        self.assertEqual(clean_for_speech("**Hola** `Dario` 😀 ```x=1``` ok"), "Hola Dario ok")

    def test_frases_y_decimales(self):
        b = SentenceBuffer(min_chars=10)
        frases = b.push("La IP 192.168.1.10 responde. Pesa 3.5 kg. Fin")
        self.assertEqual(frases, ["La IP 192.168.1.10 responde.", "Pesa 3.5 kg."])
        self.assertEqual(b.flush(), ["Fin"])

    def test_junta_frases_cortas(self):
        b = SentenceBuffer(min_chars=20)
        self.assertEqual(b.push("Sí. Claro que sí, Dario. "), ["Sí. Claro que sí, Dario."])


class SpeakerTests(unittest.TestCase):
    def test_lee_en_orden_y_stop(self):
        salida, eco = [], Eco()
        sp = Speaker(eco, sink=salida.append)
        for tok in ["Primera frase larga de prueba. ", "Segunda frase también larga. ", "Cola"]:
            sp.feed(tok)
        sp.finish()
        sp.wait()
        self.assertEqual(eco.frases, ["Primera frase larga de prueba.", "Segunda frase también larga.", "Cola"])
        self.assertEqual(len(salida), 3)
        self.assertIsNotNone(sp.first_audio_delay)
        sp.feed("Otra frase que será cancelada. ")
        sp.stop()
        sp.wait()  # no debe colgarse


if __name__ == "__main__":
    unittest.main()
