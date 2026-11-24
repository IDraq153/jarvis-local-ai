import time
import unittest

import numpy as np

from tts.base import Audio, TTSBackend
from tts.speaker import Speaker
from tts.text import SentenceBuffer, clean_for_speech, speak_times, spoken_time


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


class HorasHabladas(unittest.TestCase):
    def test_horas_sueltas(self):
        casos = {"06:30": "seis y media de la mañana", "6:30": "seis y media de la mañana", "18:00": "seis de la tarde",
                 "18:30": "seis y media de la tarde", "13:05": "una y cinco de la tarde", "07:45": "ocho menos cuarto de la mañana",
                 "12:00": "doce del día", "00:00": "doce de la noche", "21:15": "nueve y cuarto de la noche",
                 "09:20": "nueve y veinte de la mañana", "23:59": "once y cincuenta y nueve de la noche",
                 "03:40": "tres y cuarenta de la madrugada"}
        for texto, dicho in casos.items():
            self.assertEqual(speak_times(texto), dicho, texto)

    def test_en_frases_y_rangos(self):
        self.assertEqual(speak_times("Despierta a las 06:30."), "Despierta a las seis y media de la mañana.")
        self.assertEqual(speak_times("de 07:00 a 09:00 en O2-302"), "de siete a nueve de la mañana en O2-302")
        self.assertEqual(speak_times("de 11:00 a 14:00"), "de once de la mañana a dos de la tarde")
        self.assertEqual(speak_times("a las 13:00"), "a la una de la tarde")
        self.assertEqual(speak_times("de 18:00 a 20:30"), "de seis de la tarde a ocho y media de la noche")

    def test_no_toca_lo_que_no_es_hora(self):
        for t in ("IP 192.168.1.10", "relación 3:5", "10:75", "versión 1.2.3", "2026-11-27", "puerto 8080"):
            self.assertEqual(speak_times(t), t, t)

    def test_no_lee_vinetas(self):
        self.assertEqual(clean_for_speech("Resumen:\n- **Estudia** TCP\n- Haz ejercicios\n1. Repasa"), "Resumen: Estudia TCP Haz ejercicios Repasa")

    def test_clean_for_speech_las_dice(self):
        self.assertEqual(clean_for_speech("**Gym** a las 06:30 😀"), "Gym a las seis y media de la mañana")


class DecimalesHablados(unittest.TestCase):
    def test_decimales(self):
        from tts.text import speak_decimals as f
        self.assertEqual(f("Tu PPA es 17,4."), "Tu PPA es diecisiete punto cuatro.")
        self.assertEqual(f("llevas 17,61 de promedio"), "llevas diecisiete punto sesenta y uno de promedio")
        self.assertEqual(f("pesas 70.5 kg"), "pesas setenta punto cinco kg")
        self.assertEqual(f("necesitas 10,05"), "necesitas diez punto cero cinco")

    def test_no_toca(self):
        from tts.text import speak_decimals as f
        for t in ("IP 192.168.1.10", "versión 1.2.3", "2026-11-27", "15 de promedio", "Fin. 3 tareas"):
            self.assertEqual(f(t), t, t)

    def test_en_clean_for_speech(self):
        self.assertEqual(clean_for_speech("PPA **17,4** a las 06:30"), "PPA diecisiete punto cuatro a las seis y media de la mañana")


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

    def test_wait_playback_espera_a_que_termine_de_sonar(self):
        class Largo(TTSBackend):
            def stream(self, text):
                yield Audio(np.zeros(2400, dtype=np.float32), 24000)       # 0,1 s de audio
        sp = Speaker(Largo(), sink=lambda a: None)                         # el sink "escribe" al instante, como un buffer grande
        sp.feed("Una frase de prueba bastante larga. ")
        sp.finish()
        t0 = time.perf_counter()
        sp.wait()
        self.assertLess(time.perf_counter() - t0, 0.08)                    # wait() vuelve en cuanto se encoló
        sp.wait_playback()
        self.assertGreaterEqual(time.perf_counter() - t0, 0.09)            # wait_playback() espera los 0,1 s de sonido


if __name__ == "__main__":
    unittest.main()
