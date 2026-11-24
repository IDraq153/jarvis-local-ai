import sys
import unittest
from pathlib import Path

from tts import TTSError
from tts.text import split_sentences
from tts.worker_backend import WorkerBackend


class FakeBackend(WorkerBackend):
    name = "fake"
    worker_script = Path(__file__).with_name("fake_worker.py")


class WorkerTests(unittest.TestCase):
    def test_split(self):
        self.assertEqual(split_sentences("Hola. ¿Qué tal? Bien."), ["Hola.", "¿Qué tal?", "Bien."])
        self.assertEqual(split_sentences("IP 192.168.1.10 ok."), ["IP 192.168.1.10 ok."])
        largo = ", ".join(["palabra"] * 60)
        partes = split_sentences(largo, 100)
        self.assertTrue(all(len(p) <= 100 for p in partes) and len(partes) > 1)

    def test_protocolo(self):
        b = FakeBackend(sys.executable)
        try:
            a = b.synthesize("Una. Dos.")
            self.assertEqual(a.sample_rate, 24000)
            self.assertEqual(len(a.samples), 4800)  # 2 frases x 2400 muestras
            with self.assertRaises(TTSError):
                b.synthesize("falla")
        finally:
            b.unload()

    def test_streaming_varios_trozos(self):
        b = FakeBackend(sys.executable)
        try:
            trozos = list(b.stream("stream"))
            self.assertEqual([len(t.samples) for t in trozos], [1000, 1000, 1000])
        finally:
            b.unload()

    def test_python_ausente(self):
        with self.assertRaises(TTSError):
            FakeBackend("/no/existe/python").load()


if __name__ == "__main__":
    unittest.main()
