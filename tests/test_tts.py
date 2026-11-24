import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from tts import TTSError, load_backend
from tts.base import Audio, TTSBackend
from tts.playback import save_wav


class FakeBackend(TTSBackend):
    def stream(self, text):
        for _ in text.split("."):
            yield Audio(np.zeros(1000, dtype=np.float32), 22050)


class TTSTests(unittest.TestCase):
    def test_synthesize_concatena_trozos(self):
        a = FakeBackend().synthesize("uno.dos.tres")
        self.assertEqual(len(a.samples), 3000)
        self.assertAlmostEqual(a.seconds, 3000 / 22050)

    def test_save_wav(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x" / "a.wav"
            save_wav(Audio(np.zeros(500, dtype=np.float32), 16000), p)
            with wave.open(str(p)) as w:
                self.assertEqual((w.getframerate(), w.getnframes(), w.getnchannels()), (16000, 500, 1))

    def test_backend_desconocido(self):
        with self.assertRaises(TTSError):
            load_backend({"backend": "nada"})

    def test_piper_sin_voz(self):
        with self.assertRaises(TTSError):
            load_backend({"backend": "piper"})


if __name__ == "__main__":
    unittest.main()
