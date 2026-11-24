"""Lee en voz alta el texto que llega por trozos, con la síntesis y la reproducción en segundo plano.

Uso: speaker.feed(token) por cada trozo del LLM, speaker.finish() al terminar la respuesta,
speaker.stop() para callar (p. ej. cuando el usuario escribe otra cosa).
"""
import queue
import threading
import time

import numpy as np

from .base import Audio, TTSBackend, TTSError
from .text import SentenceBuffer, clean_for_speech


class SoundDeviceSink:
    """Reproduce el audio sin cortes con un OutputStream continuo."""

    def __init__(self):
        try:
            import sounddevice as sd
        except ImportError as e:
            raise TTSError("Falta sounddevice: pip install sounddevice") from e
        self._sd = sd
        self._stream = None
        self._rate = None

    def __call__(self, audio: Audio) -> None:
        if self._stream is None or audio.sample_rate != self._rate:
            self.abort()
            self._stream = self._sd.OutputStream(samplerate=audio.sample_rate, channels=1, dtype="float32")
            self._stream.start()
            self._rate = audio.sample_rate
        self._stream.write(np.ascontiguousarray(audio.samples.reshape(-1, 1), dtype=np.float32))

    def abort(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.abort()
                s.close()
            except Exception:
                pass


class Speaker:
    def __init__(self, backend: TTSBackend, sink=None, on_first_audio=None, min_chars: int = 25):
        self.backend = backend
        self._sink = sink or SoundDeviceSink()
        self._on_first_audio = on_first_audio
        self._min_chars = min_chars
        self._buf = SentenceBuffer(min_chars)
        self._text_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue()
        self._gen = 0
        self._t_feed = None
        self._announced = False
        self._play_end = 0.0           # instante (perf_counter) en que terminará de sonar lo ya encolado
        self.first_audio_delay: float | None = None
        self.last_error: str | None = None
        for target in (self._synth_loop, self._play_loop):
            threading.Thread(target=target, daemon=True).start()

    # --- API pública -------------------------------------------------------
    def feed(self, text: str) -> None:
        if self._t_feed is None:
            self._t_feed = time.perf_counter()
        for sentence in self._buf.push(text):
            self._enqueue(sentence)

    def finish(self) -> None:
        for sentence in self._buf.flush():
            self._enqueue(sentence)

    def stop(self) -> None:
        self._gen += 1
        self._buf = SentenceBuffer(self._min_chars)
        self._drain(self._text_q)
        self._drain(self._audio_q)
        abort = getattr(self._sink, "abort", None)
        if abort:
            abort()
        self._t_feed, self._announced = None, False
        self._play_end = 0.0

    def wait(self) -> None:
        self._text_q.join()
        self._audio_q.join()

    def wait_playback(self) -> None:
        """`wait()` + espera a que lo escrito en la tarjeta de sonido TERMINE de sonar.

        `wait()` vuelve cuando el último trozo entró al buffer del dispositivo, que todavía lo está reproduciendo
        (con voces largas o buffers grandes de Windows, hasta varios segundos). Se usa antes de abrir el video de
        «estoy en casa» y de reabrir el micrófono.
        """
        self.wait()
        remaining = self._play_end - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    def close(self) -> None:
        self.stop()
        self.backend.unload()

    # --- internos ------------------------------------------------------------
    def _enqueue(self, sentence: str) -> None:
        sentence = clean_for_speech(sentence)
        if sentence:
            self._text_q.put((self._gen, sentence))

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                return

    def _synth_loop(self) -> None:
        while True:
            gen, sentence = self._text_q.get()
            try:
                for chunk in self.backend.stream(sentence):
                    if gen != self._gen:
                        break
                    self._audio_q.put((gen, chunk))
            except TTSError as e:
                self.last_error = str(e)
                print(f"\n[Voz] {e}")
            finally:
                self._text_q.task_done()

    def _play_loop(self) -> None:
        while True:
            gen, audio = self._audio_q.get()
            try:
                if gen == self._gen:
                    if not self._announced:
                        self._announced = True
                        self.first_audio_delay = time.perf_counter() - (self._t_feed or time.perf_counter())
                        if self._on_first_audio:
                            self._on_first_audio(self.first_audio_delay)
                    t0 = time.perf_counter()
                    self._sink(audio)
                    rate = getattr(audio, "sample_rate", 0) or 0
                    n = len(getattr(audio, "samples", ()) )
                    if rate and n:
                        self._play_end = max(self._play_end, t0) + n / rate
            except Exception as e:  # p. ej. el stream se abortó al hacer stop()
                self.last_error = str(e)
            finally:
                self._audio_q.task_done()
