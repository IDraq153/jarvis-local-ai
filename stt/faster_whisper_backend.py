"""faster-whisper (CTranslate2). Por defecto en CPU con int8: no usa nada de VRAM (Fase 3)."""
import time

import numpy as np

from .base import SAMPLE_RATE, STTBackend, STTError, Transcript


class FasterWhisperBackend(STTBackend):
    name = "faster-whisper"

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8",
                 language: str | None = "es", beam_size: int = 1, vad_filter: bool = True,
                 cpu_threads: int = 0, initial_prompt: str | None = None):
        self.model_name = model or "small"
        self.device = device
        self.compute_type = compute_type
        self.language = language or None  # None = detectar idioma
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.cpu_threads = cpu_threads    # 0 = valor por defecto de CTranslate2
        self.initial_prompt = initial_prompt
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise STTError("Falta faster-whisper: pip install -r requirements-stt.txt") from e
        try:
            self._model = WhisperModel(self.model_name, device=self.device,
                                       compute_type=self.compute_type, cpu_threads=self.cpu_threads)
        except Exception as e:  # descarga fallida, CUDA ausente, tipo de cómputo no soportado...
            raise STTError(f"No se pudo cargar '{self.model_name}' en {self.device}/{self.compute_type}: {e}") from e

    def unload(self) -> None:
        self._model = None

    def transcribe(self, samples: np.ndarray) -> Transcript:
        self.load()
        samples = np.asarray(samples, dtype=np.float32)
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            samples, language=self.language, beam_size=self.beam_size,
            vad_filter=self.vad_filter, initial_prompt=self.initial_prompt,
            condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()  # el generador se consume aquí
        return Transcript(text=text, language=info.language,
                          audio_seconds=len(samples) / SAMPLE_RATE,
                          compute_seconds=time.perf_counter() - t0)
