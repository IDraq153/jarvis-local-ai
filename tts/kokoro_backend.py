"""Kokoro-82M: modelo pequeño (82 M de parámetros) con voces en español (pip install kokoro)."""
from typing import Iterator

import numpy as np

from .base import Audio, TTSBackend, TTSError

SAMPLE_RATE = 24000


class KokoroBackend(TTSBackend):
    name = "kokoro"

    def __init__(self, voice: str = "ef_dora", lang_code: str = "e",
                 device: str | None = None, speed: float = 1.0, trim_cache: bool = True):
        self.voice = voice          # español: ef_dora, em_alex, em_santa
        self.lang_code = lang_code  # 'e' = español
        self.device = device        # 'cuda', 'cpu' o None (automático)
        self.speed = speed
        self.trim_cache = trim_cache  # libera la caché de CUDA tras cada trozo (menos VRAM reservada)
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise TTSError("Falta kokoro: pip install kokoro soundfile") from e
        kwargs = {"device": self.device} if self.device else {}
        try:
            self._pipe = KPipeline(lang_code=self.lang_code, **kwargs)
        except Exception as e:  # modelo sin descargar, espeak-ng ausente, etc.
            raise TTSError(f"No pude iniciar Kokoro: {e}") from e

    @staticmethod
    def _empty_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload(self) -> None:
        self._pipe = None
        self._empty_cache()

    def stream(self, text: str) -> Iterator[Audio]:
        self.load()
        try:
            for result in self._pipe(text, voice=self.voice, speed=self.speed):
                audio = result.audio if hasattr(result, "audio") else result[2]
                if audio is None:
                    continue
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                out = Audio(np.asarray(audio, dtype=np.float32).reshape(-1), SAMPLE_RATE)
                if self.trim_cache:
                    self._empty_cache()
                yield out
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"Error al sintetizar con Kokoro: {e}") from e
