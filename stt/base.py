"""Interfaz común del oído (STT) de JARVIS.

Cualquier motor (faster-whisper, whisper.cpp, Vosk, ...) implementa `STTBackend.transcribe`.
El resto del sistema solo conoce esta interfaz, igual que con `llm/` y `tts/`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000  # Hz; todos los motores reciben audio mono float32 a 16 kHz


class STTError(RuntimeError):
    """Error del motor de reconocimiento (modelo ausente, dependencia no instalada, ...)."""


@dataclass
class Transcript:
    text: str
    language: str = ""
    audio_seconds: float = 0.0    # duración del audio transcrito
    compute_seconds: float = 0.0  # tiempo que tardó el motor

    @property
    def rtf(self) -> float:
        """Real-time factor: < 1 = más rápido que tiempo real."""
        return self.compute_seconds / self.audio_seconds if self.audio_seconds else 0.0


class STTBackend(ABC):
    name: str = "stt"

    def load(self) -> None:
        """Carga el modelo en memoria (idempotente). Por defecto no hace nada."""

    def unload(self) -> None:
        """Libera el modelo."""

    @abstractmethod
    def transcribe(self, samples: np.ndarray) -> Transcript:
        """Transcribe audio mono float32 a 16 kHz en [-1, 1]."""
