"""Interfaz común de las voces (TTS) de JARVIS.

Cualquier motor (Piper, Kokoro, XTTS, F5-TTS, ...) implementa `TTSBackend.stream`.
El resto del sistema solo conoce esta interfaz, igual que con `llm/`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


class TTSError(RuntimeError):
    """Error del motor de voz (modelo ausente, dependencia no instalada, ...)."""


@dataclass
class Audio:
    samples: np.ndarray  # float32 mono en [-1, 1]
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0


class TTSBackend(ABC):
    name: str = "tts"

    def load(self) -> None:
        """Carga el modelo en memoria (idempotente). Por defecto no hace nada."""

    def unload(self) -> None:
        """Libera el modelo (y la VRAM si la usaba)."""

    @abstractmethod
    def stream(self, text: str) -> Iterator[Audio]:
        """Devuelve el audio por trozos (p. ej. una frase), a medida que se genera."""

    def synthesize(self, text: str) -> Audio:
        chunks = list(self.stream(text))
        if not chunks:
            raise TTSError("El motor no devolvió audio.")
        return Audio(np.concatenate([c.samples for c in chunks]), chunks[0].sample_rate)
