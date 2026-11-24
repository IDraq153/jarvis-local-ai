"""Interfaz común de los cerebros (LLM) de JARVIS.

Cualquier backend (Ollama, llama.cpp, ...) implementa `LLMBackend.stream`.
El resto del sistema solo conoce esta interfaz.
"""
from abc import ABC, abstractmethod
from typing import Iterator

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


class LLMError(RuntimeError):
    """Error al hablar con el backend (servidor caído, modelo ausente, ...)."""


class LLMBackend(ABC):
    @abstractmethod
    def stream(self, messages: list[Message], **options) -> Iterator[str]:
        """Devuelve la respuesta por trozos de texto, a medida que se genera."""

    def chat(self, messages: list[Message], **options) -> str:
        return "".join(self.stream(messages, **options))
