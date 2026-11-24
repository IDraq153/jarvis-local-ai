"""Interfaz común de los cerebros (LLM) de JARVIS.

Cualquier backend (Ollama, llama.cpp, ...) implementa `LLMBackend.stream`.
El resto del sistema solo conoce esta interfaz.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


class LLMError(RuntimeError):
    """Error al hablar con el backend (servidor caído, modelo ausente, ...)."""


@dataclass
class ToolCall:
    """Una herramienta que el LLM decidió llamar (argumentos crudos, tal como los generó)."""
    name: str
    args: dict = field(default_factory=dict)


class LLMBackend(ABC):
    @abstractmethod
    def stream(self, messages: list[Message], **options) -> Iterator[str]:
        """Devuelve la respuesta por trozos de texto, a medida que se genera."""

    def chat(self, messages: list[Message], **options) -> str:
        return "".join(self.stream(messages, **options))

    def stream_events(self, messages: list[Message], tools: list[dict] | None = None,
                      **options) -> Iterator[str | ToolCall]:
        """Como `stream`, pero con `tools` (esquemas de Ollama): produce trozos de texto (str)
        y, si el modelo decide usar una herramienta, objetos `ToolCall`.
        La implementación por defecto no soporta herramientas y las ignora."""
        yield from self.stream(messages, **options)
