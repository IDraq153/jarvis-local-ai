"""Backend de Ollama (API HTTP local, por defecto http://127.0.0.1:11434)."""
import json
from typing import Iterator

import requests

from .base import LLMBackend, LLMError, Message, ToolCall


class OllamaBackend(LLMBackend):
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434",
                 timeout: float = 300, keep_alive: str = "10m",
                 think: bool | None = None, **default_options):
        if not model:
            raise LLMError("No hay modelo configurado (llm.model en config.yaml).")
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.think = think  # False apaga el modo 'pensar' (qwen3); None = por defecto
        self.default_options = default_options  # p. ej. temperature, num_ctx
        self.last_stats: dict = {}
        self.last_thinking_chars = 0  # diagnóstico: texto de 'pensar' recibido

    def stream(self, messages: list[Message], **options) -> Iterator[str]:
        for item in self.stream_events(messages, None, **options):
            yield item

    def stream_events(self, messages: list[Message], tools: list[dict] | None = None,
                      **options) -> Iterator[str | ToolCall]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {**self.default_options, **options},
        }
        if self.think is not None:
            payload["think"] = self.think
        if tools:
            payload["tools"] = tools
        self.last_thinking_chars = 0
        try:
            with requests.post(f"{self.host}/api/chat", json=payload,
                               stream=True, timeout=self.timeout) as r:
                if r.status_code == 404:
                    raise LLMError(f"Modelo '{self.model}' no encontrado. "
                                   f"Descárgalo con: ollama pull {self.model}")
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise LLMError(data["error"])
                    msg = data.get("message", {})
                    self.last_thinking_chars += len(msg.get("thinking", "") or "")
                    chunk = msg.get("content", "")
                    if chunk:
                        yield chunk
                    for call in msg.get("tool_calls") or []:
                        parsed = _parse_tool_call(call)
                        if parsed:
                            yield parsed
                    if data.get("done"):
                        self.last_stats = data
        except requests.ConnectionError as e:
            raise LLMError(f"No puedo conectar con Ollama en {self.host}. "
                           f"¿Está abierto el servicio?") from e
        except requests.RequestException as e:
            raise LLMError(f"Error HTTP con Ollama: {e}") from e

    def warmup(self) -> None:
        """Carga el modelo en memoria para que la primera respuesta no pague la carga."""
        try:
            requests.post(f"{self.host}/api/chat", timeout=self.timeout,
                          json={"model": self.model, "messages": [], "keep_alive": self.keep_alive})
        except requests.RequestException as e:
            raise LLMError(f"No puedo conectar con Ollama en {self.host}.") from e

    def tokens_per_second(self) -> float | None:
        s = self.last_stats
        if s.get("eval_count") and s.get("eval_duration"):
            return s["eval_count"] / (s["eval_duration"] / 1e9)
        return None


def _parse_tool_call(raw) -> ToolCall | None:
    """{"function": {"name": ..., "arguments": {...}}} -> ToolCall (arguments puede venir como texto JSON)."""
    fn = (raw or {}).get("function") or {}
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return ToolCall(name, args if isinstance(args, dict) else {})
