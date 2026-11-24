"""Embeddings locales para la memoria.

`OllamaEmbedder` usa /api/embed de Ollama. Por defecto corre en CPU (`num_gpu: 0`):
con el LLM y la voz cargados quedan solo ~535 MB de VRAM libres (docs/fase2.md).
"""
from abc import ABC, abstractmethod

import numpy as np
import requests


class EmbedError(RuntimeError):
    """No se pudo calcular un embedding (Ollama caído, modelo sin descargar...)."""


def normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.size == 0:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class Embedder(ABC):
    model: str

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Matriz (n, d) float32 con filas normalizadas (coseno = producto punto)."""


class OllamaEmbedder(Embedder):
    def __init__(self, model: str = "bge-m3", host: str = "http://127.0.0.1:11434",
                 device: str = "cpu", keep_alive: str = "10m", timeout: float = 60):
        self.model = model
        self.host = host.rstrip("/")
        self.device = device
        self.keep_alive = keep_alive
        self.timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        payload = {"model": self.model, "input": texts, "keep_alive": self.keep_alive}
        if self.device == "cpu":
            payload["options"] = {"num_gpu": 0}
        try:
            r = requests.post(f"{self.host}/api/embed", json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise EmbedError(f"No puedo conectar con Ollama en {self.host}.") from e
        except requests.RequestException as e:
            raise EmbedError(f"Error HTTP con Ollama: {e}") from e
        if r.status_code == 404:
            raise EmbedError(f"Modelo de embeddings '{self.model}' no encontrado. "
                             f"Descárgalo con: ollama pull {self.model}")
        if not r.ok:
            raise EmbedError(f"Ollama respondió {r.status_code}: {r.text[:200]}")
        try:
            vecs = np.asarray(r.json()["embeddings"], dtype=np.float32)
        except (KeyError, ValueError) as e:
            raise EmbedError(f"Respuesta de embeddings inesperada: {e}") from e
        if vecs.ndim != 2 or len(vecs) != len(texts):
            raise EmbedError("Ollama devolvió un número de embeddings distinto al pedido.")
        return normalize_rows(vecs)


def load_embedder(cfg: dict) -> Embedder:
    cfg = dict(cfg or {})
    backend = cfg.pop("backend", "ollama")
    if backend == "ollama":
        return OllamaEmbedder(**cfg)
    raise EmbedError(f"Backend de embeddings desconocido: {backend}")
