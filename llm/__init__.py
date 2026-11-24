from .base import LLMBackend, LLMError, ToolCall
from .ollama_backend import OllamaBackend


def load_backend(cfg: dict) -> LLMBackend:
    """Crea el backend a partir de la sección `llm` de config.yaml."""
    cfg = dict(cfg)
    backend = cfg.pop("backend", "ollama")
    if backend == "ollama":
        cfg.pop("candidates", None)
        options = cfg.pop("options", {}) or {}
        return OllamaBackend(**cfg, **options)
    raise LLMError(f"Backend LLM desconocido: {backend}")
