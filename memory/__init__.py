"""Memoria persistente de JARVIS (Fase 5): SQLite + embeddings locales."""
from pathlib import Path

from .embeddings import EmbedError, Embedder, OllamaEmbedder, load_embedder
from .manager import Memory, MemoryConfig
from .store import MemoryStore


def load_memory(cfg: dict, base_dir, warn=print) -> Memory:
    """Crea la memoria a partir de la sección `memory` de config.yaml (ruta relativa a `base_dir`)."""
    cfg = dict(cfg or {})
    db_path = Path(base_dir) / cfg.get("db_path", "data/jarvis_memory.db")
    return Memory(MemoryStore(db_path), load_embedder(cfg.get("embedder")),
                  MemoryConfig.from_dict(cfg), warn=warn)
