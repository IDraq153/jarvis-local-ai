"""Selección de herramientas por parecido (embeddings) para no gastar el contexto de 4096.

Cada herramienta se embebe una vez (descripción + frases de ejemplo). En cada turno se
embebe lo que dijo Dario y se pasan al LLM solo las `top_k` herramientas más parecidas
cuyo mejor parecido supere `min_score`. Si ninguna lo supera, no se envía ningún esquema:
la charla normal no paga tokens y el LLM no se siente tentado a llamar herramientas.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Scored:
    spec: object
    score: float


class ToolSelector:
    def __init__(self, specs, embedder, top_k: int = 4, min_score: float = 0.45):
        self.specs = list(specs)
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self._mat = None      # (n_textos, d)
        self._owner = []      # índice de herramienta por fila

    def index(self) -> int:
        """Embebe el catálogo. Devuelve cuántos textos se indexaron."""
        texts, owner = [], []
        for i, spec in enumerate(self.specs):
            for t in spec.embed_texts():
                texts.append(t)
                owner.append(i)
        self._mat = self.embedder.embed(texts)
        self._owner = np.asarray(owner)
        return len(texts)

    def rank(self, query: str) -> list[Scored]:
        """Todas las herramientas con su mejor parecido, de mayor a menor."""
        if self._mat is None:
            self.index()
        q = self.embedder.embed([query])[0]
        sims = self._mat @ q
        best = np.full(len(self.specs), -1.0, dtype=np.float32)
        for row, s in zip(self._owner, sims):
            if s > best[row]:
                best[row] = s
        order = np.argsort(-best)
        return [Scored(self.specs[i], float(best[i])) for i in order]

    def select(self, query: str) -> list:
        """Las herramientas (ToolSpec) que merecen ir en el prompt de este turno."""
        return [s.spec for s in self.rank(query)[: self.top_k] if s.score >= self.min_score]
