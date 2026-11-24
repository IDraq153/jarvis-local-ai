"""Contrato de las herramientas (Fase 6).

Una herramienta tiene dos partes que viven separadas:
- `ToolSpec`: lo que ve el LLM (nombre, descripción, parámetros) y las frases de ejemplo
  con que se embebe para seleccionarla. Es solo datos; se mide y se prueba sin Ollama.
- el manejador (en `tools/unigym/`): lo que hace Python cuando el LLM la elige.
"""
import json
from dataclasses import dataclass

READ = "read"    # lectura libre, sin confirmación
WRITE = "write"  # escritura: siempre propuesta + confirmación por voz


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict            # JSON Schema de tipo "object"
    examples: tuple = ()        # frases como las diría Dario (se embeben junto a la descripción)
    level: str = READ

    def ollama_schema(self) -> dict:
        """Formato `tools` de /api/chat de Ollama."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def embed_texts(self) -> list[str]:
        """Textos con los que se compara lo que dices: descripción + ejemplos."""
        return [self.description, *self.examples]


def approx_tokens(obj) -> int:
    """Estimación barata de tokens (JSON compacto en español ~3.2 caracteres por token).

    Solo para presupuestar y probar; el valor real sale de `prompt_eval_count` de Ollama
    (lo mide scripts/bench_tools.py).
    """
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return round(len(text) / 3.2)
