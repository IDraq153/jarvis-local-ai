"""Resumen de la sesión y extracción de hechos con el propio LLM (al cerrar la sesión)."""
import json
import re

from .store import TurnRec
from .text import looks_secret

MAX_TRANSCRIPT_CHARS = 6000  # num_ctx es 4096 tokens: la charla debe caber con las instrucciones
MAX_FACTS = 5

_INSTRUCTIONS = """Lees una conversación entre Dario y JARVIS y devuelves SOLO un JSON válido, sin texto extra:
{"resumen": "1 o 2 frases sobre de qué se habló", "hechos": ["..."]}
Reglas para "hechos":
- Solo datos duraderos sobre Dario que él mismo dijo: estudios, proyectos, gustos, personas, decisiones, rutinas, preferencias.
- En tercera persona y frases cortas, p. ej. "Dario estudia Ingeniería de Sistemas".
- No incluyas lo que dijo JARVIS, datos pasajeros, contraseñas, números de documento ni de tarjeta.
- Máximo 5. Si no hay nada que valga la pena, usa [].
Responde siempre en español."""


def build_transcript(turns: list[TurnRec]) -> str:
    lines = [f"Dario: {t.user_text[:400]}\nJARVIS: {t.reply_text[:300]}" for t in turns]
    text = "\n".join(lines)
    return text[-MAX_TRANSCRIPT_CHARS:]


def build_messages(turns: list[TurnRec]) -> list[dict]:
    return [{"role": "system", "content": _INSTRUCTIONS},
            {"role": "user", "content": f"Conversación:\n{build_transcript(turns)}\n\nDevuelve el JSON."}]


def parse_extraction(raw: str) -> tuple[str, list[str]]:
    """Extrae (resumen, hechos) de la salida del LLM, tolerando ```json y texto alrededor."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return "", []
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return "", []
    if not isinstance(data, dict):
        return "", []
    summary = data.get("resumen")
    summary = re.sub(r"\s+", " ", summary).strip()[:400] if isinstance(summary, str) else ""
    facts = []
    for f in data.get("hechos") or []:
        if isinstance(f, str):
            f = re.sub(r"\s+", " ", f).strip()
            if 8 <= len(f) <= 200 and not looks_secret(f) and f not in facts:
                facts.append(f)
    return summary, facts[:MAX_FACTS]
