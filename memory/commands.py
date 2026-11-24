"""Órdenes de memoria en lenguaje natural ("recuerda que...", "olvida lo de...", "qué sabes de mí").

Funciona igual con texto escrito y con lo que devuelve el STT (con o sin tildes/puntuación).
"""
from .text import strip_accents

_REMEMBER = [["recuerda", "que"], ["acuerdate", "de", "que"], ["acuerdate", "que"], ["anota", "que"],
             ["apunta", "que"], ["guarda", "en", "memoria", "que"], ["recuerda"]]
_FORGET = [["olvida", "que"], ["olvida", "lo", "de"], ["olvidate", "de"], ["borra", "de", "la", "memoria"],
           ["olvida"]]
_LIST = {"que sabes de mi", "que sabes sobre mi", "que recuerdas de mi", "que recuerdas sobre mi",
         "que has aprendido de mi", "que cosas sabes de mi"}


def _key(word: str) -> str:
    return strip_accents(word).lower().strip(".,;:!?¡¿\"'()")


def parse_memory_command(text: str) -> tuple[str, str] | None:
    """('remember'|'forget'|'list', argumento) o None si no es una orden de memoria."""
    ws = text.split()
    keys = [_key(w) for w in ws]
    if keys and keys[0] == "jarvis":
        ws, keys = ws[1:], keys[1:]
    if not keys:
        return None
    if " ".join(keys) in _LIST:
        return "list", ""
    for kind, prefixes in (("remember", _REMEMBER), ("forget", _FORGET)):
        for p in prefixes:  # los prefijos largos van primero
            if keys[:len(p)] == p:
                arg = " ".join(ws[len(p):]).strip()
                if len(arg.split()) >= (2 if kind == "remember" else 1):
                    return kind, arg
    return None
