"""Órdenes de memoria en lenguaje natural ("recuerda que...", "olvida lo de...", "qué sabes de mí").

Funciona igual con texto escrito y con lo que devuelve el STT (con o sin tildes/puntuación).
"""
from .text import strip_accents

_REMEMBER = [["recuerda", "que"], ["acuerdate", "de", "que"], ["acuerdate", "que"], ["anota", "que"],
             ["apunta", "que"], ["guarda", "en", "memoria", "que"], ["recuerda"]]
_FORGET = [["olvida", "que"], ["olvida", "lo", "de"], ["olvidate", "de"], ["borra", "de", "la", "memoria"],
           ["olvida"]]
_NAME = [["quiero", "que", "me", "llames"], ["puedes", "llamarme"], ["ahora", "llamame"], ["llamame"]]
_NAME_FILLER = {"como", "por", "solo", "simplemente", "mejor"}
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
    for p in _NAME:
        if keys[:len(p)] == p:
            rest = [(w.strip(".,;:!?¡¿\"'()"), k) for w, k in zip(ws[len(p):], keys[len(p):])]
            while rest and rest[0][1] in _NAME_FILLER:
                rest = rest[1:]
            if 1 <= len(rest) <= 3:
                return "name", " ".join(w for w, _ in rest)
            break
    for kind, prefixes in (("remember", _REMEMBER), ("forget", _FORGET)):
        for p in prefixes:  # los prefijos largos van primero
            if keys[:len(p)] == p:
                arg = " ".join(ws[len(p):]).strip()
                if len(arg.split()) >= (2 if kind == "remember" else 1):
                    return kind, arg
    return None
