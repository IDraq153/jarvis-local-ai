"""Utilidades de texto para la memoria: normalización, palabras clave y filtro de secretos."""
import re
import unicodedata

STOPWORDS = {
    "los", "las", "del", "con", "por", "para", "una", "uno", "unos", "unas", "que", "como", "mas",
    "pero", "sin", "sobre", "esto", "esta", "este", "eso", "ese", "hay", "son", "era", "muy",
    "mis", "sus", "tus", "nos", "les", "fue", "ser", "has", "han", "the", "and",
}
_SECRET_WORDS = {"contrasena", "contrasenas", "password", "passwd", "apikey"}
_LONG_DIGITS = re.compile(r"\d{8,}")
_CARD_LIKE = re.compile(r"(?:\d{4}[ -]){3}\d{3,4}")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def words(text: str) -> list[str]:
    """Palabras en minúsculas y sin tildes (ñ -> n): 'Contraseña' -> 'contrasena'."""
    return re.findall(r"[a-z0-9]+", strip_accents(text).lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if len(w) > 2 and w not in STOPWORDS]


def looks_secret(text: str) -> bool:
    """Contraseñas, números largos (documentos, cuentas) o de tarjeta: JARVIS no los guarda."""
    if _LONG_DIGITS.search(text) or _CARD_LIKE.search(text):
        return True
    w = words(text)
    return any(x in _SECRET_WORDS for x in w) or ("api" in w and "key" in w)
