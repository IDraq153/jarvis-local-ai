"""Utilidades de texto para TTS."""
import re

_SENT = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    """Parte el texto en frases; las frases largas se cortan en comas o espacios."""
    parts: list[str] = []
    for s in _SENT.split(text.strip()):
        s = s.strip()
        while len(s) > max_chars:
            cut = max(s.rfind(", ", 0, max_chars), s.rfind("; ", 0, max_chars))
            if cut <= 0:
                cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars - 1
            parts.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            parts.append(s)
    return parts


_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_MARKDOWN = re.compile(r"[*_#`>~|]+")
_BOUNDARY = re.compile(r"(?<=[.!?…])\s|\n")


_UNITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
          "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno",
          "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"]
_TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta"}


def _words(n: int) -> str:
    """0-59 en palabras."""
    if n < 30:
        return _UNITS[n]
    t, u = divmod(n, 10)
    return _TENS[t] + (f" y {_UNITS[u]}" if u else "")


def _period(h: int) -> str:
    if h == 12:
        return "del día"
    if h < 6:
        return "de la madrugada"
    if h < 12:
        return "de la mañana"
    if h < 19:
        return "de la tarde"
    return "de la noche"


def spoken_time(h: int, m: int, with_period: bool = True) -> str:
    """06:30 -> «seis y media de la mañana»; 18:00 -> «seis de la tarde»; 07:45 -> «ocho menos cuarto de la mañana»."""
    shown = h
    if m == 45:
        shown = (h + 1) % 24
    h12 = shown % 12 or 12
    hour = "una" if h12 == 1 else _words(h12)
    if m == 0:
        out = hour
    elif m == 15:
        out = f"{hour} y cuarto"
    elif m == 30:
        out = f"{hour} y media"
    elif m == 45:
        out = f"{hour} menos cuarto"
    else:
        out = f"{hour} y {_words(m)}"
    if h == 0 and m == 0:
        return "doce de la noche" if with_period else "doce"
    return f"{out} {_period(shown if m == 45 else h)}" if with_period else out


_TIME = r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])"
_RANGE = re.compile(_TIME + r"\s*(?:a|-|–|hasta)\s*" + _TIME.replace("(?<![\\d:])", "(?<![\\d:])"))
_ONE = re.compile(_TIME)


def speak_times(text: str) -> str:
    """Escribe en palabras las horas «HH:MM» para que la voz diga «seis y media de la mañana» y no «cero seis treinta».

    Un rango «07:00 a 09:00» se dice «siete a nueve de la mañana» (el periodo una sola vez si es el mismo).
    """
    def rng(m):
        h1, m1, h2, m2 = (int(g) for g in m.groups())
        if _period(h1) == _period(h2 if m2 != 45 else (h2 + 1) % 24) and not (h1 == 0 and m1 == 0):
            return f"{spoken_time(h1, m1, False)} a {spoken_time(h2, m2)}"
        return f"{spoken_time(h1, m1)} a {spoken_time(h2, m2)}"

    def one(m):
        return spoken_time(int(m.group(1)), int(m.group(2)))
    out = _ONE.sub(one, _RANGE.sub(rng, text))
    return re.sub(r"\b([Ll])as una\b", r"\1a una", out)          # «a las una» -> «a la una»


_ONES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece",
         "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós",
         "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"]
_TENS_ALL = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_HUNDREDS = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos", 6: "seiscientos",
             7: "setecientos", 8: "ochocientos", 9: "novecientos"}


def number_words(n: int) -> str:
    """0-999 en palabras (para decimales hablados)."""
    if n < 30:
        return _ONES[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _TENS_ALL[t] + (f" y {_ONES[u]}" if u else "")
    if n == 100:
        return "cien"
    h, r = divmod(n, 100)
    return _HUNDREDS[h] + (f" {number_words(r)}" if r else "")


_DECIMAL = re.compile(r"(?<![\d.,])(\d{1,3})[,.](\d{1,2})(?![\d.,]*\d)")


def speak_decimals(text: str) -> str:
    """«17,4» -> «diecisiete punto cuatro»; «17,61» -> «diecisiete punto sesenta y uno». No toca IPs ni versiones largas."""
    def one(m):
        dec = m.group(2)
        dec_words = ("cero " + number_words(int(dec[1]))) if dec.startswith("0") and len(dec) == 2 else number_words(int(dec))
        return f"{number_words(int(m.group(1)))} punto {dec_words}"
    return _DECIMAL.sub(one, text)


def clean_for_speech(text: str) -> str:
    """Quita lo que no se debe leer en voz alta (markdown, bloques de código, emojis) y dice las horas en palabras."""
    text = speak_times(text)
    text = speak_decimals(text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = _EMOJI.sub("", text)
    text = re.sub(r"(?m)^\s*(?:[-•*]|\d+[.)])\s+", "", text)   # viñetas y listas numeradas no se leen
    text = _MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceBuffer:
    """Acumula texto que llega en trozos (tokens del LLM) y devuelve frases completas.

    No corta en "192.168.1.10" ni en "3.5": exige un espacio o salto de línea tras el punto.
    Las frases muy cortas se juntan con la siguiente para que la voz suene natural.
    """

    def __init__(self, min_chars: int = 25):
        self.min_chars = min_chars
        self._buf = ""

    def push(self, chunk: str) -> list[str]:
        self._buf += chunk
        out, pos = [], 0
        while True:
            m = _BOUNDARY.search(self._buf, pos)
            if not m:
                break
            candidate = self._buf[:m.start() + (0 if m.group() == "\n" else 1)]
            if len(candidate.strip()) >= self.min_chars:
                out.append(candidate.strip())
                self._buf = self._buf[m.end():]
                pos = 0
            else:
                pos = m.end()
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []
