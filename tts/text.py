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


def clean_for_speech(text: str) -> str:
    """Quita lo que no se debe leer en voz alta (markdown, bloques de código, emojis)."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = _EMOJI.sub("", text)
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
