"""Bucle de conversación por voz (Fase 4): micrófono -> STT -> LLM -> TTS, por turnos.

La lógica no conoce el hardware: recibe funciones (`listen`, `transcribe`) y objetos
(`llm`, `speaker`) por inyección, así se prueba sin micrófono, GPU ni modelos.

Anti-eco (half-duplex): el micrófono NO está abierto mientras JARVIS habla. Se abre
de nuevo `tail_ms` después de que termina el último audio (la sala y el buffer de la
tarjeta de sonido aún "suenan"). Como segunda barrera, se descartan transcripciones
que repiten lo que JARVIS acaba de decir.
"""
import time
from dataclasses import dataclass, field

from llm import LLMError
from stt.metrics import normalize

EXIT_PHRASES = {"salir", "adios", "hasta luego", "apagate", "apagate jarvis", "jarvis salir"}
RESET_PHRASES = {"reinicia", "reiniciar", "olvida todo", "nueva conversacion", "jarvis reinicia"}
# Frases que Whisper inventa con casi-silencio o ruido (aun con VAD).
HALLUCINATIONS = {
    "suscribete", "gracias por ver el video", "gracias por ver", "subtitulos",
    "subtitulos por la comunidad de amara org", "amara org", "gracias",
}


def parse_command(text: str) -> str | None:
    """'exit' / 'reset' si toda la frase es una orden de control; si no, None."""
    key = " ".join(normalize(text))
    if key in EXIT_PHRASES:
        return "exit"
    if key in RESET_PHRASES:
        return "reset"
    return None


def is_noise(text: str) -> bool:
    """Transcripción vacía, de una letra o una alucinación típica de Whisper."""
    words = normalize(text)
    if not words:
        return True
    key = " ".join(words)
    return key in HALLUCINATIONS or len(key) < 3


def is_echo(text: str, last_reply: str, min_words: int = 4, ratio: float = 0.8) -> bool:
    """True si casi todas las palabras de `text` ya estaban en la última respuesta de JARVIS."""
    heard = normalize(text)
    if len(heard) < min_words or not last_reply:
        return False
    said = set(normalize(last_reply))
    return sum(w in said for w in heard) / len(heard) >= ratio


def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """Prompt de sistema + los últimos `max_turns` intercambios previos (+ la pregunta pendiente).

    El LLM tiene num_ctx 4096: en una charla larga hay que olvidar lo más antiguo.
    """
    n = 2 * max_turns + (1 if history and history[-1]["role"] == "user" else 0)
    if max_turns <= 0 or len(history) <= 1 + n:
        return history
    tail = history[-n:]
    while tail and tail[0]["role"] != "user":
        tail = tail[1:]
    return history[:1] + tail


@dataclass
class Turn:
    kind: str                    # "reply" | "silence" | "noise" | "echo" | "exit" | "reset" | "error"
    heard: str = ""
    reply: str = ""
    stt_s: float = 0.0           # transcripción
    llm_first_token_s: float = 0.0
    first_audio_s: float = 0.0   # desde el primer texto hasta el primer sonido
    perceived_s: float = 0.0     # fin de tu frase -> primer sonido de JARVIS
    audio_s: float = 0.0         # duración de lo que se grabó
    error: str = ""


@dataclass
class VoiceLoop:
    listen: object               # () -> np.ndarray float32 16 kHz (vacío = no hubo voz)
    transcribe: object           # (np.ndarray) -> Transcript
    llm: object                  # LLMBackend
    speaker: object | None       # Speaker o None (solo texto)
    history: list = field(default_factory=list)
    tail_ms: int = 500
    max_turns: int = 8
    on_text: object = None       # callback(str) con cada trozo del LLM (para imprimir)
    on_heard: object = None      # callback(str) con lo que se te entendió, antes de responder
    sleep: object = time.sleep
    last_reply: str = ""

    def _say(self, text: str) -> None:
        if self.speaker:
            self.speaker.feed(text)
            self.speaker.finish()
            self.speaker.wait()
            self.sleep(self.tail_ms / 1000)

    def run_turn(self) -> Turn:
        audio = self.listen()
        t_end = time.perf_counter()
        if len(audio) == 0:
            return Turn("silence")
        r = self.transcribe(audio)
        stt_s = time.perf_counter() - t_end
        text = (r.text or "").strip()
        base = dict(heard=text, stt_s=stt_s, audio_s=getattr(r, "audio_seconds", 0.0))
        if is_noise(text):
            return Turn("noise", **base)
        if is_echo(text, self.last_reply):
            return Turn("echo", **base)
        cmd = parse_command(text)
        if cmd == "exit":
            return Turn("exit", **base)
        if cmd == "reset":
            del self.history[1:]
            self.last_reply = ""
            self._say("Conversación reiniciada.")
            return Turn("reset", **base)

        if self.on_heard:
            self.on_heard(text)
        self.history.append({"role": "user", "content": text})
        messages = trim_history(self.history, self.max_turns)
        if self.speaker:
            self.speaker.stop()
        reply, ttft, t_llm = "", None, time.perf_counter()
        try:
            for chunk in self.llm.stream(messages):
                if ttft is None:
                    ttft = time.perf_counter() - t_llm
                if self.on_text:
                    self.on_text(chunk)
                if self.speaker:
                    self.speaker.feed(chunk)
                reply += chunk
        except LLMError as e:
            if self.speaker:
                self.speaker.stop()
            self.history.pop()
            return Turn("error", error=str(e), **base)
        if self.speaker:
            self.speaker.finish()
        self.history.append({"role": "assistant", "content": reply})
        self.last_reply = reply
        first_audio = (getattr(self.speaker, "first_audio_delay", None) or 0.0) if self.speaker else 0.0
        if self.speaker:
            self.speaker.wait()          # el micrófono sigue cerrado hasta que JARVIS calle
            self.sleep(self.tail_ms / 1000)
        ttft = ttft or 0.0
        return Turn("reply", reply=reply, llm_first_token_s=ttft, first_audio_s=first_audio,
                    perceived_s=stt_s + ttft + first_audio, **base)
