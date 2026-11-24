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

from llm import LLMError, ToolCall
from tools.runtime import NOT_FROM_DATA, NOT_SAVED, claims_unbacked_write, invents_data, known_context
from stt.metrics import normalize

EXIT_PHRASES = {"salir", "adios", "hasta luego", "apagate", "apagate jarvis", "jarvis salir"}
RESET_PHRASES = {"reinicia", "reiniciar", "olvida todo", "nueva conversacion", "jarvis reinicia"}
# Frases que Whisper inventa con casi-silencio o ruido (aun con VAD).
HALLUCINATIONS = {
    "suscribete", "gracias por ver el video", "gracias por ver", "subtitulos",
    "subtitulos por la comunidad de amara org", "amara org", "gracias",
}
EMPTY_REPLY = "No te entendí bien. ¿Me lo repites?"
SHORT_ANSWERS = {"si", "no"}


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
    if key in SHORT_ANSWERS:   # "sí" / "no" son respuestas válidas (confirmar o cancelar una escritura)
        return False
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
    kind: str                    # "reply" | "memory" | "tool" | "silence" | "noise" | "echo" | "exit" | "reset" | "error"
    heard: str = ""
    reply: str = ""
    stt_s: float = 0.0           # transcripción
    llm_first_token_s: float = 0.0
    first_audio_s: float = 0.0   # desde el primer texto hasta el primer sonido
    perceived_s: float = 0.0     # fin de tu frase -> primer sonido de JARVIS
    audio_s: float = 0.0         # duración de lo que se grabó
    error: str = ""
    tool: str = ""               # herramienta usada ("" si ninguna); kind == "tool"
    tool_select_s: float = 0.0   # tiempo de elegir herramientas con embeddings


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
    memory: object | None = None  # memory.Memory (Fase 5) o None: sin memoria a largo plazo
    tools: object | None = None   # tools.runtime.ToolRuntime (Fase 6) o None: sin herramientas
    routines: object | None = None  # tools.routines.RoutineRunner ("estoy en casa") o None
    last_reply: str = ""

    def _say(self, text: str) -> None:
        if self.speaker:
            self.speaker.feed(text)
            self.speaker.finish()
            self.speaker.wait()
            getattr(self.speaker, "wait_playback", lambda: None)()   # que termine de SONAR, no solo de encolarse
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
        prefix = ""
        if self.tools:  # ¿hay una escritura esperando "sí"? Se resuelve antes que todo lo demás
            pre = self.tools.before(text)
            if pre.outcome is not None:
                return self._tool_turn(text, pre.outcome, base)
            prefix = pre.prefix
        if self.routines:  # "Jarvis, estoy en casa": rutina fija, no pasa por el LLM
            rr = self.routines.match(text)
            if rr is not None:
                speech = (prefix + " " + rr.speech).strip()
                if self.on_text:
                    self.on_text(speech)
                self.last_reply = speech
                if rr.with_speech:
                    rr.after()             # el video arranca a la vez que JARVIS habla
                self._say(speech)
                if not rr.with_speech:
                    rr.after()             # o, si así se configura, cuando JARVIS terminó de hablar
                return Turn("tool", reply=speech, tool=rr.tool, **base)
        if self.memory:  # "recuerda que...", "olvida lo de...", "qué sabes de mí": no pasan por el LLM
            answer = self.memory.handle_command(text)
            if answer is not None:
                answer = (prefix + " " + answer).strip()
                if self.on_text:
                    self.on_text(answer)
                self._say(answer)
                return Turn("memory", reply=answer, **base)
        self.history.append({"role": "user", "content": text})
        messages = trim_history(self.history, self.max_turns)
        if self.memory:  # añade al prompt de sistema lo recordado (el historial no se toca)
            messages = self.memory.augment(messages, text)
        schemas, select_s = None, 0.0
        if self.tools:  # elige las herramientas más parecidas y añade la guía al prompt de sistema
            messages, schemas = self.tools.prepare(messages, text)
            select_s = self.tools.last_select_s
        if self.speaker:
            self.speaker.stop()
        reply, ttft, t_llm, call = "", None, time.perf_counter(), None
        if prefix:  # "Descarté lo que estaba pendiente." antes de atender la orden nueva
            if self.on_text:
                self.on_text(prefix + " ")
            if self.speaker:
                self.speaker.feed(prefix + " ")
            reply = prefix + " "
        try:
            events = self.llm.stream_events(messages, tools=schemas) if schemas else self.llm.stream(messages)
            for chunk in events:
                if ttft is None:
                    ttft = time.perf_counter() - t_llm
                if isinstance(chunk, ToolCall):
                    call = call or chunk  # una herramienta por turno
                    continue
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
        if call is None and not reply.replace(prefix, "").strip():   # el LLM no dijo nada: mejor pedir que repita
            reply = (reply + EMPTY_REPLY).strip()
            if self.on_text:
                self.on_text(EMPTY_REPLY)
            if self.speaker:
                self.speaker.feed(EMPTY_REPLY)
        if call is not None:  # el LLM eligió una herramienta: Python responde (exacto) o pide confirmación
            out = self.tools.execute(call, text)
            say = out.text  # si había prefijo, ya se dijo antes
            if self.on_text:
                self.on_text(("\n" if reply.strip() and not prefix else "") + say)
            if self.speaker:
                self.speaker.feed(say)
                self.speaker.finish()
            reply = (prefix + " " + say).strip() if prefix else say
            self.tools.remember(self.history, call, out)   # ver ToolRuntime.remember: no se guardan preguntas ni confirmaciones
            self.last_reply = reply
            if self.speaker:
                self.speaker.wait()
                self.sleep(self.tail_ms / 1000)
            first_audio = (getattr(self.speaker, "first_audio_delay", None) or 0.0) if self.speaker else 0.0
            ttft = ttft or 0.0
            return Turn("tool", reply=reply, llm_first_token_s=ttft, first_audio_s=first_audio, tool=call.name,
                        tool_select_s=select_s, perceived_s=stt_s + select_s + ttft + first_audio, **base)
        if self.tools and call is None:
            self.tools.plain_turn()
        warn = NOT_SAVED if (schemas and self.tools and claims_unbacked_write(reply)) else \
            NOT_FROM_DATA if (self.tools and invents_data(reply, known_context(text, self.history))) else ""
        if warn:   # dijo que guardó algo, o dio notas/promedios, sin usar ninguna herramienta
            reply += warn
            if self.on_text:
                self.on_text(warn)
            if self.speaker:
                self.speaker.feed(warn)
        if self.speaker:
            self.speaker.finish()
        if warn:          # una respuesta falsa no entra al historial: el LLM la imitaría en el turno siguiente
            self.history.pop()
        else:
            self.history.append({"role": "assistant", "content": reply})
        self.last_reply = reply
        if self.memory and not warn:  # se guarda mientras JARVIS aún habla: no suma latencia
            self.memory.record_turn(text, reply)
        first_audio = (getattr(self.speaker, "first_audio_delay", None) or 0.0) if self.speaker else 0.0
        if self.speaker:
            self.speaker.wait()          # el micrófono sigue cerrado hasta que JARVIS calle
            self.sleep(self.tail_ms / 1000)
        ttft = ttft or 0.0
        return Turn("reply", reply=reply, llm_first_token_s=ttft, first_audio_s=first_audio,
                    tool_select_s=select_s, perceived_s=stt_s + select_s + ttft + first_audio, **base)

    def _tool_turn(self, text: str, out, base: dict) -> Turn:
        """Respuesta a una confirmación pendiente ("sí"/"no"): no pasa por el LLM."""
        if self.on_text:
            self.on_text(out.text)
        self.last_reply = out.text      # preguntas/confirmaciones no entran al historial; lecturas sí (formato nativo)
        if self.tools:
            self.tools.remember_direct(self.history, text, out)
        self._say(out.text)
        return Turn("tool", reply=out.text, tool=out.tool, **base)
