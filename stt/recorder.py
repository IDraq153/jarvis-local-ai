"""Captura del micrófono con detección de fin de frase (endpointing) por energía.

`Endpointer` es lógica pura (se prueba sin micrófono). `record_utterance` la conecta a sounddevice.
La VAD fina (quitar silencios internos) la hace faster-whisper con `vad_filter`.
"""
import queue
import time
from dataclasses import dataclass, field

import numpy as np

from .base import SAMPLE_RATE, STTError

FRAME_MS = 30
FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 480 muestras


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else 0.0


@dataclass
class Endpointer:
    """Decide cuándo empieza y termina una frase a partir de tramas consecutivas.

    - Empieza al detectar `start_frames` tramas seguidas por encima del umbral.
    - Termina tras `silence_ms` de silencio, o al llegar a `max_seconds`.
    - Conserva `preroll_ms` previos al inicio para no cortar la primera sílaba.
    """
    threshold: float = 0.015
    silence_ms: int = 800
    preroll_ms: int = 300
    start_frames: int = 3
    max_seconds: float = 20.0
    min_speech_ms: int = 250

    _pre: list = field(default_factory=list, init=False)
    _speech: list = field(default_factory=list, init=False)
    _loud_run: int = field(default=0, init=False)
    _quiet_ms: int = field(default=0, init=False)
    started: bool = field(default=False, init=False)
    done: bool = field(default=False, init=False)

    def feed(self, frame: np.ndarray) -> bool:
        """Procesa una trama. Devuelve True cuando la frase terminó."""
        if self.done:
            return True
        loud = rms(frame) >= self.threshold
        if not self.started:
            self._pre.append(frame)
            keep = max(self.start_frames, self.preroll_ms // FRAME_MS + self.start_frames)
            if len(self._pre) > keep:
                self._pre.pop(0)
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.start_frames:
                self.started = True
                self._speech = list(self._pre)
                self._quiet_ms = 0
            return False
        self._speech.append(frame)
        self._quiet_ms = 0 if loud else self._quiet_ms + FRAME_MS
        too_long = len(self._speech) * FRAME_MS >= self.max_seconds * 1000
        if self._quiet_ms >= self.silence_ms or too_long:
            self.done = True
        return self.done

    def audio(self) -> np.ndarray:
        """Audio de la frase (vacío si fue demasiado corta para ser voz)."""
        if not self._speech:
            return np.zeros(0, dtype=np.float32)
        voiced_ms = (len(self._speech) * FRAME_MS) - self._quiet_ms
        if voiced_ms < self.min_speech_ms:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._speech).astype(np.float32)


def list_input_devices() -> list[tuple[int, str]]:
    try:
        import sounddevice as sd
    except ImportError as e:
        raise STTError("Falta sounddevice: pip install -r requirements-stt.txt") from e
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def calibrate_threshold(device: int | None = None, seconds: float = 1.0, factor: float = 3.0,
                        floor: float = 0.008) -> float:
    """Mide el ruido ambiente y devuelve un umbral = factor × ruido (con un mínimo)."""
    frames = _grab(seconds, device)
    noise = float(np.median([rms(f) for f in frames])) if frames else 0.0
    return max(floor, noise * factor)


def _grab(seconds: float, device: int | None) -> list[np.ndarray]:
    import sounddevice as sd
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    out, deadline = [], time.monotonic() + seconds
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=FRAME, device=device, callback=cb):
        while time.monotonic() < deadline:
            try:
                out.append(q.get(timeout=0.5))
            except queue.Empty:
                break
    return out


def record_utterance(device: int | None = None, threshold: float = 0.015, silence_ms: int = 800,
                     max_seconds: float = 20.0, wait_seconds: float | None = None) -> np.ndarray:
    """Espera a que hables y devuelve tu frase (float32, 16 kHz). Vacío si no hubo voz."""
    try:
        import sounddevice as sd
    except ImportError as e:
        raise STTError("Falta sounddevice: pip install -r requirements-stt.txt") from e
    ep = Endpointer(threshold=threshold, silence_ms=silence_ms, max_seconds=max_seconds)
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    t0 = time.monotonic()
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=FRAME, device=device, callback=cb):
            while True:
                try:
                    frame = q.get(timeout=0.5)
                except queue.Empty:
                    raise STTError("El micrófono no entrega audio.")
                if ep.feed(frame):
                    break
                if wait_seconds is not None and not ep.started and time.monotonic() - t0 > wait_seconds:
                    break
    except sd.PortAudioError as e:
        raise STTError(f"No se pudo abrir el micrófono: {e}") from e
    return ep.audio()
