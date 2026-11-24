"""Reproducir y guardar audio."""
import wave
from pathlib import Path

import numpy as np

from .base import Audio, TTSError


def save_wav(audio: Audio, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio.samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes(pcm.tobytes())


def play(audio: Audio, block: bool = True) -> None:
    try:
        import sounddevice as sd
    except ImportError as e:
        raise TTSError("Falta sounddevice: pip install sounddevice") from e
    sd.play(audio.samples, audio.sample_rate)
    if block:
        sd.wait()
