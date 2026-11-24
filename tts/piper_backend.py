"""Piper: voces ONNX ligeras que corren en CPU (pip install piper-tts)."""
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import Audio, TTSBackend, TTSError


class PiperBackend(TTSBackend):
    name = "piper"

    def __init__(self, model_path: str | Path, use_cuda: bool = False,
                 length_scale: float | None = None, noise_scale: float | None = None,
                 noise_w_scale: float | None = None, volume: float | None = None):
        self.model_path = Path(model_path)
        self.use_cuda = use_cuda  # requiere onnxruntime-gpu; en CPU Piper ya es rápido
        self._syn = {k: v for k, v in dict(length_scale=length_scale, noise_scale=noise_scale,
                                           noise_w_scale=noise_w_scale, volume=volume).items()
                     if v is not None}
        self._voice = None
        self._syn_config = None

    def load(self) -> None:
        if self._voice is not None:
            return
        if not self.model_path.exists():
            raise TTSError(f"No encuentro la voz {self.model_path}. Descárgala con: "
                           f"python -m piper.download_voices {self.model_path.stem}")
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise TTSError("Falta piper-tts: pip install piper-tts") from e
        self._voice = PiperVoice.load(str(self.model_path), use_cuda=self.use_cuda)
        if self._syn:
            try:
                from piper import SynthesisConfig
            except ImportError as e:
                raise TTSError("Esta versión de piper-tts no expone SynthesisConfig.") from e
            self._syn_config = SynthesisConfig(**self._syn)

    def unload(self) -> None:
        self._voice = None
        self._syn_config = None

    def stream(self, text: str) -> Iterator[Audio]:
        self.load()
        kwargs = {"syn_config": self._syn_config} if self._syn_config else {}
        for chunk in self._voice.synthesize(text, **kwargs):
            pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            yield Audio(pcm.astype(np.float32) / 32768.0, chunk.sample_rate)
