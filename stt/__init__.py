from .base import SAMPLE_RATE, STTBackend, STTError, Transcript


def load_backend(cfg: dict) -> STTBackend:
    """Crea el backend a partir de la sección `stt` de config.yaml."""
    cfg = dict(cfg)
    backend = cfg.pop("backend", "faster-whisper")
    # Campos que pertenecen al micrófono/endpointing, no al motor.
    for k in ("mic", "silence_ms", "threshold", "device_index"):
        cfg.pop(k, None)
    if backend == "faster-whisper":
        from .faster_whisper_backend import FasterWhisperBackend
        return FasterWhisperBackend(**cfg)
    raise STTError(f"Backend STT desconocido: {backend}")
