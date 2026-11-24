from pathlib import Path

from .base import Audio, TTSBackend, TTSError

ROOT = Path(__file__).resolve().parent.parent


def load_backend(cfg: dict) -> TTSBackend:
    """Crea el backend a partir de la sección `tts` de config.yaml (o de una voz de voicelab)."""
    cfg = dict(cfg)
    backend = cfg.pop("backend", "piper")
    if backend == "piper":
        from .piper_backend import PiperBackend
        voice = cfg.pop("voice", None)
        model = cfg.pop("model", None)
        models_dir = cfg.pop("models_dir", "voicelab/models/piper")
        if not model:
            if not voice:
                raise TTSError("Falta tts.voice (p. ej. es_MX-claude-high) en config.yaml.")
            model = Path(models_dir) / f"{voice}.onnx"
        model = Path(model)
        if not model.is_absolute():
            model = ROOT / model
        return PiperBackend(model_path=model, **cfg)
    if backend == "kokoro":
        from .kokoro_backend import KokoroBackend
        cfg.pop("models_dir", None)
        voice = cfg.get("voice", "")
        if str(voice).endswith(".pt") and not Path(voice).is_absolute():
            cfg["voice"] = str(ROOT / voice)  # pack de voz propio (p. ej. creado con inno-kokoro)
        return KokoroBackend(**cfg)
    if backend == "xtts":
        from .xtts_backend import XttsBackend
        cfg.pop("models_dir", None)
        cfg.pop("voice", None)
        return XttsBackend(**cfg)
    raise TTSError(f"Backend TTS desconocido: {backend}")
