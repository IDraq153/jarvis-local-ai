"""XTTS v2 (coqui-tts): clona una voz a partir de un audio de referencia. Corre en voicelab/envs/xtts."""
from pathlib import Path

from .worker_backend import ROOT, WorkerBackend, venv_python


class XttsBackend(WorkerBackend):
    name = "xtts"
    worker_script = ROOT / "voicelab" / "workers" / "xtts_worker.py"

    def __init__(self, ref_audio: str | Path, language: str = "es", device: str = "cuda",
                 temperature: float = 0.7, stream_chunk: int = 20,
                 python: str | Path | None = None):
        ref = Path(ref_audio)
        if not ref.is_absolute():
            ref = ROOT / ref
        super().__init__(python or venv_python("xtts"),
                         ["--ref", ref, "--lang", language, "--device", device,
                          "--temperature", temperature, "--stream-chunk", stream_chunk])
