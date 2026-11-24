"""Motores que viven en su propio venv y hablan con JARVIS por una tubería (JSON por líneas).

Así XTTS, F5-TTS, etc. pueden tener sus propias versiones de torch/transformers sin romper
el venv principal. Protocolo:
  worker -> JARVIS: {"ready": true, ...} al terminar de cargar, o {"error": "..."}
  JARVIS -> worker: {"text": "una frase"}  /  {"cmd": "quit"}
  worker -> JARVIS: {"ok": true, "path": "x.wav"}  o  {"error": "..."}
  (streaming: varios {"ok": true, "path": ..., "last": false} y al final {"ok": true, "last": true})
"""
import json
import os
import subprocess
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import Audio, TTSBackend, TTSError
from .text import split_sentences

ROOT = Path(__file__).resolve().parent.parent


def venv_python(env_name: str) -> Path:
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return ROOT / "voicelab" / "envs" / env_name / sub


def read_wav(path: Path) -> Audio:
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return Audio(pcm.astype(np.float32) / 32768.0, w.getframerate())


class WorkerBackend(TTSBackend):
    worker_script: Path  # lo define cada motor

    def __init__(self, python: str | Path, worker_args: list[str] | None = None, max_chars: int = 200):
        self.python = Path(python)
        self.worker_args = [str(a) for a in (worker_args or [])]
        self.max_chars = max_chars
        self._proc: subprocess.Popen | None = None

    def load(self) -> None:
        if self._proc is not None:
            return
        if not self.python.exists():
            raise TTSError(f"No encuentro el Python del motor: {self.python}. "
                           f"Crea su venv (ver docs/fase2.md).")
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        self._proc = subprocess.Popen(
            [str(self.python), str(self.worker_script), *self.worker_args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            bufsize=1, env=env)  # stderr heredado: así ves descargas y errores en tu terminal
        msg = self._read()
        if msg.get("error"):
            self.unload()
            raise TTSError(msg["error"])

    def _read(self) -> dict:
        while True:
            line = self._proc.stdout.readline()
            if not line:
                code = self._proc.poll()
                self._proc = None
                raise TTSError(f"El worker de {self.name} terminó (código {code}). "
                               f"Mira el error impreso arriba en la terminal.")
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue

    def _send(self, obj: dict) -> None:
        try:
            self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except OSError as e:
            self._proc = None
            raise TTSError(f"Se cayó el worker de {self.name}.") from e

    def unload(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            p.stdin.write('{"cmd": "quit"}\n')
            p.stdin.flush()
            p.wait(timeout=15)
        except Exception:
            p.kill()
        finally:
            for f in (p.stdin, p.stdout):
                try:
                    f.close()
                except Exception:
                    pass

    def stream(self, text: str) -> Iterator[Audio]:
        self.load()
        for sentence in split_sentences(text, self.max_chars):
            self._send({"text": sentence})
            while True:  # el worker puede mandar varios trozos por frase (streaming)
                r = self._read()
                if r.get("error"):
                    raise TTSError(r["error"])
                if r.get("path"):
                    path = Path(r["path"])
                    try:
                        yield read_wav(path)
                    finally:
                        path.unlink(missing_ok=True)
                if r.get("last", True):
                    break
