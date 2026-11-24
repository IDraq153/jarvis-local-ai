"""Banco de pruebas de voces: latencia, factor de tiempo real y VRAM, más muestras .wav.

Uso (desde la raíz del repo, con el venv activo):
    python -m voicelab.bench                       # todas las voces del catálogo
    python -m voicelab.bench --voice piper-es_MX-claude-high --runs 3
Los resultados se fusionan en voicelab/results.json y se regenera voicelab/samples/index.html.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from tts import TTSError, load_backend
from tts.playback import save_wav
from voicelab.report import build_index

LAB = Path(__file__).resolve().parent
SAMPLES = LAB / "samples"
RESULTS = LAB / "results.json"


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class VramPeak:
    """Sondea nvidia-smi en segundo plano y guarda el máximo de VRAM usada."""

    def __init__(self, every: float = 0.1):
        self.every, self.peak, self._stop = every, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            v = vram_used_mb()
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v
            self._stop.wait(self.every)

    def __enter__(self):
        self.peak = vram_used_mb()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def time_phrase(backend, text: str):
    """Devuelve (primer_audio_s, total_s, audio) para una frase."""
    t0 = time.perf_counter()
    first, chunks = None, []
    for chunk in backend.stream(text):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(chunk)
    total = time.perf_counter() - t0
    if not chunks:
        raise TTSError("Sin audio.")
    import numpy as np
    from tts.base import Audio
    return first, total, Audio(np.concatenate([c.samples for c in chunks]), chunks[0].sample_rate)


def bench_voice(entry: dict, phrases: list[str], runs: int) -> dict:
    cfg = {k: v for k, v in entry.items() if k not in ("id", "label")}
    backend = load_backend(cfg)
    baseline = vram_used_mb()
    res = {"label": entry.get("label", entry["id"]), "backend": entry["backend"],
           "date": datetime.now().isoformat(timespec="seconds"), "vram_baseline_mb": baseline, "phrases": []}
    with VramPeak() as vp:
        t0 = time.perf_counter()
        backend.load()
        res["load_s"] = round(time.perf_counter() - t0, 2)
        after_load = vram_used_mb()
        time_phrase(backend, phrases[0])  # calentamiento, no se cuenta
        for i, text in enumerate(phrases, 1):
            firsts, totals, audio = [], [], None
            for _ in range(runs):
                f, t, audio = time_phrase(backend, text)
                firsts.append(f)
                totals.append(t)
            wav = SAMPLES / entry["id"] / f"{i}.wav"
            save_wav(audio, wav)
            synth = statistics.mean(totals)
            res["phrases"].append({
                "text": text, "first_audio_s": round(statistics.mean(firsts), 3),
                "synth_s": round(synth, 3), "audio_s": round(audio.seconds, 2),
                "rtf": round(synth / audio.seconds, 3), "wav": f"{entry['id']}/{i}.wav"})
    res["vram_after_load_mb"] = after_load
    res["vram_peak_mb"] = vp.peak
    res["vram_delta_mb"] = (vp.peak - baseline) if (vp.peak is not None and baseline is not None) else None
    backend.unload()
    return res


AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def expand_clones(catalog: dict) -> list[dict]:
    """Voces fijas del catálogo + una voz clonada por cada audio de voicelab/refs/ y motor de `clones`."""
    voices = list(catalog["voices"])
    refs = LAB / "refs"
    for c in catalog.get("clones") or []:
        for f in sorted(refs.glob("*")) if refs.exists() else []:
            if f.suffix.lower() in AUDIO_EXT:
                voices.append({**c, "id": f"{c['backend']}-{f.stem}",
                               "label": f"{c['backend'].upper()} · {f.stem}",
                               "ref_audio": f"voicelab/refs/{f.name}"})
    packs = LAB / "packs"
    for f in sorted(packs.glob("*.pt")) if packs.exists() else []:
        voices.append({"id": f"kokoro-{f.stem}", "label": f"Kokoro pack · {f.stem}", "backend": "kokoro",
                       "voice": f"voicelab/packs/{f.name}", "device": "cuda"})
    return voices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="append", help="id de voz del catálogo (repetible); por defecto todas")
    ap.add_argument("--runs", type=int, default=2, help="repeticiones por frase (se promedian)")
    args = ap.parse_args()

    catalog = yaml.safe_load((LAB / "voices.yaml").read_text(encoding="utf-8"))
    all_voices = expand_clones(catalog)
    voices = [v for v in all_voices if not args.voice or v["id"] in args.voice]
    if not voices:
        sys.exit("Ninguna voz coincide con --voice. Ids: " + ", ".join(v["id"] for v in all_voices))

    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    print(f"VRAM usada antes de empezar: {vram_used_mb()} MB (esto incluye el LLM si está cargado)")
    for v in voices:
        print(f"\n== {v['id']} ==")
        try:
            r = bench_voice(v, catalog["phrases"], args.runs)
        except TTSError as e:
            print(f"  [Error] {e}")
            continue
        results[v["id"]] = r
        print(f"  carga {r['load_s']} s | VRAM +{r['vram_delta_mb']} MB (pico {r['vram_peak_mb']} MB)")
        for i, p in enumerate(r["phrases"], 1):
            print(f"  frase {i}: primer audio {p['first_audio_s']} s | síntesis {p['synth_s']} s | "
                  f"audio {p['audio_s']} s | RTF {p['rtf']}")
        RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    build_index(results, SAMPLES)
    print(f"\nMuestras: {SAMPLES / 'index.html'}  (ábrelo en el navegador)")


if __name__ == "__main__":
    main()
