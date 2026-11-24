"""Benchmark del oído (Fase 3): graba tus frases y compara modelos de STT.

1) Grabar:    python scripts/bench_stt.py --record          (lee scripts/frases_stt.txt)
2) Comparar:  python scripts/bench_stt.py --models tiny,base,small --device cpu
Resultados en docs/fase3_bench.json y docs/fase3_bench.md. Los .wav quedan en stt/samples/ (no se versionan).
"""
import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stt import SAMPLE_RATE, STTError  # noqa: E402
from stt.faster_whisper_backend import FasterWhisperBackend  # noqa: E402
from stt.metrics import wer  # noqa: E402

SAMPLES = ROOT / "stt" / "samples"
PHRASES = Path(__file__).with_name("frases_stt.txt")


def save_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1, f"{path}: se espera mono 16 kHz"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def record(mic: int | None, only: set[int] | None = None) -> None:
    from stt.recorder import calibrate_threshold, record_utterance
    SAMPLES.mkdir(parents=True, exist_ok=True)
    phrases = [p.strip() for p in PHRASES.read_text(encoding="utf-8").splitlines() if p.strip()]
    print("Midiendo ruido ambiente (1 s, no hables)...")
    thr = calibrate_threshold(mic)
    print(f"Umbral {thr:.4f}. Lee cada frase en voz natural; se corta sola tras un breve silencio.\n")
    for i, text in enumerate(phrases, 1):
        if only and i not in only:
            continue
        while True:
            print(f"[{i}/{len(phrases)}] {text}")
            audio = record_utterance(mic, threshold=thr)
            if len(audio):
                break
            print("  (no oí nada, repite)")
        save_wav(SAMPLES / f"{i:02d}.wav", audio)
        (SAMPLES / f"{i:02d}.txt").write_text(text, encoding="utf-8")
        print(f"  guardado ({len(audio) / SAMPLE_RATE:.1f} s)\n")
    print(f"Listo: {len(phrases)} frases en {SAMPLES}")


def bench(models: list[str], device: str, compute_type: str, lang: str | None, beam: int,
          vad: bool = True, prompt: str | None = None, tag: str = "") -> list[dict]:
    wavs = sorted(SAMPLES.glob("*.wav"))
    if not wavs:
        sys.exit("No hay grabaciones. Primero: python scripts/bench_stt.py --record")
    clips = [(p, load_wav(p), p.with_suffix(".txt").read_text(encoding="utf-8")) for p in wavs]
    proc = psutil.Process()
    results = []
    for model in models:
        be = FasterWhisperBackend(model=model, device=device, compute_type=compute_type,
                                  language=lang, beam_size=beam, vad_filter=vad, initial_prompt=prompt)
        rss0 = proc.memory_info().rss
        t0 = time.perf_counter()
        try:
            be.load()
        except STTError as e:
            print(f"{model}: {e}")
            continue
        load_s = time.perf_counter() - t0
        be.transcribe(clips[0][1])  # calentamiento: la primera inferencia es más lenta
        rows = []
        for p, audio, ref in clips:
            r = be.transcribe(audio)
            rows.append({"file": p.name, "ref": ref, "hyp": r.text, "wer": wer(ref, r.text),
                         "audio_s": r.audio_seconds, "compute_s": r.compute_seconds})
        rss_mb = (proc.memory_info().rss - rss0) / 1e6
        lat = [x["compute_s"] for x in rows]
        res = {"model": model + (f" [{tag}]" if tag else ""), "device": device, "compute_type": compute_type, "load_s": round(load_s, 1),
               "ram_mb": round(rss_mb), "wer": round(statistics.mean(x["wer"] for x in rows), 3),
               "latency_mean_s": round(statistics.mean(lat), 2), "latency_max_s": round(max(lat), 2),
               "rtf": round(sum(lat) / sum(x["audio_s"] for x in rows), 3), "rows": rows}
        results.append(res)
        print(f"{model:8s} WER {res['wer']:.1%}  latencia {res['latency_mean_s']:.2f} s "
              f"(máx {res['latency_max_s']:.2f})  RTF {res['rtf']:.2f}  RAM +{res['ram_mb']} MB  carga {res['load_s']} s")
        be.unload()
    return results


def write_reports(results: list[dict], tag: str = "") -> None:
    docs = ROOT / "docs"
    sfx = f"_{tag}" if tag else ""
    (docs / f"fase3_bench{sfx}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Fase 3: benchmark de STT", "",
             "| Modelo | Dispositivo | WER | Latencia media | Latencia máx | RTF | RAM extra | Carga |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['device']}/{r['compute_type']} | {r['wer']:.1%} | {r['latency_mean_s']} s | "
                     f"{r['latency_max_s']} s | {r['rtf']} | {r['ram_mb']} MB | {r['load_s']} s |")
    lines += ["", "WER = tasa de error de palabras (sin tildes ni puntuación). RTF < 1 = más rápido que tiempo real.", ""]
    for r in results:
        bad = sorted(r["rows"], key=lambda x: -x["wer"])[:3]
        lines.append(f"**{r['model']}: peores frases**")
        for x in bad:
            if x["wer"] > 0:
                lines.append(f"- ({x['wer']:.0%}) «{x['ref']}» → «{x['hyp']}»")
        lines.append("")
    (docs / f"fase3_bench{sfx}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Informe: docs/fase3_bench{sfx}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="graba las frases de scripts/frases_stt.txt")
    ap.add_argument("--mic", type=int, help="índice del micrófono")
    ap.add_argument("--only", help="con --record: regraba solo estas frases, p. ej. 1 o 1,5")
    ap.add_argument("--models", default="tiny,base,small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--lang", default="es")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--no-vad", action="store_true", help="desactiva el filtro VAD de faster-whisper")
    ap.add_argument("--prompt", help="vocabulario inicial (initial_prompt), p. ej. 'Jarvis, Ollama, Qwen'")
    ap.add_argument("--tag", default="", help="etiqueta de la variante (se añade al nombre del informe)")
    args = ap.parse_args()
    if args.record:
        record(args.mic, {int(x) for x in args.only.split(',')} if args.only else None)
        return
    res = bench(args.models.split(","), args.device, args.compute_type,
                None if args.lang == "auto" else args.lang, args.beam,
                 vad=not args.no_vad, prompt=args.prompt, tag=args.tag)
    if res:
        write_reports(res, args.tag)


if __name__ == "__main__":
    main()
