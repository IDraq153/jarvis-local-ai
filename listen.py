"""Oído de JARVIS (Fase 3): micrófono -> texto, 100 % local.
Uso: python listen.py [--model small] [--device cpu] [--once] [--debug]
Di "salir" o pulsa Ctrl+C para terminar.
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

from stt import STTError, load_backend
from stt.metrics import normalize
from stt.recorder import calibrate_threshold, list_input_devices, record_utterance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="tiny, base, small... (sobrescribe stt.model)")
    ap.add_argument("--device", help="cpu o cuda (sobrescribe stt.device)")
    ap.add_argument("--compute-type", help="int8, float16... (sobrescribe stt.compute_type)")
    ap.add_argument("--lang", help="es, en o auto (sobrescribe stt.language)")
    ap.add_argument("--mic", type=int, help="índice del micrófono (ver --list-mics)")
    ap.add_argument("--list-mics", action="store_true", help="lista los micrófonos y termina")
    ap.add_argument("--threshold", type=float, help="umbral de voz (RMS); por defecto se calibra solo")
    ap.add_argument("--once", action="store_true", help="escucha una sola frase y termina")
    ap.add_argument("--debug", action="store_true", help="muestra tiempos, RTF y umbral")
    args = ap.parse_args()

    if args.list_mics:
        try:
            for i, name in list_input_devices():
                print(f"  [{i}] {name}")
        except STTError as e:
            sys.exit(f"Error: {e}")
        return

    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8"))
    stt_cfg = dict(cfg.get("stt") or {})
    for key, val in (("model", args.model), ("device", args.device),
                     ("compute_type", args.compute_type)):
        if val:
            stt_cfg[key] = val
    if args.lang:
        stt_cfg["language"] = None if args.lang == "auto" else args.lang
    mic = args.mic if args.mic is not None else stt_cfg.get("device_index")
    silence_ms = stt_cfg.get("silence_ms", 800)
    threshold = args.threshold or stt_cfg.get("threshold")

    try:
        stt = load_backend(stt_cfg)
        print(f"Cargando {getattr(stt, 'model_name', stt.name)} "
              f"({stt_cfg.get('device', 'cpu')}/{stt_cfg.get('compute_type', 'int8')})...", end=" ", flush=True)
        t0 = time.time()
        stt.load()
        print(f"listo ({time.time() - t0:.1f} s)")
        if not threshold:
            print("Midiendo el ruido ambiente (1 s, no hables)...", end=" ", flush=True)
            threshold = calibrate_threshold(mic)
            print(f"umbral {threshold:.4f}")
        print("Habla cuando quieras. Di \"salir\" o pulsa Ctrl+C para terminar.")
        while True:
            print("\n🎤 escuchando...", end=" ", flush=True)
            audio = record_utterance(mic, threshold=threshold, silence_ms=silence_ms)
            if len(audio) == 0:
                print("(no oí nada)")
                if args.once:
                    return
                continue
            r = stt.transcribe(audio)
            print(f"\r> {r.text or '(vacío)'}")
            if args.debug:
                print(f"  [stt] audio {r.audio_seconds:.1f} s, transcripción {r.compute_seconds:.2f} s, "
                      f"RTF {r.rtf:.2f}, idioma {r.language}")
            if args.once or normalize(r.text) == ["salir"]:
                return
    except STTError as e:
        sys.exit(f"\nError: {e}")
    except KeyboardInterrupt:
        print("\nHasta luego.")


if __name__ == "__main__":
    main()
