"""Prueba rápida de ajustes de una voz de Kokoro: las mismas frases a varias velocidades.

Uso: python -m voicelab.tune [--voice kokoro-am_voz1] [--speeds 0.9 1.0 1.1]
Escucha voicelab/samples/tune/index.html y copia la velocidad que prefieras a tts.speed en config.yaml.
"""
import argparse
import sys
from html import escape

import yaml

from tts import TTSError, load_backend
from tts.playback import save_wav
from voicelab.bench import LAB, SAMPLES, expand_clones

EXTRA = "Entendido, Dario. Esta tarde tienes un laboratorio de redes; si quieres, te preparo un repaso rápido de direccionamiento IP."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="kokoro-am_voz1")
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.9, 1.0, 1.1])
    args = ap.parse_args()

    catalog = yaml.safe_load((LAB / "voices.yaml").read_text(encoding="utf-8"))
    entry = next((v for v in expand_clones(catalog) if v["id"] == args.voice), None)
    if not entry or entry["backend"] != "kokoro":
        sys.exit("--voice debe ser una voz de Kokoro del catálogo (p. ej. kokoro-am_voz1).")
    phrases = [catalog["phrases"][0], catalog["phrases"][1], EXTRA]
    backend = load_backend({k: v for k, v in entry.items() if k not in ("id", "label")})
    out_dir = SAMPLES / "tune"
    rows = []
    try:
        backend.load()
        for speed in args.speeds:
            backend.speed = speed
            players = []
            for i, text in enumerate(phrases, 1):
                audio = backend.synthesize(text)
                rel = f"{args.voice}/v{speed}_{i}.wav"
                save_wav(audio, out_dir / rel)
                players.append(f"<div><small>{escape(text[:70])}…</small>"
                               f"<audio controls preload='none' src='{escape(rel)}'></audio></div>")
            rows.append(f"<tr><td><b>speed {speed}</b></td><td>{''.join(players)}</td></tr>")
    except TTSError as e:
        sys.exit(f"[Error] {e}")
    finally:
        backend.unload()
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Ajuste de voz</title>"
        "<style>body{font:15px system-ui;margin:2rem;background:#111;color:#eee}td{padding:.6rem;vertical-align:top}"
        "audio{display:block;width:280px}small{color:#999}</style>"
        f"<h1>{escape(args.voice)}</h1><table>{''.join(rows)}</table>", encoding="utf-8")
    print(f"Listo: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
