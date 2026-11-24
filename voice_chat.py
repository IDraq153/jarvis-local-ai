"""Conversación por voz con JARVIS (Fase 4): hablas, él te escucha, piensa y te responde en voz alta.

Uso: python voice_chat.py [--debug] [--sin-voz] [--stt-model base] [--mic N]
Di "salir" (o Ctrl+C) para terminar, "reinicia" para borrar la conversación.
Mientras JARVIS habla el micrófono está cerrado (anti-eco); espera a que termine.
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

from core import gpu
from core.persona import SYSTEM_PROMPT
from core.voice_loop import VoiceLoop
from llm import LLMError, load_backend as load_llm
from stt import STTError, load_backend as load_stt
from stt.recorder import calibrate_threshold, record_utterance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="sobrescribe llm.model")
    ap.add_argument("--stt-model", help="tiny/base/small (sobrescribe stt.model)")
    ap.add_argument("--mic", type=int, help="índice del micrófono (python listen.py --list-mics)")
    ap.add_argument("--threshold", type=float, help="umbral de voz (RMS); por defecto se calibra")
    ap.add_argument("--sin-voz", action="store_true", help="no sintetiza voz (solo escucha y escribe)")
    ap.add_argument("--debug", action="store_true", help="muestra la latencia de cada turno")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8"))
    vcfg = cfg.get("voice") or {}
    stt_cfg = dict(cfg["stt"])
    if args.stt_model:
        stt_cfg["model"] = args.stt_model
    if args.model:
        cfg["llm"]["model"] = args.model
    mic = args.mic if args.mic is not None else stt_cfg.get("device_index")
    silence_ms = stt_cfg.get("silence_ms", 800)
    threshold = args.threshold or stt_cfg.get("threshold")

    try:
        llm = load_llm(cfg["llm"])
        print(f"Cargando {llm.model}...", end=" ", flush=True)
        t0 = time.time()
        llm.warmup()
        print(f"listo ({time.time() - t0:.1f} s)")

        stt = load_stt(stt_cfg)
        print(f"Cargando el oído ({stt_cfg['model']}, {stt_cfg['device']})...", end=" ", flush=True)
        t0 = time.time()
        stt.load()
        print(f"listo ({time.time() - t0:.1f} s)")

        speaker = None
        if not args.sin_voz:
            from tts import load_backend as load_tts
            from tts.speaker import Speaker
            print("Cargando la voz...", end=" ", flush=True)
            t0 = time.time()
            tts_cfg = dict(cfg["tts"])
            min_chars = tts_cfg.pop("min_chars", 25)
            voice = load_tts(tts_cfg)
            voice.load()
            list(voice.stream("Listo."))  # calentamiento
            speaker = Speaker(voice, min_chars=min_chars)
            print(f"lista ({time.time() - t0:.1f} s)")
            used, total = gpu.vram_used_mb(), gpu.vram_total_mb()
            if used is not None and total:
                print(f"  VRAM con LLM y voz: {used} / {total} MB (libres: {total - used} MB)")

        if not threshold:
            print("Midiendo el ruido ambiente (1 s, no hables)...", end=" ", flush=True)
            threshold = calibrate_threshold(mic)
            print(f"umbral {threshold:.4f}")
    except (LLMError, STTError) as e:
        sys.exit(f"\nError: {e}")
    except Exception as e:  # TTSError u otros al cargar
        sys.exit(f"\nError: {e}")

    def listen():
        print("\n🎤 escuchando...", end=" ", flush=True)
        return record_utterance(mic, threshold=threshold, silence_ms=silence_ms)

    def on_heard(text):
        print(f"\r> {text}\nJARVIS > ", end="", flush=True)

    def on_text(chunk):
        print(chunk, end="", flush=True)

    loop = VoiceLoop(
        listen=listen, transcribe=stt.transcribe, llm=llm, speaker=speaker,
        history=[{"role": "system", "content": SYSTEM_PROMPT}], on_heard=on_heard,
        tail_ms=vcfg.get("tail_ms", 500), max_turns=vcfg.get("max_turns", 8), on_text=on_text)

    print('JARVIS en línea. Habla cuando quieras ("salir" termina, "reinicia" borra la conversación).')
    try:
        while True:
            turn = loop.run_turn()
            if turn.kind == "silence":
                print("(no oí nada)")
            elif turn.kind == "noise":
                print("(ruido, ignorado)")
            elif turn.kind == "echo":
                print(f"\r(eco de JARVIS, ignorado: {turn.heard})")
            elif turn.kind == "exit":
                break
            elif turn.kind == "reset":
                print(f"\r> {turn.heard}\nConversación reiniciada.")
            elif turn.kind == "error":
                print(f"\r> {turn.heard}\n[Error] {turn.error}")
            elif turn.kind == "reply":
                print()
                if args.debug:
                    print(f"  [turno] oído {turn.audio_s:.1f} s | STT {turn.stt_s:.2f} s | "
                          f"1.er token {turn.llm_first_token_s:.2f} s | 1.er audio +{turn.first_audio_s:.2f} s | "
                          f"percibido {turn.perceived_s:.2f} s")
    except KeyboardInterrupt:
        pass
    except STTError as e:
        print(f"\nError: {e}")
    if speaker:
        speaker.close()
    print("\nHasta luego, Dario.")


if __name__ == "__main__":
    main()
