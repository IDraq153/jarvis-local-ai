"""Chat de JARVIS por terminal (Fase 1). Uso: python chat.py [--model NOMBRE]"""
import argparse
import sys
import time
from pathlib import Path

import yaml

from core.persona import SYSTEM_PROMPT
from llm import LLMError, load_backend

HELP = "Comandos: /reset borra la conversación, /salir termina."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="sobrescribe llm.model de config.yaml")
    ap.add_argument("--debug", action="store_true", help="muestra tiempos y tokens por respuesta")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8"))
    llm_cfg = cfg["llm"]
    if args.model:
        llm_cfg["model"] = args.model
    try:
        llm = load_backend(llm_cfg)
    except LLMError as e:
        sys.exit(f"Error: {e}")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Cargando {llm.model}...", end=" ", flush=True)
    t0 = time.time()
    try:
        llm.warmup()
    except LLMError as e:
        sys.exit(f"\nError: {e}")
    print(f"listo ({time.time() - t0:.1f} s)")
    print(f"JARVIS en línea. {HELP}")
    while True:
        try:
            user = input("\nDario > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/salir":
            break
        if user == "/reset":
            history = history[:1]
            print("Conversación reiniciada.")
            continue
        history.append({"role": "user", "content": user})
        print("JARVIS > ", end="", flush=True)
        reply = ""
        t_start, ttft = time.time(), None
        try:
            for chunk in llm.stream(history):
                if ttft is None:
                    ttft = time.time() - t_start
                print(chunk, end="", flush=True)
                reply += chunk
        except LLMError as e:
            print(f"\n[Error] {e}")
            history.pop()
            continue
        print()
        history.append({"role": "assistant", "content": reply})
        tps = llm.tokens_per_second()
        if tps and ttft is not None:
            print(f"  (primer token: {ttft:.1f} s, {tps:.1f} tok/s)")
        if args.debug:
            st = llm.last_stats
            print(f"  [debug] tokens generados={st.get('eval_count')}, caracteres visibles={len(reply)}, "
                  f"caracteres de 'pensar'={llm.last_thinking_chars}, "
                  f"carga={st.get('load_duration', 0) / 1e9:.2f}s, "
                  f"prompt={st.get('prompt_eval_count')} tok en {st.get('prompt_eval_duration', 0) / 1e9:.2f}s, "
                  f"total en el servidor={st.get('total_duration', 0) / 1e9:.2f}s")
    print("\nHasta luego, Dario.")


if __name__ == "__main__":
    main()
