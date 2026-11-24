"""Chat de JARVIS por terminal (Fase 1). Uso: python chat.py [--model NOMBRE]"""
import argparse
import sys
import time
from pathlib import Path

import yaml

from core import gpu
from core.persona import SYSTEM_PROMPT
from llm import LLMError, load_backend

HELP = ("Comandos: /reset borra la conversación (no la memoria), /memoria lista lo que sé de ti, "
        "/recordar TEXTO, /olvidar N|TEXTO, /fijar N, /salir termina. También: \"recuerda que...\", \"olvida lo de...\".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="sobrescribe llm.model de config.yaml")
    ap.add_argument("--debug", action="store_true", help="muestra tiempos y tokens por respuesta")
    ap.add_argument("--voz", action="store_true", help="lee las respuestas en voz alta (usa la sección tts de config.yaml)")
    ap.add_argument("--sin-memoria", action="store_true", help="no usa ni guarda la memoria persistente (Fase 5)")
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
    memory = None
    mcfg = cfg.get("memory") or {}
    if mcfg.get("enabled", True) and not args.sin_memoria:
        from memory import load_memory
        try:
            memory = load_memory(mcfg, Path(__file__).parent)
            st = memory.start_session()
            print("Cargando la memoria...", end=" ", flush=True)
            w = memory.warmup()
            print(f"lista ({w:.1f} s)" if w is not None else "sin embeddings (busco por palabras)")
            print(f"  Memoria: {st['facts']} hechos, {st['turns']} recuerdos, {st['sessions']} sesiones previas")
        except Exception as e:  # la memoria nunca debe impedir hablar con JARVIS
            print(f"\n[memoria desactivada] {e}")
            memory = None
    speaker = None
    if args.voz:
        from tts import TTSError, load_backend as load_tts
        from tts.speaker import Speaker
        try:
            print("Cargando la voz...", end=" ", flush=True)
            t1 = time.time()
            tts_cfg = dict(cfg["tts"])
            min_chars = tts_cfg.pop("min_chars", 25)  # frases más largas = más fluidez, algo más de espera
            voice = load_tts(tts_cfg)
            voice.load()
            list(voice.stream("Listo."))  # calentamiento: la primera síntesis es más lenta
            speaker = Speaker(voice, min_chars=min_chars, on_first_audio=(
                lambda d: print(f"\n  [voz] primer audio {d:.2f} s después del primer texto") if args.debug else None))
        except TTSError as e:
            sys.exit(f"\nError de voz: {e}")
        print(f"lista ({time.time() - t1:.1f} s)")
        used, total = gpu.vram_used_mb(), gpu.vram_total_mb()
        if used is not None and total:
            print(f"  VRAM con LLM y voz cargados: {used} / {total} MB (libres: {total - used} MB)")
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
        if memory:
            if user == "/memoria":
                facts = memory.facts()
                print("\n".join(f"  [{f.id}]{'*' if f.pinned else ' '} {f.text}" for f in facts)
                      + "\n  (* = fijado: se aplica en todos los turnos)" if facts else "  (sin hechos guardados)")
                continue
            if user.startswith("/fijar ") and user[7:].strip().isdigit():
                ok = memory.store.set_pinned(int(user[7:]), True)
                print("Fijado." if ok else "No existe ese número.")
                continue
            if user.startswith("/olvidar ") and user[9:].strip().isdigit():
                print("Olvidado." if memory.forget_id(int(user[9:])) else "No existe ese número.")
                continue
            if user.startswith("/recordar "):
                answer = memory.handle_command("recuerda que " + user[10:].strip())
            elif user.startswith("/olvidar "):
                answer = memory.handle_command("olvida " + user[9:].strip())
            else:
                answer = memory.handle_command(user)
            if answer is not None:
                print(f"JARVIS > {answer}")
                if speaker:
                    speaker.stop()
                    speaker.feed(answer)
                    speaker.finish()
                continue
        if speaker:
            speaker.stop()
        history.append({"role": "user", "content": user})
        messages = memory.augment(history, user) if memory else history
        print("JARVIS > ", end="", flush=True)
        reply = ""
        t_start, ttft = time.time(), None
        try:
            for chunk in llm.stream(messages):
                if ttft is None:
                    ttft = time.time() - t_start
                print(chunk, end="", flush=True)
                if speaker:
                    speaker.feed(chunk)
                reply += chunk
        except LLMError as e:
            print(f"\n[Error] {e}")
            if speaker:
                speaker.stop()
            history.pop()
            continue
        print()
        if speaker:
            speaker.finish()
        history.append({"role": "assistant", "content": reply})
        if memory:
            memory.record_turn(user, reply)
        tps = llm.tokens_per_second()
        if tps and ttft is not None:
            print(f"  (primer token: {ttft:.1f} s, {tps:.1f} tok/s)")
        if args.debug and memory:
            print(f"  [memoria] búsqueda {memory.last_retrieval_s * 1000:.0f} ms, bloque inyectado "
                  f"{len(memory.last_block)} car.")
            if memory.last_block:
                print("    " + memory.last_block.replace("\n", "\n    "))
        if args.debug:
            st = llm.last_stats
            print(f"  [debug] tokens generados={st.get('eval_count')}, caracteres visibles={len(reply)}, "
                  f"caracteres de 'pensar'={llm.last_thinking_chars}, "
                  f"carga={st.get('load_duration', 0) / 1e9:.2f}s, "
                  f"prompt={st.get('prompt_eval_count')} tok en {st.get('prompt_eval_duration', 0) / 1e9:.2f}s, "
                  f"total en el servidor={st.get('total_duration', 0) / 1e9:.2f}s")
    if speaker:
        speaker.close()
    if memory:
        print("\nGuardando la memoria...", end=" ", flush=True)
        res = memory.close_session(llm)
        print(f"{res['nuevos']} hechos nuevos, {res['actualizados']} actualizados." if res else "listo.")
    print("\nHasta luego, Dario.")


if __name__ == "__main__":
    main()
