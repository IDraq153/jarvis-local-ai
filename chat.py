"""Chat de JARVIS por terminal (Fase 1). Uso: python chat.py [--model NOMBRE]"""
import argparse
import sys
import time
from pathlib import Path

import yaml

from core import gpu
from core.persona import SYSTEM_PROMPT
from llm import LLMError, ToolCall, load_backend
from tools.runtime import claims_unbacked_write

HELP = ("Comandos: /reset borra la conversación (no la memoria), /memoria lista lo que sé de ti, "
        "/recordar TEXTO, /olvidar N|TEXTO, /fijar N, /salir termina. También: \"recuerda que...\", \"olvida lo de...\".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="sobrescribe llm.model de config.yaml")
    ap.add_argument("--debug", action="store_true", help="muestra tiempos y tokens por respuesta")
    ap.add_argument("--voz", action="store_true", help="lee las respuestas en voz alta (usa la sección tts de config.yaml)")
    ap.add_argument("--sin-memoria", action="store_true", help="no usa ni guarda la memoria persistente (Fase 5)")
    ap.add_argument("--sin-herramientas", action="store_true", help="no usa herramientas (UniGym, Fase 6)")
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
    tools = None
    if not args.sin_herramientas:
        try:
            from tools.runtime import load_tools
            from memory import load_embedder
            embedder = memory.embedder if memory else load_embedder(mcfg.get("embedder"))
            tools = load_tools(cfg, Path(__file__).parent, embedder)
            if tools:
                print("Cargando las herramientas...", end=" ", flush=True)
                t1 = time.time()
                n = tools.selector.index()
                print(f"listas ({n} textos indexados, {time.time() - t1:.1f} s)")
        except Exception as e:  # las herramientas nunca deben impedir hablar con JARVIS
            print(f"\n[herramientas desactivadas] {e}")
            tools = None
    routines = None
    try:
        from tools.runtime import load_routines
        routines = load_routines(cfg, Path(__file__).parent, tools)
    except Exception as e:
        print(f"\n[rutinas desactivadas] {e}")
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
        prefix = ""
        if routines:  # "Jarvis, estoy en casa": rutina fija, no pasa por el LLM
            rr = routines.match(user)
            if rr is not None:
                if tools:
                    tools.before(user)     # si había algo pendiente de confirmar, se descarta
                print(f"JARVIS > {rr.speech}")
                opened = rr.after() if rr.with_speech else None      # el video arranca a la vez que JARVIS habla
                if speaker:
                    speaker.stop()
                    speaker.feed(rr.speech)
                    speaker.finish()
                    speaker.wait_playback()
                if opened is None:
                    opened = rr.after()
                print(f"  (video abierto en {opened})")
                continue
        if tools:  # una escritura pendiente espera "sí"/"no" antes que cualquier otra cosa
            pre = tools.before(user)
            if pre.outcome is not None:
                print(f"JARVIS > {pre.outcome.text}")
                tools.remember_direct(history, user, pre.outcome)
                if speaker:
                    speaker.stop()
                    speaker.feed(pre.outcome.text)
                    speaker.finish()
                continue
            prefix = pre.prefix
        if speaker:
            speaker.stop()
        history.append({"role": "user", "content": user})
        messages = memory.augment(history, user) if memory else history
        schemas = None
        if tools:
            messages, schemas = tools.prepare(messages, user)
        print("JARVIS > ", end="", flush=True)
        reply, call = "", None
        if prefix:
            print(prefix + " ", end="", flush=True)
            reply = prefix + " "
            if speaker:
                speaker.feed(prefix + " ")
        t_start, ttft = time.time(), None
        try:
            events = llm.stream_events(messages, tools=schemas) if schemas else llm.stream(messages)
            for chunk in events:
                if ttft is None:
                    ttft = time.time() - t_start
                if isinstance(chunk, ToolCall):
                    call = call or chunk
                    continue
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
        if call is None and not reply.replace(prefix, "").strip():   # el LLM no dijo nada: mejor pedir que repita
            reply = (reply + "No te entendí bien. ¿Me lo repites?").strip()
            print("No te entendí bien. ¿Me lo repites?", end="")
            if speaker:
                speaker.feed("No te entendí bien. ¿Me lo repites?")
        if call is not None:  # el LLM eligió una herramienta: responde Python (exacto) o pide confirmación
            out = tools.execute(call, user)
            print(out.text)
            if speaker:
                speaker.feed(out.text)
                speaker.finish()
            tools.remember(history, call, out)
            if args.debug:
                print(f"  [herramienta] {call.name} {call.args} -> {out.kind} | ofrecidas: "
                      f"{', '.join(tools.last_selected)} | selección {tools.last_select_s * 1000:.0f} ms | "
                      f"1.er evento {ttft or 0:.2f} s")
            continue
        from tools.runtime import NOT_FROM_DATA, NOT_SAVED, invents_data, claims_unbacked_write, known_context
        if tools and call is None:
            tools.plain_turn()
        warn = NOT_SAVED if (schemas and tools and claims_unbacked_write(reply)) else \
            NOT_FROM_DATA if (tools and invents_data(reply, known_context(user, history))) else ""
        if warn:   # dijo que guardó algo, o dio notas/promedios, sin usar ninguna herramienta
            print(warn, end="")
            reply += warn
            if speaker:
                speaker.feed(warn)
        print()
        if speaker:
            speaker.finish()
        if warn:          # una respuesta falsa no entra al historial: el LLM la imitaría en el turno siguiente
            history.pop()
        else:
            history.append({"role": "assistant", "content": reply})
        if memory and not warn:
            memory.record_turn(user, reply)
        if args.debug and tools:
            print(f"  [herramientas] ofrecidas: {', '.join(tools.last_selected) or 'ninguna'} "
                  f"(selección {tools.last_select_s * 1000:.0f} ms)")
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
