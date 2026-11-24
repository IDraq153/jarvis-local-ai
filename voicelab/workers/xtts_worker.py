r"""Worker de XTTS v2. Se ejecuta con el Python de voicelab/envs/xtts (no con el venv principal).

  python voicelab\workers\xtts_worker.py --download     # una vez, a mano: descarga el modelo
Lo lanza tts.xtts_backend.XttsBackend con el protocolo de tts/worker_backend.py.
"""
import argparse
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

SAMPLE_RATE = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="audio de referencia de la voz a clonar")
    ap.add_argument("--lang", default="es")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--stream-chunk", type=int, default=20,
                    help="tokens por trozo en streaming (menor = primer audio antes); 0 = sin streaming")
    ap.add_argument("--download", action="store_true", help="solo descarga/carga el modelo y sale")
    args = ap.parse_args()

    proto = sys.stdout
    sys.stdout = sys.stderr  # los logs de las librerías no deben mezclarse con el protocolo

    def reply(obj):
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    try:
        import numpy as np
        import torch
        from TTS.api import TTS
    except ImportError as e:
        reply({"error": f"Faltan dependencias en el venv de XTTS: {e}"})
        return

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    t0 = time.time()
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    except Exception as e:  # incluye no haber aceptado la licencia CPML en modo no interactivo
        reply({"error": f"No pude cargar XTTS: {e}. Ejecuta una vez a mano con --download "
                        f"para descargar el modelo y aceptar su licencia."})
        return
    if args.download:
        print(f"Modelo XTTS listo en {time.time() - t0:.1f} s.")
        return
    ref = Path(args.ref or "")
    if not ref.is_file():
        reply({"error": f"No encuentro el audio de referencia: {ref}"})
        return

    # Camino rápido: calcular una vez los vectores de la voz y reutilizarlos en cada frase.
    model = getattr(getattr(tts, "synthesizer", None), "tts_model", None)
    latents = None
    if model is not None and hasattr(model, "get_conditioning_latents"):
        latents = model.get_conditioning_latents(audio_path=[str(ref)])

    def to_np(wav):
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    def write_wav(wav):
        pcm = (np.clip(to_np(wav), -1.0, 1.0) * 32767).astype(np.int16)
        fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fd.close()
        with wave.open(fd.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return fd.name

    can_stream = (args.stream_chunk > 0 and latents is not None
                  and hasattr(model, "inference_stream"))

    def synth_chunks(text):
        """Genera el audio por trozos: con streaming llegan en cuanto están listos."""
        if can_stream:
            yield from model.inference_stream(
                text, args.lang, latents[0], latents[1], temperature=args.temperature,
                stream_chunk_size=args.stream_chunk, enable_text_splitting=False)
        elif latents is not None:
            yield model.inference(text, args.lang, latents[0], latents[1],
                                  temperature=args.temperature)["wav"]
        else:
            yield tts.tts(text=text, speaker_wav=str(ref), language=args.lang)

    reply({"ready": True, "load_s": round(time.time() - t0, 2), "sample_rate": SAMPLE_RATE,
           "streaming": can_stream})
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if req.get("cmd") == "quit":
            break
        try:
            for chunk in synth_chunks(req["text"]):
                reply({"ok": True, "path": write_wav(chunk), "last": False})
            reply({"ok": True, "last": True})
        except Exception as e:
            reply({"error": f"Fallo al sintetizar: {e}"})


if __name__ == "__main__":
    main()
