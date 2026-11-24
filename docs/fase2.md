# Fase 2: laboratorio de voces (TTS)

**Hito cumplido (24-nov-2026):** `python chat.py --voz` lee en voz alta las respuestas de JARVIS, frase por frase
mientras el LLM sigue escribiendo, con una voz clonada de un clip de referencia.

## Qué se construyó

- `tts/`: interfaz común `TTSBackend` (`stream` → `Audio`), igual que `llm/`. Motores: Piper, Kokoro y XTTS
  (este último en su propio venv, vía worker por tubería JSON). `speaker.py` sintetiza y reproduce en segundo
  plano sin cortes; `text.py` limpia markdown y parte el texto en frases.
- `voicelab/`: catálogo `voices.yaml`, `bench.py` (latencia, RTF y VRAM; guarda `results.json` y muestras .wav),
  `report.py` (genera `samples/index.html` con reproductores), `tune.py` (compara velocidades de Kokoro).
- `chat.py --voz` y `core/gpu.py` (muestra la VRAM real con LLM y voz cargados).

## Resultados (RTX 5060 8 GB, 4 frases en español, qwen3 descargado salvo en la prueba final)

| Voz | Carga | VRAM extra | Primer audio | RTF |
|---|---|---|---|---|
| Piper es_MX-claude-high | 1.2 s | 0 MB | 0.21 s | 0.036 |
| Piper es_MX-ald-medium | 0.6 s | 0 MB | 0.20 s | 0.034 |
| Kokoro ef_dora (GPU) | 24 s* | 1305 MB | 0.30 s | 0.038 |
| Kokoro em_alex (GPU) | 1.7 s | 1216 MB | 0.32 s | 0.043 |
| Kokoro em_santa (GPU) | 1.8 s | 1214 MB | 0.26 s | 0.033 |
| Kokoro ef_dora (CPU) | 1.5 s | 0 MB | 3.42 s | 0.362 |
| **Kokoro + pack am_voz1 (GPU)** | 8.7 s | 1296 MB | 0.29 s | 0.029 |
| XTTS v2 clonando voz1 (streaming) | 26.8 s | 2605 MB | 0.64 s | 0.724 |

\* la primera carga incluye la descarga del modelo. RTF < 1 = más rápido que tiempo real.
XTTS sin streaming tardaba 6-7 s en el primer audio de frases largas (RTF 0.85-0.90).

## Decisión

Voz elegida: **Kokoro con un pack clonado de `voz1`** (`voicelab/packs/am_voz1.pt`, creado con `inno-kokoro`),
`speed: 1.1`. Con qwen3:8b y la voz cargados: **7616 / 8151 MB de VRAM, 535 MB libres**.

Por qué no las otras:
- Piper y las voces de serie de Kokoro: rápidas y ligeras, pero no clonan y sonaron "mid".
- XTTS: suena más natural y clona bien, pero con +2.6 GB no cabe junto a qwen3:8b (sí con qwen3:4b).
- Fish Audio (S2-Pro 4B, Drama 3 35B): clonan muy bien, pero no caben en 8 GB.
- F5-TTS: no se probó (quedó como comparación opcional; el finetune en español es CC-BY-NC).

## Lecciones

- Los venvs aislados valieron la pena: XTTS exige `transformers>=4.57,<5` (la 5.x da `get_seq_length ... NoneType`)
  y `torch 2.8.0+cu128` evita necesitar torchcodec/FFmpeg. Falta `six` en la instalación de coqui-tts.
- La licencia de XTTS (CPML) se acepta a mano ejecutando `xtts_worker.py --download`; por la tubería del
  benchmark no se puede responder. Solo uso no comercial, y también cubre el audio generado.
- El streaming de XTTS (`inference_stream`) bajó el primer audio de ~6 s a ~0.64 s.
- `inno-kokoro` es solo para inglés: con español clona el timbre de forma superficial; aun así fue suficiente.
- `trim_cache` (vaciar la caché de CUDA tras cada trozo) mantiene baja la VRAM reservada de Kokoro.
- Medir la VRAM del sistema, no estimarla: el cálculo previo daba 8368 MB y lo real fue 7616 MB.

## Repetir la instalación

- Venv principal: `pip install -r requirements-tts.txt`; voces de Piper con
  `python -m piper.download_voices es_MX-claude-high es_MX-ald-medium` dentro de `voicelab\models\piper`.
- XTTS: `voicelab/requirements-xtts.txt`. Pack de voz: `voicelab/requirements-kclone.txt`.
- Benchmark: `python -m voicelab.bench [--voice ID] [--runs N]`; muestras en `voicelab/samples/index.html`.
- Los clips (`voicelab/refs/`) y packs (`voicelab/packs/`) no se versionan: son datos biométricos y el repo es público.

## Para la Fase 3 y siguientes

- VRAM libre con LLM + voz: ~535 MB. El STT planeado (faster-whisper small, ~1 GB) **no cabe en GPU**:
  usar CPU (int8), un modelo menor, o liberar VRAM (bajar `num_ctx`, `OLLAMA_KV_CACHE_TYPE=q8_0`, qwen3:4b).
- Si se quiere más naturalidad: XTTS con qwen3:4b, o entrenar una voz de Piper con frases generadas por XTTS.
