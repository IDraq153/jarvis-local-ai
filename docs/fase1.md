# Fase 1: cerebro LLM

**Hito cumplido:** chat con JARVIS por terminal, 100 % local (`python chat.py`).

## Decisión
Modelo base: **qwen3:8b** (Ollama, `think: false`). Alternativa ligera: `qwen2.5:7b-instruct-q4_K_M`
(cambiar `llm.model` en `config.yaml`).

| Modelo | tok/s | VRAM | Notas |
|---|---|---|---|
| qwen2.5:7b-instruct-q4_K_M | ~77 | 4.43 GB | rápido, buen español |
| llama3.1:8b-instruct-q4_K_M | ~72 | 4.92 GB | inventa capacidades, verboso |
| qwen3:8b | ~67 | 5.20 GB | mejor calidad; elegido |

Detalle y respuestas de muestra: `fase1_bench.md`.

## Lecciones
- En Windows, `localhost` en `requests` tardaba ~2 s por conexión (IPv6 antes que IPv4). Usar
  `127.0.0.1`: el primer token pasó de ~2.1 s a ~0.3 s.
- qwen3 necesita `think: false` para conversación fluida (el pensamiento llega en un campo aparte).
- El primer prompt tras una carga en frío sale muy lento en el benchmark; mirar los siguientes.
- Precargar el modelo al abrir el chat (`warmup`) evita pagar la carga en el primer mensaje.

## Presupuesto de VRAM resultante (8 GB)
LLM ~5.2 GB, quedan ~2.8 GB para STT (faster-whisper small ~1 GB) y TTS ligero.
XTTS y F5-TTS se prueban solos (descargando el LLM antes).

## Archivos
`llm/` (interfaz + backend Ollama), `core/persona.py`, `chat.py`, `scripts/bench_models.py`,
`tests/test_ollama_backend.py` (`python -m unittest discover -s tests`).
