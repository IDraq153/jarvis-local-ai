# Fase 4: conversación por voz

Hito: `python voice_chat.py` permite hablar con JARVIS de principio a fin (micrófono -> faster-whisper -> qwen3:8b -> Kokoro con la voz clonada), 100 % local.

## Diseño
- `core/voice_loop.py`: `VoiceLoop.run_turn()` con `listen`, `transcribe`, `llm` y `speaker` inyectados (se prueba sin hardware). `voice_chat.py` solo cablea config, modelos y micrófono.
- **Anti-eco por turnos (half-duplex):** el micrófono no está abierto mientras JARVIS habla; se reabre `voice.tail_ms` (500 ms) tras acabar el último audio. Segunda barrera: se ignora una transcripción cuyas palabras (>= 4, 80 %) ya estaban en la última respuesta.
- Filtro de ruido: transcripciones vacías o alucinaciones típicas de Whisper ("Suscríbete", "Gracias por ver el video").
- Historial acotado: el LLM ve el sistema + los últimos `voice.max_turns` (8) intercambios + la pregunta pendiente (num_ctx 4096); el historial completo se conserva.
- Comandos por voz: "salir" / "adiós" / "hasta luego"; "reinicia" / "olvida todo".
- Sin barge-in (interrumpir a JARVIS hablando): queda como mejora futura.

## Medición (small CPU int8 + qwen3:8b + Kokoro voz1, 4 turnos reales de Dario)
| Tramo | Medido |
|---|---|
| STT | 1.62 - 2.06 s |
| 1.er token del LLM | 0.16 - 0.27 s |
| 1.er audio tras el 1.er texto | +0.55 - 1.28 s |
| **Percibido (fin de tu frase -> 1.er sonido)** | **2.55 - 3.51 s** |

El STT es ~60-70 % de la espera. Alternativas si molesta: `--stt-model base` (0.56 s, WER 18.5 %), bajar `stt.silence_ms` de 800 a 600 (-0.2 s), o STT por trozos.

## Lecciones
- El modelo se dirigía a Dario como "Amo" y se negaba a hacer de JARVIS de la película: se corrigió en `core/persona.py` (llamarlo por su nombre y asumir el papel de JARVIS).
- `clean_for_speech` ya quita los asteriscos de markdown (`*Iron Man*`), no se leen en voz alta.
- Con voz a ~7.6 GB de VRAM usados, el STT debe seguir en CPU.
