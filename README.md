# JARVIS — Personal Local AI Laboratory

Asistente personal de IA que funciona de forma local: cerebro LLM, laboratorio de voces (TTS), reconocimiento de voz (STT), memoria persistente, herramientas controladas e interfaz futurista. Objetivo: máximo nivel posible de funcionamiento offline.

## Arquitectura (modular)

```
Micrófono → [STT] → [Orquestador] ⇄ [LLM local]
                         ⇅              ⇅
                    [Memoria]      [Herramientas]
                         ↓
                      [TTS] → Altavoz        Interfaz (UI)
```

Cada módulo expone una interfaz fija, así se puede cambiar un modelo sin reconstruir el sistema.

## Estructura

| Carpeta | Contenido |
|---|---|
| `core/` | Orquestador y bucle principal |
| `llm/` | Adaptadores: Ollama, llama.cpp |
| `stt/` | Adaptadores: faster-whisper, whisper.cpp, Vosk |
| `tts/` | Adaptadores: Piper, Kokoro, XTTS, F5-TTS |
| `voicelab/` | Laboratorio de voces independiente |
| `memory/` | SQLite + base vectorial |
| `tools/` | Herramientas con permisos |
| `ui/` | Interfaz futurista |
| `docs/` | Notas y decisiones por fase |

## Fases

- [x] 0. Base y hardware
- [x] 1. Cerebro (LLM)
- [x] 2. Laboratorio de voces (TTS)
- [x] 3. Oído (STT)
- [x] 4. Conversación por voz
- [x] 5. Memoria persistente
- [ ] 6. Herramientas
- [ ] 7. Interfaz futurista
- [ ] 8. Pulido offline

## Estado

Fases 0 a 4 completadas: cerebro (qwen3:8b vía Ollama), voz (Kokoro clonada, `chat.py --voz`) y oído (faster-whisper small en CPU, `python listen.py`). Conversación por voz completa con `python voice_chat.py --debug` (docs/fase4.md). Fase 5 completada: memoria persistente con SQLite y vectores (bge-m3 en CPU), ver docs/fase5.md. Siguiente: Fase 6 (herramientas).
