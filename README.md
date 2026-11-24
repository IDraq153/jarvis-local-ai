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
- [ ] 3. Oído (STT)
- [ ] 4. Conversación por voz
- [ ] 5. Memoria
- [ ] 6. Herramientas
- [ ] 7. Interfaz futurista
- [ ] 8. Pulido offline

## Estado

Fases 0 y 1 completadas (cerebro: qwen3:8b vía Ollama, chat por terminal). Siguiente: Fase 2 (laboratorio de voces).
