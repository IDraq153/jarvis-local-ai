# Fase 5: memoria persistente (SQLite + vectores)

Hito: JARVIS recuerda entre sesiones. Lo que le cuentas hoy lo puede usar mañana, sin gastar VRAM y sin tocar el historial de la conversación.

## Diseño

Tres capas en un solo SQLite local (`data/jarvis_memory.db`, ignorado por git: el repo es público):

| Capa | Qué guarda | Cómo se crea |
|---|---|---|
| **Hechos** | Datos duraderos sobre Dario | "recuerda que...", `/recordar`, o extraídos por el LLM al salir |
| **Recuerdos** | Cada turno (pregunta + respuesta) con su vector | Automático, tras cada respuesta |
| **Resumen** | 1-2 frases de la última sesión | Al salir, con el LLM |

En cada turno: se embebe lo que dices, se buscan los hechos y recuerdos más parecidos y se **añaden al prompt de sistema** de esa llamada (copia; `history` queda intacto). Bloque limitado a `max_block_chars` (~300 tokens) para respetar `num_ctx: 4096`.

- **Vectores:** Ollama `/api/embed` con `bge-m3` (multilingüe) en **CPU** (`num_gpu: 0`): 0 MB de VRAM. Coseno por fuerza bruta con numpy sobre BLOBs float32; para una memoria personal sobra y no necesita extensiones nativas en Windows (sqlite-vec queda como mejora si algún día hay >100 000 filas).
- **Cambio de modelo:** cada vector guarda su `embed_model`; al iniciar, `reindex()` re-embebe lo que sea de otro modelo o no tenga vector.
- **Duplicados y correcciones:** un hecho con coseno >= `dedupe_score` (0.90) a otro lo **actualiza** (gana el más nuevo), así "ahora vivo en X" corrige al antiguo.
- **Degradación:** si Ollama no responde o falta el modelo, avisa una vez y busca por palabras en común; al volver, reindexa solo. La memoria nunca debe tumbar el chat.
- **Privacidad:** no se guardan contraseñas, números de 8+ dígitos (documentos, cuentas) ni de tarjeta, ni con "recuerda que" ni por extracción automática (`memory/text.py::looks_secret`).
- **Cierre de sesión:** con >= 2 turnos, el LLM devuelve `{"resumen", "hechos"}` (máx. 5 hechos, en tercera persona). Tarda unos segundos al salir.

## Uso

```
python chat.py            # memoria activa por defecto (--sin-memoria la apaga)
python voice_chat.py
```

| Dices / escribes | Efecto |
|---|---|
| "recuerda que mi examen de redes es el viernes" / `/recordar ...` | Guarda un hecho |
| "olvida lo de mi examen" / `/olvidar texto` | Borra el hecho más parecido (dice cuál) |
| `/memoria` / `/olvidar 3` | Lista con números / borra por número (solo chat.py) |
| "qué sabes de mí" | Resume lo que hay guardado |
| "reinicia" / `/reset` | Solo borra la conversación en curso; **la memoria se conserva** |

`--debug` muestra el tiempo de búsqueda y el bloque exacto inyectado en cada turno.

## Código

`memory/`: `store.py` (SQLite + búsqueda), `embeddings.py` (interfaz Embedder + Ollama), `manager.py` (`Memory`: augment, record_turn, remember/forget, close_session), `commands.py` (órdenes en lenguaje natural), `extract.py` (resumen y hechos con el LLM), `text.py`. Integración: `chat.py`, `voice_chat.py`, `core/voice_loop.py` (`memory=`; turno de tipo `"memory"`). Pruebas: `tests/test_memory.py` (17; 54 en total) con embedder y LLM falsos.

## Pendiente de verificar en la PC (con Ollama real)

1. `ollama pull bge-m3`
2. `python scripts/bench_embeddings.py` -> latencia en CPU, acierto y **umbral sugerido**; ajustar `memory.min_fact_score` / `min_episode_score` en config.yaml. Los valores por defecto (0.45 / 0.55) son estimaciones, no medidas.
3. Probar dos sesiones seguidas con `python voice_chat.py --debug` y mirar `[memoria] búsqueda N ms`: suma a la latencia percibida (antes 2.5-3.5 s). Si molesta, probar `embeddinggemma` o `nomic-embed-text`.
4. Comprobar que `bge-m3` en CPU no desplaza a qwen3:8b de la GPU (`ollama ps`).

## Límites conocidos

- Los hechos no caducan ("el examen es el viernes" sigue ahí la semana siguiente): se limpian con "olvida...". Posible mejora: fecha de caducidad o repaso periódico.
- Un hecho en contradicción pero poco parecido (< 0.90) convive con el antiguo.
- La extracción depende de qwen3:8b y puede omitir o inventar; revisar con `/memoria`.
- Sin barge-in y sin resumen intermedio: una sesión muy larga solo se resume al salir (si se cierra a la fuerza, quedan los turnos pero no el resumen).
