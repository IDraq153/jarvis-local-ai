# Fase 3: oído (STT)

**Hito cumplido (24-nov-2026):** `python listen.py` convierte lo que dices al micrófono en texto, 100 % local,
sin usar VRAM (faster-whisper `small` en CPU con int8).

## Qué se construyó

- `stt/`: interfaz común `STTBackend` (`transcribe` → `Transcript` con texto, idioma, duración y RTF), igual que
  `llm/` y `tts/`. Motor actual: faster-whisper. `recorder.py` captura el micrófono a 16 kHz y detecta el fin de
  frase por energía (`Endpointer`, lógica pura y probada); `metrics.py` calcula el WER.
- `listen.py`: bucle micrófono → texto (`--debug` muestra tiempos y RTF, `--list-mics`, `--mic N`, `--once`).
  Di "salir" (como frase completa) o Ctrl+C para terminar.
- `scripts/bench_stt.py` + `scripts/frases_stt.txt`: graba 12 frases tuyas y compara modelos (WER, latencia, RTF, RAM).
- `tests/test_stt.py`: 11 pruebas sin micrófono ni modelos.

## Resultados (Ryzen 7 5700G, CPU int8, 12 frases en español de Dario, `beam_size=1`)

| Modelo | Vocabulario | WER | Latencia media | RTF | RAM extra |
|---|---|---|---|---|---|
| tiny | no | ~34 %* | 0.31 s | 0.08 | 154 MB |
| base | no | ~24 %* | 0.59 s | 0.16 | 103-181 MB |
| base | sí | 18.5 % | 0.56 s | 0.15 | 177 MB |
| small | no | ~13 %* | 1.6-2.0 s | 0.42-0.55 | 280 MB |
| **small** | **sí** | **9.1 %** | **2.0 s** | **0.54** | **274 MB** |

\* Primera tanda, estimada sin la frase 1, que salió mal grabada (cortada) en esa ronda. Informes completos:
`docs/fase3_bench*.md/json` (`_final` es la tanda definitiva, con la frase 1 regrabada).
WER sin tildes ni puntuación y con números como palabras ("8" = "ocho").

## Decisiones

- **`small` en CPU (int8)**: precisión de GPU sin gastar VRAM (con LLM + voz solo quedan ~535 MB libres).
  `base` queda como alternativa si en la Fase 4 la pausa de ~2 s resulta incómoda (cambiar `stt.model` en `config.yaml`).
- **`initial_prompt` con vocabulario** ("Jarvis, Ollama, Qwen, CCNA, GitHub, VRAM, ciberseguridad, Lima"): bajó el WER
  de `small` de ~13 % a 9 % y arregló "olama"/"Queen". Ampliar la lista cuando aparezcan términos nuevos.
- **VAD de faster-whisper activo** (`vad_filter: true`): con casi-silencio, sin VAD `small` inventó "¡Suscríbete!"; con VAD devuelve vacío.
- **Latencia casi fija** (~1.5-2 s en `small` aunque el audio dure 2 o 6 s): coste del modelo en CPU, no de la duración.

## Lecciones

- Una frase mal grabada (cortada, volumen bajo) fallaba igual en los tres modelos: antes de culpar al modelo, mirar el .wav.
- Tu voz sale con volumen bajo (RMS ~0.015): subir la ganancia del micrófono en Windows mejora el corte de frases.
- Errores que quedan en `small`: "fase" → "pase", "revisa" → "de revisas", sufijos como "CCNA uno" → "CCNA1". Se ven poco en órdenes cortas.
- Medir con las frases propias (`--record`) vale más que cualquier tabla genérica de Whisper.
- Los .wav grabados (`stt/samples/`) no se versionan: son datos biométricos y el repo es público.

## Repetir la instalación y las medidas

- `pip install -r requirements-stt.txt` en el venv principal. El modelo se descarga solo la primera vez (small ~460 MB).
- Prueba: `python listen.py --debug`. Micrófonos: `python listen.py --list-mics`.
- Benchmark: `python scripts\bench_stt.py --record`, luego `python scripts\bench_stt.py --models tiny,base,small --prompt "..." --tag X`.
  `--only N` regraba solo la frase N; `--no-vad` desactiva el VAD.

## Para la Fase 4

- `listen.py` ya devuelve el texto: falta enlazarlo con `chat.py` y turnarse con la voz (no escucharse a sí mismo mientras habla JARVIS).
- Presupuesto de latencia por turno: ~2 s STT + ~0.3 s primer token + ~0.3 s primer audio. Si molesta, probar `base` o STT por trozos.
