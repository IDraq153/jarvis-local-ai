# Fase 3: benchmark de STT

| Modelo | Dispositivo | WER | Latencia media | Latencia máx | RTF | RAM extra | Carga |
|---|---|---|---|---|---|---|---|
| base [sin_vad] | cpu/int8 | 28.0% | 0.65 s | 0.71 s | 0.177 | 156 MB | 1.0 s |
| small [sin_vad] | cpu/int8 | 19.9% | 1.99 s | 2.28 s | 0.54 | 275 MB | 1.1 s |

WER = tasa de error de palabras (sin tildes ni puntuación). RTF < 1 = más rápido que tiempo real.

**base [sin_vad]: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «»
- (75%) «Abre el navegador y busca información sobre ciberseguridad.» → «Abren de dor y busca información sobre si ver seguridad.»
- (60%) «Ejecuta el servidor de Ollama con el modelo qwen tres.» → «Ejecú trae el servidor de olama en el modelo Q3.»

**small [sin_vad]: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «¡Suscríbete!»
- (40%) «Ejecuta el servidor de Ollama con el modelo qwen tres.» → «ejecuto el servidor de olama en el modelo Queen 3»
- (38%) «Revisa si la tarjeta gráfica tiene memoria libre.» → «de revisas y la tarjeta gráfica tiene memoria libre.»
