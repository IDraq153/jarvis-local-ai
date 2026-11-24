# Fase 3: benchmark de STT

| Modelo | Dispositivo | WER | Latencia media | Latencia máx | RTF | RAM extra | Carga |
|---|---|---|---|---|---|---|---|
| base [base_vad] | cpu/int8 | 28.0% | 0.6 s | 0.69 s | 0.163 | 181 MB | 1.1 s |
| small [base_vad] | cpu/int8 | 19.9% | 2.03 s | 2.29 s | 0.551 | 280 MB | 1.3 s |

WER = tasa de error de palabras (sin tildes ni puntuación). RTF < 1 = más rápido que tiempo real.

**base [base_vad]: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «»
- (75%) «Abre el navegador y busca información sobre ciberseguridad.» → «Abren de dor y busca información sobre si ver seguridad.»
- (60%) «Ejecuta el servidor de Ollama con el modelo qwen tres.» → «Ejecú trae el servidor de olama en el modelo Q3.»

**small [base_vad]: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «»
- (40%) «Ejecuta el servidor de Ollama con el modelo qwen tres.» → «ejecuto el servidor de olama en el modelo Queen 3»
- (38%) «Revisa si la tarjeta gráfica tiene memoria libre.» → «de revisas y la tarjeta gráfica tiene memoria libre.»
