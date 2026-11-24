# Fase 3: benchmark de STT

| Modelo | Dispositivo | WER | Latencia media | Latencia máx | RTF | RAM extra | Carga |
|---|---|---|---|---|---|---|---|
| tiny | cpu/int8 | 39.4% | 0.31 s | 0.71 s | 0.083 | 154 MB | 5.9 s |
| base | cpu/int8 | 30.4% | 0.59 s | 0.72 s | 0.161 | 103 MB | 6.0 s |
| small | cpu/int8 | 22.5% | 1.56 s | 1.78 s | 0.423 | 278 MB | 1.2 s |

WER = tasa de error de palabras (sin tildes ni puntuación). RTF < 1 = más rápido que tiempo real.

**tiny: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «»
- (75%) «Abre el navegador y busca información sobre ciberseguridad.» → «Abren, navegó, y busca información sobre si verse Buriedad.»
- (71%) «¿Cuántos gigabytes de VRAM me quedan disponibles?» → «¿Cuántos y el guadides debidram de me quedan disponibles?»

**base: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «»
- (75%) «Abre el navegador y busca información sobre ciberseguridad.» → «Abren de dor y busca información sobre si ver seguridad.»
- (60%) «Ejecuta el servidor de Ollama con el modelo qwen tres.» → «Ejecú trae el servidor de olama en el modelo Q3.»

**small: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «»
- (50%) «Ejecuta el servidor de Ollama con el modelo qwen tres.» → «ejecuto el servidor de olama en el modelo Queen 3»
- (38%) «Necesito practicar para el examen de CCNA uno.» → «Necesito practicar para el examen de C-CNA-1»
