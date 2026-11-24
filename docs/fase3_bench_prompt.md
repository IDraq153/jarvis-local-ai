# Fase 3: benchmark de STT

| Modelo | Dispositivo | WER | Latencia media | Latencia máx | RTF | RAM extra | Carga |
|---|---|---|---|---|---|---|---|
| base [prompt] | cpu/int8 | 24.5% | 0.56 s | 0.59 s | 0.151 | 153 MB | 1.2 s |
| small [prompt] | cpu/int8 | 17.4% | 1.73 s | 1.84 s | 0.469 | 272 MB | 1.1 s |

WER = tasa de error de palabras (sin tildes ni puntuación). RTF < 1 = más rápido que tiempo real.

**base [prompt]: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «»
- (50%) «Abre el navegador y busca información sobre ciberseguridad.» → «Abren, la de GOR y busca información sobre ciberseguridad.»
- (38%) «Revisa si la tarjeta gráfica tiene memoria libre.» → «de revisas y la tarjeta gráfica tiene memoria libre.»

**small [prompt]: peores frases**
- (100%) «Hola Jarvis, ¿qué hora es en Lima?» → «»
- (38%) «Revisa si la tarjeta gráfica tiene memoria libre.» → «de revisas y la tarjeta gráfica tiene memoria libre.»
- (25%) «Necesito practicar para el examen de CCNA uno.» → «necesito practicar para el examen de CCNA1»
