# Fase 3: benchmark de STT

| Modelo | Dispositivo | WER | Latencia media | Latencia máx | RTF | RAM extra | Carga |
|---|---|---|---|---|---|---|---|
| base [final] | cpu/int8 | 18.5% | 0.56 s | 0.6 s | 0.152 | 177 MB | 1.1 s |
| small [final] | cpu/int8 | 9.1% | 2.01 s | 2.21 s | 0.542 | 274 MB | 1.2 s |

WER = tasa de error de palabras (sin tildes ni puntuación). RTF < 1 = más rápido que tiempo real.

**base [final]: peores frases**
- (50%) «Abre el navegador y busca información sobre ciberseguridad.» → «Abren, la de GOR y busca información sobre ciberseguridad.»
- (38%) «Revisa si la tarjeta gráfica tiene memoria libre.» → «de revisas y la tarjeta gráfica tiene memoria libre.»
- (33%) «Sube los cambios a GitHub con un commit que diga fase tres.» → «Sube los camis al GitHub con un comit que diga Pase 3»

**small [final]: peores frases**
- (38%) «Revisa si la tarjeta gráfica tiene memoria libre.» → «de revisas y la tarjeta gráfica tiene memoria libre.»
- (25%) «Necesito practicar para el examen de CCNA uno.» → «necesito practicar para el examen de CCNA1»
- (17%) «Sube los cambios a GitHub con un commit que diga fase tres.» → «sube los cambios a GitHub con un commit que diga Pace3.»
