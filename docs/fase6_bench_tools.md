# Fase 6, 6.0: bench de tool calling (qwen3:8b, num_ctx 4096)

Memoria simulada en el prompt: sí. VRAM usada: 6400 MB antes, 6371 MB después (de 8151 MB).

## Vuelta 1 (el LLM elige herramienta)

| N herram. | tokens esquemas (aprox.) | tokens prompt (mediana) | herramienta correcta | args correctos | falsos positivos | latencia mediana | p95 |
|---|---|---|---|---|---|---|---|
| 3 | 333 | 1276 | 84/90 (93 %) | 84/90 (93 %) | 0/24 (0 %) | 0.49 s | 0.90 s |
| 5 | 589 | 1480 | 83/90 (92 %) | 83/90 (92 %) | 0/24 (0 %) | 0.58 s | 0.97 s |
| 8 | 930 | 1762 | 82/90 (91 %) | 82/90 (91 %) | 0/24 (0 %) | 0.70 s | 1.17 s |
| 13 | 1498 | 2232 | 80/90 (89 %) | 80/90 (89 %) | 0/24 (0 %) | 1.00 s | 1.57 s |

## Vuelta 2 (el LLM redacta con el resultado de la herramienta)

Muestras: 26. Primer token: mediana 0.06 s. Total: mediana 0.46 s. Alternativa con plantilla de Python: ~0 s.

## Fallos de la vuelta 1

- N=3: «¿Cuántos pasos he dado hoy?» esperaba progreso_diario, dio None ; «Ya terminé el informe de redes» esperaba completar_tarea, dio None ; «Ya hice el hábito de leer» esperaba marcar_habito, dio None ; «Ya cumplí con el hábito de ejercicio» esperaba marcar_habito, dio None ; «Ya terminé el informe de redes» esperaba completar_tarea, dio None ; «Ya cumplí con el hábito de ejercicio» esperaba marcar_habito, dio None 
- N=5: «Ya terminé el informe de redes» esperaba completar_tarea, dio None ; «Ya hice el hábito de leer» esperaba marcar_habito, dio None ; «Ya cumplí con el hábito de ejercicio» esperaba marcar_habito, dio None ; «Ya terminé el informe de redes» esperaba completar_tarea, dio None ; «Ya hice el hábito de leer» esperaba marcar_habito, dio None ; «Léeme mis notas del ciclo» esperaba notas_curso, dio record_academico {'ciclo': 'todos'}; «Ya cumplí con el hábito de ejercicio» esperaba marcar_habito, dio None 
- N=8: «Ya terminé el informe de redes» esperaba completar_tarea, dio None ; «Ya hice el hábito de leer» esperaba marcar_habito, dio None ; «Léeme mis notas del ciclo» esperaba notas_curso, dio record_academico {'ciclo': 'todos'}; «Ya cumplí con el hábito de ejercicio» esperaba marcar_habito, dio None ; «Ya terminé el informe de redes» esperaba completar_tarea, dio None ; «Léeme mis notas del ciclo» esperaba notas_curso, dio record_academico {'ciclo': 'ciclo'}; «¿Ya cumplí mi meta de pasos?» esperaba progreso_diario, dio None ; «Ya cumplí con el hábito de ejercicio» esperaba marcar_habito, dio None 
- N=13: «Ya terminé el informe de redes» esperaba completar_tarea, dio None ; «Ya hice el hábito de leer» esperaba marcar_habito, dio None ; «Léeme mis notas del ciclo» esperaba notas_curso, dio record_academico {'ciclo': 'todos'}; «Ya cumplí con el hábito de ejercicio» esperaba marcar_habito, dio None ; «Ya terminé el informe de redes» esperaba completar_tarea, dio None ; «Marca mi hábito de agua» esperaba marcar_habito, dio None ; «Ya hice el hábito de leer» esperaba marcar_habito, dio None ; «Léeme mis notas del ciclo» esperaba notas_curso, dio record_academico {'ciclo': 'todos'}; «¿Ya cumplí mi meta de pasos?» esperaba progreso_diario, dio None ; «Ya cumplí con el hábito de ejercicio» esperaba marcar_habito, dio None 

## Selector por embeddings (bge-m3, CPU)

Textos indexados: 77 (en 3.17 s, una sola vez). Latencia por consulta: mediana 41 ms, máx. 65 ms.

### Frases que SÍ están en los ejemplos (optimista: el selector las conoce)

Positivas: 16, charla normal: 0.

| top-k | herramienta correcta dentro |
|---|---|
| 3 | 16/16 (100 %) |
| 4 | 16/16 (100 %) |
| 5 | 16/16 (100 %) |

Parecido mínimo de una frase de herramienta: 0.9477636814117432; máximo de charla normal: None.
Fuera del top-4: ninguna. Charla con parecido > 0.55: ninguna.

### Frases NUEVAS (la medida que importa)

Positivas: 29, charla normal: 12.

| top-k | herramienta correcta dentro |
|---|---|
| 3 | 27/29 (93 %) |
| 4 | 28/29 (97 %) |
| 5 | 28/29 (97 %) |

Parecido mínimo de una frase de herramienta: 0.5717484951019287; máximo de charla normal: 0.751156210899353.
Fuera del top-4: ['¿Cómo me está yendo en gestión financiera?']. Charla con parecido > 0.55: ['Cuéntame un chiste corto', '¿Cuántas horas debería dormir una persona?', 'Dame consejos para organizar mi tiempo de estudio', '¿Qué es la progresión de carga en el gimnasio?', '¿Qué significa tener una buena nota en una universidad?'].

Umbral sugerido con frases nuevas (`tools.selector.min_score`): 0.66.
