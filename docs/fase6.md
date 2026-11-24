# Fase 6: herramientas (UniGym Tracker por voz y texto)

Estado: **en curso**. Hechas 6.0 a 6.5 (falta probar 6.5 con micrófono real). Falta 6.6.
Las pruebas pasan: 59 de las fases anteriores + 165 nuevas = 224.

## Decisiones (24-nov-2026, con Dario)
- **Integración: opción B refinada.** JARVIS es dueño de `data/unigym.json` (ignorado por git). La web lo lee y
  escribe por `GET/PUT /api/state`; si no hay servidor cae a localStorage como hoy.
- **Servidor web como comando aparte** (`python unigym_serve.py`, reemplaza a `python -m http.server 8080`).
  JARVIS escribe directo al archivo, así que funciona aunque el servidor esté apagado.
- **Presupuesto de latencia: se decide con el bench (6.0)**, no antes. Se compararán respuesta por plantilla
  de Python (sin segunda vuelta del LLM) y respuesta redactada por qwen3:8b.
- Opciones descartadas: A (respaldo, solo lectura), C (SQLite ahora, trabajo de la Fase 7), D (cola de órdenes
  que solo funciona con la pestaña abierta).

## Arquitectura
```
tools/
  base.py        ToolSpec (nombre, descripción, parámetros, frases de ejemplo, nivel READ/WRITE), approx_tokens
  schema.py      atajos para escribir esquemas JSON compactos
  selector.py    ToolSelector: embebe el catálogo (bge-m3) y devuelve solo las top-k para el turno
  prompt.py      guía de herramientas para el prompt de sistema (con la fecha de hoy)
  unigym/
    specs.py     13 herramientas (7 de lectura, 6 de escritura) como datos
    repo.py      capa de datos: JsonFileRepository (bloqueo, revisión, copia previa, escritura atómica)
    dates.py     "el viernes", "28 de noviembre", "veintiocho de noviembre" -> ISO; pregunta si es ambiguo
    names.py     "EDA 2" / "eda dos" / "EDA2" -> el mismo curso; si hay dudas, devuelve candidatos
    stats.py     courseStats y rachas portados del JS (verificados contra el JS real)
    readers.py   las 7 herramientas de lectura: estado + argumentos -> frase lista para decir
scripts/bench_tools.py            bench de tool calling (lo corre Dario, ver abajo)
scripts/check_stats_parity.py     compara stats.py con el JS de la web (necesita Node)
tests/golden/course_stats.json    valores calculados por el JS real, para la prueba sin Node
```
- El LLM solo elige herramienta y pasa argumentos crudos ("viernes", "EDA 2", "70.5"). Python resuelve fechas,
  cursos y rangos. El LLM no calcula fechas ni inventa cursos.
- Las herramientas hablan con `UniGymRepository` (interfaz), no con el archivo: la Fase 7 puede pasar a SQLite
  sin reescribirlas.
- El archivo guarda `{app, format, revision, updatedAt, by, reason, data}`. Acepta como origen un respaldo
  exportado por la web, o el estado a secas. Las claves que Python no conoce se conservan.

## Seguridad (diseño)
| Nivel | Qué | Cómo |
|---|---|---|
| Lectura | las 7 herramientas de `readers.py` | libre; nunca modifican el estado (hay prueba) |
| Escritura reversible | crear/completar tarea, hábito, peso, registro diario, nota | propuesta sin guardar, JARVIS repite lo entendido, solo guarda con un sí claro (6.4) |
| Destructiva | borrados | fuera de esta fase salvo que Dario lo pida |

- Copia previa en `data/backups/unigym-AAAAMMDD-HHMMSS-rN.json` antes de cada escritura (últimas 30).
- Cada escritura exige que el estado resultante valide (tipos, fechas ISO, kg numérico) antes de guardar.
- Conflictos: `expected_revision`; si la web escribió o JARVIS escribió entre medias, `RevisionConflict`.
- Registro de cada llamada a herramienta en `data/jarvis_memory.db` (tabla `tool_calls`, 6.4).
- Entradas del STT: fechas resueltas en Python (rechaza fechas pasadas para tareas, avisa si "el martes"
  dicho un martes es ambiguo), cursos por parecido con candidatos si hay duda, rangos numéricos (6.4).

## 6.0 Bench de tool calling (2.ª corrida, 24-nov-2026, qwen3:8b, num_ctx 4096, 45 frases sintéticas × 2)
`python scripts\bench_tools.py --repeats 2`; informe completo en `docs/fase6_bench_tools.md`. 29 frases son nuevas (no están en los
ejemplos de `specs.py`) y 12 son charla normal cercana al dominio.

| N herramientas en el prompt | tokens de prompt (mediana) | herramienta correcta | falsos positivos | latencia vuelta 1 (mediana / p95) |
|---|---|---|---|---|
| 3 | 892 | 87/90 (97 %) | 0/24 | 0.56 s / 1.57 s |
| 5 | 1076 | 86/90 (96 %) | 0/24 | 0.65 s / 3.32 s |
| 8 | 1346 | 86/90 (96 %) | 0/24 | 0.80 s / 2.06 s |
| 13 | 1807 | 83/90 (92 %) | 0/24 | 1.01 s / 2.19 s |

- **Decisión: `top_k: 4`.** Cada herramienta extra suma ~35-40 tokens y ~45 ms; con 13 baja el acierto. VRAM: 6460 → 6501 MB (el contexto es fijo).
- **Segunda vuelta** (24 muestras): primer token 0.06 s, total 0.52 s (el prompt ya está en caché). Es casi gratis en latencia, así que la
  elección entre plantilla y LLM es de calidad, no de velocidad: `reply_mode: template` por defecto (números exactos; las confirmaciones
  de escritura son siempre de plantilla). `llm` quedó sin implementar (ver 6.5).
- **Selector con frases nuevas:** la herramienta correcta cae en el top-4 en 28/29 (97 %); la que falla es «¿Cómo me está yendo en gestión
  financiera?», la misma que el LLM nunca resolvió. **No hay umbral limpio**: la frase de herramienta más baja midió 0.52 y 5 frases de charla
  (p. ej. «Cuéntame un chiste corto») llegan a 0.55-0.75. Se fija `min_score: 0.50` (no pierde ninguna positiva conocida) y la protección real
  contra llamadas indebidas es el propio LLM, que no hizo ninguna en 24 casos.
- **Fallos del LLM (4-7 de 90):** (1) «¿Cómo me está yendo en gestión financiera?» y «¿Qué nota necesito en EDA 2 para aprobar?» sin llamar herramienta;
  (2) «Tomé un vaso de agua» → `marcar_habito` en vez de `registrar_diario` (hay un hábito llamado agua: ambigüedad real); (3) con 13 herramientas,
  «Anota que tengo que entregar el informe de redes el lunes» → `registrar_nota` con **nota 18 inventada**.
- **Cambios derivados (a verificar repitiendo el bench):** descripciones que separan hábito de registro diario; `meta` omitida de `nota_necesaria`
  significa «para aprobar» (nota mínima del curso); línea de guía en el prompt; frases de ejemplo nuevas distintas de las del bench; y la
  comprobación anti-alucinación de números (6.4).

## 6.1 y 6.2: lo verificado
- Capa de datos: 19 pruebas (importar respaldo, revisión, conflicto, mutador que falla, validación, copias y
  poda, bloqueo vivo/huérfano, 4 hilos × 15 escrituras sin perder ninguna).
- `stats.py`: paridad con `courseStats` del JS real sobre 11 cursos sintéticos (`check_stats_parity.py` dio
  "Paridad OK"); la prueba unitaria usa los valores guardados y no necesita Node.
- Fechas y nombres: 17 pruebas, incluido "el viernes" -> viernes 27-nov-2026 (caso del hito) y "EDA 2" /
  "eda dos" -> EDA2.
- Lectores: 26 pruebas con un estado inventado (agenda de hoy, tareas, rutina, notas, nota necesaria,
  progreso diario, hábitos), tolerancia a argumentos basura y a estado vacío.

## 6.3 Servidor, web y migración
- **Migración hecha:** `python scripts/unigym_import.py data/uni-gym-tracker-respaldo-2026-11-24.json` creó `data/unigym.json`
  (revisión 1; 6 cursos, 7 tareas, 4 hábitos, 18 bloques, 18 sesiones de gym, 8 pesos, 16 días de registro). Antes se copió la
  web entera a `data/web-backup-pre-fase6/` (la web no es repo git; `data/` está ignorado).
- **Servidor:** `python unigym_serve.py` (reemplaza a `python -m http.server 8080`). `tools/unigym/server.py`: sirve la web y
  `GET /api/revision`, `GET/PUT /api/state`. Solo escucha en 127.0.0.1; el PUT exige cabecera `X-UniGym`, JSON y Host de localhost
  (contra otras páginas abiertas en el navegador); nunca envía cabeceras CORS; no sirve nada fuera de la carpeta de la web.
- **Web (`js/core/store.js`, `js/app.js`):** `connectRemote()` antes de arrancar; con servidor, el archivo es la fuente de verdad y
  localStorage queda de caché; guarda con PUT (150 ms de espera) con la revisión base; sondea `/api/revision` cada 2 s y
  recarga si JARVIS escribió, pero no mientras estás escribiendo en un campo o con un modal abierto. Si guardas a la vez que
  JARVIS (409), gana lo del archivo y sale un aviso. Sin servidor (`http.server`, `file://`) todo sigue como antes.
- **Hallazgo corregido:** normalizar convertía un contenedor mal tipado (`tasks: "x"`) en lista vacía, lo que habría borrado
  tareas en silencio. Ahora todo lo que se escribe se normaliza en modo estricto y falla.
- **Pruebas nuevas:** servidor por HTTP real (11), 2 de punta a punta con el `store.js` real en Node contra el servidor real
  (carga, guardado, cambio externo, no pisar mientras editas, conflicto 409, recuperación, modo sin archivo). Se comprobó que la
  prueba falla si se rompe la cabecera. Se omiten solas si no hay Node.
- Dario abrió la web con el servidor nuevo y se ve normal. Falta comprobar que un cambio de JARVIS aparece en su pestaña (se hará en 6.6).

## 6.4 Escrituras con confirmación
```
run("crear_tarea", {...}, heard)  -> valida  -> "¿Agrego la tarea «Tarea de EDA2» para viernes 27 de noviembre?"   [nada guardado]
answer("sí")                      -> repo.write (copia previa) -> "Listo, tarea agregada."                      [guardado, revisión +1]
answer("no") / otra frase / >60 s -> se descarta; una frase que no es sí/no se procesa como orden nueva
```
- `writers.py`: 6 escrituras (`crear_tarea`, `completar_tarea`, `marcar_habito`, `registrar_peso`, `registrar_diario`, `registrar_nota`). Cada una
  valida y devuelve una `Proposal` (pregunta, texto de éxito y `apply`, que escribe con los mismos campos y formatos que la web, id al estilo `uid()`
  de la web) o una pregunta aclaratoria, o «ya estaba hecho» sin confirmar. `apply` busca por id sobre el estado más reciente: si lo que se iba a
  cambiar desapareció, falla sin guardar.
- Validación del STT: fechas en Python (ambiguas se preguntan; las tareas no aceptan pasado; los registros no aceptan futuro ni >7 días atrás, el
  peso >30); cursos, evaluaciones, tareas y hábitos por parecido con candidatos si hay duda; rangos (peso 30-250 kg, nota 0-20, pasos ≤100 000,
  sueño ≤16 h, agua ≤40 vasos, enteros donde corresponde); no duplica tareas; reemplazos se avisan («Antes había 71»).
- `grounding.py`: un número que el LLM pasa a una herramienta debe haberse dicho (dígitos, «70 con 8», «7 y media», o en palabras). Cierra el caso
  de la nota 18 inventada.
- `confirm.py`: solo un sí corto y claro confirma («sí», «dale», «confirmo», «hazlo»...); una negación, cualquier otra frase o pasar el tiempo cancelan.
- `audit.py`: tabla `tool_calls` en `data/jarvis_memory.db` (heard, args, resultado, estado, revisiones antes/después, ms; 500 caracteres máx.).
- `executor.py` (`UniGymTools`): une todo; lectura libre, escritura solo vía `repo.write`.
- Pruebas: 139 nuevas en total en la fase; las de 6.4 incluyen el flujo del hito («agrega una tarea de EDA2 para el viernes» → propuesta → sí →
  tarea con fecha 2026-11-27, copia previa, registro), cancelaciones, sí tardío, tarea borrada entre pregunta y sí, y cambios de la web que se conservan.

## 6.5 Integración con el LLM y la voz
```
frase -> (¿escritura pendiente? "sí"/"no" se resuelven en Python, sin LLM) -> comandos de memoria -> selector top-4
      -> llm.stream_events(messages, tools=...)  -> texto: se habla como siempre
                                                  -> ToolCall: Python ejecuta y responde (lectura) o propone (escritura)
```
- `llm/`: `ToolCall` y `stream_events(messages, tools)`; el backend de Ollama envía `tools` y devuelve texto o llamadas con streaming (la charla
  normal conserva su latencia: el texto sigue saliendo por trozos). Sin herramientas, `stream` es igual que antes.
- `tools/runtime.py` (`ToolRuntime`, `load_tools`): selector + guía del prompt (solo si hay herramientas ofrecidas) + ejecutor. Compartido por
  `core/voice_loop.py` (Turn `kind="tool"`), `chat.py` y `voice_chat.py` (`--sin-herramientas` las desactiva; `--debug` muestra ofrecidas y ms).
- Orden de cada frase: eco/ruido → salir/reiniciar → **confirmación pendiente** → memoria → LLM con herramientas. Una frase que no es sí/no descarta
  la propuesta, dice «Descarté lo que estaba pendiente.» y se atiende normal.
- Un turno de herramienta guarda en el historial solo el texto final (sin mensajes de herramienta: ahorra contexto) y no se registra como recuerdo.
- **Bug encontrado con las pruebas:** `is_noise` descartaba «sí» y «no» (menos de 3 letras), así que la confirmación por voz nunca habría llegado.
  Corregido (`SHORT_ANSWERS`).
- **Diálogo guiado (probando con Dario):** si falta un dato, JARVIS pregunta y la frase siguiente lo rellena sin pasar por el LLM:
  `crear tarea` → «¿Qué tarea agrego?» → «¿Es de algún curso? ... o «ninguno»» → «¿Para qué día es? ... o «sin fecha»» → confirmación → «sí».
  Con todo dicho («tarea de EDA2 para el viernes») no pregunta nada. Cancelar («cancela», «olvídalo», «mejor no»), una pregunta u otra orden
  descartan; fecha ilegible o pasada se repregunta (3 intentos). TTL 60 s. `fecha` de `crear_tarea` ya no dice «si no lo dicen, hoy».
- **Arreglos tras la 1.ª prueba real de Dario (24-nov):** (1) el LLM pasó casi toda la frase como `curso` → `names.clean_course_query` quita el relleno
  («agrega una tarea para el curso redes» → «redes»); (2) el LLM inventó el título «estudiar» → `title_is_grounded`: un título que Dario no dijo se
  pregunta; (3) al confirmar, «no, mejor para el sábado» / «sí, estudiaré el sábado» cancelaba → ahora una fecha dentro de la frase corrige
  la fecha y vuelve a proponer (`find_date_phrase`; solo `crear_tarea`; un «no» a secas sigue cancelando); (4) bug: la sigla «rc» del curso se
  «encontraba» dentro de «parcial» y no se anteponía el curso al título → las siglas solo cuentan como palabra entera. 255 pruebas.
- **Bug de la 2.ª prueba real (24-nov): el LLM imitaba el historial.** Guardábamos «¿Marco como hecha «EDA2»?» como mensaje del asistente; en el
  turno siguiente el LLM lo copió como TEXTO, dejó de llamar a la herramienta y llegó a decir «se marca como listo» sin guardar nada (la web no
  cambió). Arreglo: `ToolRuntime.remember` — una lectura se guarda como llamada + resultado en formato nativo (`tool_calls` / rol `tool`);
  preguntas, confirmaciones, errores, rutinas y el diálogo guiado NO entran al historial del LLM (lo lleva Python). Red de seguridad: si el LLM
  responde con texto que dice que guardó/marcó algo sin llamar a la herramienta, se añade «Ojo: no guardé nada. Pídemelo de nuevo.». 266 pruebas.
  Lección: lo que el LLM ve en el historial es lo que imita; solo debe ver lo que queremos que repita.
- **Herramienta `temas_examen`** (lectura; pedida por Dario): dice los temas (`topics`) o el plan de estudio (`guide`) de una evaluación: la próxima
  (por defecto), la última del curso (`cual=ultimo`, p. ej. el examen final, aunque aún no se rinda) o una por nombre/curso. «Examen»,
  «proyecto», «práctica»... filtran por tipo (1.ª versión devolvía el proyecto cuando Dario pedía el examen; `names.compact` quita «examen»
  como palabra vacía: no usarlo para detectar el tipo). Si el LLM omite el tipo, se repara con lo que se oyó. Dice hasta 8 temas y avisa de los demás. 14 herramientas.
  El presupuesto de tokens de las 4 más caras sube de 650 a 700.
- **`temas_examen` interactiva (3.ª ronda con Dario):** EDA2 no tiene «exámenes» sino ejercicios individuales y un proyecto, y el LLM cambió
  «tercer ejercicio individual» por `evaluacion: "examen"`. Ahora `eval_spec` lee tipo, número/ordinal y «último» de lo que DIJO Dario (manda sobre
  los argumentos del LLM; el «2» de «estructura de datos 2» no cuenta como evaluación 2). Si lo pedido no existe no responde solo «no hay»:
  lista las evaluaciones del curso y pregunta «¿de cuál?»; la respuesta («el tercero», «el proyecto») se resuelve en Python (mismo mecanismo de
  dato pendiente que crear_tarea, ahora también para lecturas). Ofrece el plan si lo hay y conserva «plan» tras la repregunta.
- **4.ª ronda (24-nov), «que aprenda, no repetirle»:** (1) herramienta nueva `evaluaciones_proximas` (15 herramientas); (2) `_reroute`: si lo
  dicho contradice la herramienta elegida se corrige (p. ej. «las evaluaciones que se acercan» con `nota_necesaria` y meta 14 inventada;
  «el ejercicio individual 3» con `consultar_agenda`); (3) contexto de la charla: `last_course` rellena el curso en seguimientos; las repreguntas
  de lectura quedan en el historial en formato nativo para que el LLM sepa de qué se hablaba; (4) oferta «¿Quieres los temas de alguna?»: un «sí»
  o «el ejercicio 3» se resuelve sin LLM, cualquier otra frase la descarta; (5) `plan` lo decide lo dicho («plan», «estudiar», «preparar»), no
  el LLM; (6) si el LLM no dice nada, «No te entendí bien. ¿Me lo repites?»; (7) los ítems ya no se leen con «.;».
- **5.ª ronda:** el LLM inventaba («tu promedio sería 15», un examen de Programación Web ya rendido) porque no había herramienta para eso.
  Nueva `simular_notas` (16 herramientas): lee de lo dicho pares «15 en el proyecto», «doce en todo lo que falta», con la fórmula de la web;
  si quedan evaluaciones sin suponer, dice qué nota necesitaría en ellas; si nombra una ya calificada usa la nota real. `notas_curso` ahora
  dice las notas ya sacadas, lo que falta y lo necesario para aprobar. Persona: no inventar notas/fechas/evaluaciones y no empezar cada
  respuesta con «Dario.».
- **6.ª ronda:** tras «¿cómo voy en EDA2?», Dario dijo «progra web» y el LLM inventó notas copiando el formato del resultado anterior
  (no había herramienta ofrecida para una frase tan corta). (1) Seguimiento en Python: una frase corta que es casi solo el nombre de un curso,
  justo después de una lectura por curso, repite esa lectura con el otro curso sin LLM («redes», «¿y en procesos?»); (2) aviso
  «Ojo: eso no lo saqué de tus datos...» si el LLM da notas/promedios con números sin usar herramienta.
- **7.ª ronda («cursos nuevos se deben asociar solos»):** «¿cómo voy en iot?» y «progra web» seguían saliendo inventados: «iot» no coincidía
  con «Internet de las Cosas» y el LLM no usó herramienta. (1) Las preguntas de notas («cómo voy / qué tal voy / mis notas / mi promedio» + curso)
  se resuelven en Python sin LLM; «mis notas» a secas = todos los cursos. (2) `resolve_course` es genérico para cualquier curso que se agregue:
  coincidencia por letras/siglas → abreviaturas habladas (`COURSE_ABBREVIATIONS`: iot, progra, bd, ia, finanzas...) → parecido semántico con
  bge-m3 contra los nombres reales (umbral 0.60 y margen 0.06 sin medir aún; si no hay margen, pregunta).
- **8.ª ronda:** tras «redes» (resuelto sin LLM), «si saco 15 en el último examen» se calculó con Internet de las Cosas: el LLM sacó el
  curso de una charla vieja porque los turnos resueltos en Python no entraban a su historial. (1) Ahora sí entran (formato nativo);
  (2) si el LLM pasa un curso que Dario no nombró, manda el curso de la charla; (3) «el último (examen)» en simulaciones = el último por fecha;
  (4) «en el último de redes 15» (nota al final) también se entiende; (5) en la repregunta, «de redes» corrige el curso y lo dicho se suma.
- **9.ª ronda («mucho texto, más comprimido»):** temas: 4 resumidos (`brief`: hasta «:», «(» o 9 palabras) + «Y N más, ¿te los digo?»;
  plan: 3 pasos resumidos + «¿sigo?»; un «sí» da el resto (oferta). Persona: 1-3 frases cortas, sin listas ni negritas. La voz ya no lee viñetas.
- **10.ª ronda:** (1) «marca X como lista/hecha», «marcar tarea como lista», «completa la tarea X» → `completar_tarea` en Python (el LLM
  escribía «¿Marcar como listo...?» como texto y nunca llamaba a la herramienta); sin título pregunta listando las pendientes. (2) Las
  respuestas con aviso («no guardé nada») ya no entran al historial (el LLM repetía su propio aviso). Una pregunta de confirmación escrita
  por el LLM también se avisa. (3) El curso de contexto se olvida al cambiar de tema (turno sin herramienta, agenda, tareas...);
  «qué tareas tengo» ya no hereda el último curso.
- **11.ª ronda:** (1) «ya di el último examen, saqué 20» (pasado, no «si saco») → propuesta de `registrar_nota` en Python; «el último examen»
  se entiende también al registrar (`find_eval`). (2) El aviso de datos inventados no salta si los números los dijo Dario. (3) Herramienta nueva
  `record_academico` (17): PPA, último PPP, racha, ranking y a dónde iría el PPA con el promedio actual del ciclo (fórmula `ppaWith` de la web);
  «récord», «PPA», «ponderado», «cómo voy en la carrera» se resuelven sin LLM.
- **12.ª ronda:** `record_academico` responde por ciclo o nivel («nivel 1», «el segundo ciclo», «2025-2», con comparación con el anterior) o
  todos («todos los ciclos»); los seguimientos tras hablar del récord se resuelven sin LLM (el LLM decía «no tengo datos de ciclos anteriores»).
- **13.ª ronda:** voz: decimales hablados («17,4» → «diecisiete punto cuatro»), `tts.speed` 1.1 → 1.3, persona con más energía y sin ofrecer
  temas no pedidos. El aviso de datos inventados ya no salta si los números vienen de un resultado de herramienta reciente.
- **14.ª ronda («que analice, no solo números»):** `record_academico` habla en simple («promedio del ciclo» / «promedio acumulado» en vez de
  PPP/PPA), da los cambios en por ciento, compara dos niveles («nivel 4 y 5», «cuarto y quinto ciclo»: cambio, carga de créditos y una
  conclusión) y redondea a un decimal.
- `reply_mode: llm` aún no está implementado (solo plantilla).
- Si no existe `data/unigym.json` o fallan los embeddings, JARVIS sigue sin herramientas y avisa.

## Herramienta 2: «Jarvis, estoy en casa»
- `tools/routines.py` + `routines.casa` en `config.yaml`. Es una **rutina fija en Python** (no pasa por el LLM, como los comandos de memoria):
  instantánea y sin falsos positivos por parecido. Solo activan frases completas («estoy en casa», «ya llegué», con o sin «Jarvis»);
  «estoy en casa de mi tío», «¿estoy en casa?» o «no estoy en casa» no.
- Dice «Bienvenido, señor.» + la tarea pendiente más urgente de UniGym (si hay) + una frase motivadora (lista propia, sin atribuciones inventadas).
- **Después de hablar** abre el video en Firefox (`-new-window`, argumentos en lista, sin shell). La URL sale solo de `config.yaml` y debe ser
  `https` de `allowed_hosts` (youtube.com, youtu.be); si no, no se abre. Sin Firefox usa el navegador predeterminado.
- Registro en `tool_calls` (`said`, `opened`, `rejected`, `error`).
- Aviso: con el video sonando, el micrófono puede transcribir la música (el anti-eco solo conoce lo que dice JARVIS).

### Ronda 15 (24-nov): lista de cursos y siglas mal oídas

- Caso real: con 7 cursos registrados, el LLM respondió "estás inscrito en…" con solo 4 cursos y se inventó uno. "El examen del curso aéa" / "AAA" no encontraba el curso **AEA**.
- Nueva herramienta de lectura `lista_cursos` (18 en total). Lee los cursos de `unigym.json` y nunca usa la memoria del LLM. Frases como "qué cursos tengo", "todos los cursos en los que estoy inscrito" o "la lista de cursos registrados" se resuelven en Python (`_direct_courses`) sin pasar por el LLM. No se activa si la frase habla de notas, tareas, exámenes o de crear algo.
- "Examen/temas … del curso X": el curso se toma de lo que va después de "curso" (se quita "llamado") y va directo a `temas_examen`.
- `names._close_short`: si una sigla de 3–4 letras está a **una** letra de un solo curso ("aaa" → "AEA"), se toma ese curso. Si hay dos posibles, se pregunta.
- Si Dario cambia de tema cuando JARVIS esperaba una respuesta ("¿de cuál?"), la nueva frase igual pasa por las rutas de Python antes que por el LLM. Si lo que se descarta es una escritura, se avisa ("Descarté…").
- La guía del LLM ahora dice: "¿Qué cursos tengo?" es `lista_cursos` (nunca nombres cursos de memoria).
- Pruebas: 337 OK (semillas 0 y 21). Las frases reales se probaron contra una copia de los datos.

### Ronda 16 (24-nov): bench con 18 herramientas y frases de hábito

- Bench con 18 herramientas (`bench_tools.py --repeats 2`, qwen3:8b), vuelta 1:
  - Primera pasada: N=3: 79/90, N=5: 77/90, N=8: 80/90, N=13: 78/90 (antes ~97 % / 96 % / 96 % / 92 %). Falsos positivos 0/24 en todas. Latencia mediana 0.47–0.98 s.
  - Causa: la línea de la personalidad "Responde solo a lo que Dario pregunta" hacía que el LLM no llamara herramientas cuando Dario *cuenta* algo ("tomé un vaso de agua", "dormí siete horas"). Corregida ("Atiende lo que Dario te pide o te cuenta") y la guía ahora dice que registre lo que Dario cuenta.
  - Segunda pasada: N=3: 84/90 (93 %), N=5: 83/90, N=8: 82/90, N=13: 80/90 (89 %). Se arreglaron las frases de registro diario.
  - Fallos que quedan: "ya terminé el informe", "ya hice el hábito de leer", "ya cumplí con el hábito de ejercicio" (el LLM felicita en texto), "léeme mis notas del ciclo" (da récord académico; ambigua) y, solo con 8-13 herramientas, "¿ya cumplí mi meta de pasos?".
- Hábitos en Python (`_direct_habit`): "ya hice el hábito de leer", "marca mi hábito de agua", "marca el hábito de X como hecho" van directo a `marcar_habito` (con confirmación), sin LLM. Se necesita la palabra "hábito" y no se activa con preguntas. "Ya terminé el informe" ya iba por Python (`_direct_done`).
- Lo que el bench mide es el LLM solo; en el uso real esas frases ya no dependen de él. Falta repetir el bench solo si cambian las descripciones.
- Pruebas: 339 OK (semillas 0 y 21).

### Ronda 17 (24-nov): primera prueba del hito 6.6 con chat real

- Resultado: "agrega una tarea EDA2 para el viernes" → "sí" → "Listo, tarea agregada" funcionó (1.er evento 1.3 s). Lista de cursos y hábito por frase directa también.
- Fallos corregidos:
  1. «marca la tarea EDA2 ford como hecha» no encontraba «Estructura de Datos II: ford»: `resolve_task` ahora reintenta expandiendo siglas de cursos (`eda2` → `estructura de datos 2`).
  2. Tras un «¿cuál?», la frase entera «marca la tarea ford como hecha» se tomaba como título. Ahora, si la respuesta es una orden completa de completar, se interpreta como orden (`_direct_done` dentro de `_answer_slot`).
  3. «comparar nivel 4 y 5» se iba al LLM («no tengo acceso a datos»): ahora «comparar» y «académico» también activan el récord. «nivel de récord académico 4 y 5» ya entiende los niveles aunque haya palabras entre «nivel» y los números.
- La lista de cursos dijo 6 cursos (antes 7): el curso de prueba AEA ya no estaba en los datos; no es un fallo.
- Memoria: el bloque inyectado traía frases como «Dario quiere que JARVIS calcule el impacto de una nota de 15...» (instrucciones extraídas de charlas de prueba). Limpiar con `/memoria` y `/olvidar N` en chat.py.
- Pruebas: 342 OK (semillas 0 y 21).

## Pendiente
- 6.6 hito de punta a punta, medidas finales aquí y `claude/estado-fases.md` en el proyecto.
