"""Catálogo de herramientas de UniGym para el LLM (solo datos; los manejadores van aparte).

Reglas de diseño (docs/fase6.md):
- Descripciones y parámetros mínimos: cada esquema cuesta tokens del contexto de 4096.
- El LLM pasa argumentos CRUDOS en texto ("viernes", "EDA 2", "70.5"). Python resuelve
  fechas, cursos y rangos; el LLM nunca calcula una fecha ni inventa un curso.
- Las de escritura (level=WRITE) nunca guardan solas: preparan una propuesta que Dario
  confirma por voz.
"""
from ..schema import obj, num, text
from ..base import READ, WRITE, ToolSpec

_FECHA = text("Día: hoy, mañana, un día de la semana o fecha. Si no lo dicen, hoy.")
_CURSO = text("Nombre del curso tal como lo dijo Dario (p. ej. EDA2, redes).")

SPECS: tuple[ToolSpec, ...] = (
    # ---------------------------------- lectura ----------------------------------
    ToolSpec(
        "consultar_agenda",
        "Dice qué clases y tareas tiene Dario en un día.",
        obj({"fecha": _FECHA}),
        ("¿qué tengo hoy?", "¿qué tengo mañana?", "¿qué clases tengo el jueves?", "dime mi agenda de hoy"),
        READ),
    ToolSpec(
        "tareas_pendientes",
        "Lista las tareas pendientes de Dario, las más urgentes primero.",
        obj({"curso": _CURSO}),
        ("¿qué tareas me faltan?", "¿qué tareas pendientes tengo de EDA2?", "¿qué se me vence pronto?"),
        READ),
    ToolSpec(
        "rutina_gym",
        "Dice qué rutina de gym le toca a Dario en un día y sus ejercicios.",
        obj({"fecha": _FECHA}),
        ("¿qué rutina de gym me toca hoy?", "¿qué entreno mañana?", "¿toca gym el sábado?"),
        READ),
    ToolSpec(
        "notas_curso",
        "Dice cómo va Dario en un curso (notas que ya sacó, promedio y lo que falta), o en todos si no nombra uno.",
        obj({"curso": _CURSO}),
        ("¿cómo voy en redes?", "¿qué tal voy en estadística?", "¿cuál es mi promedio en EDA2?", "dime mis notas"),
        READ),
    ToolSpec(
        "nota_necesaria",
        "Calcula qué nota necesita Dario en lo que falta de un curso para aprobar o para llegar a una meta.",
        obj({"curso": _CURSO, "meta": num("Nota final deseada (0-20). Omítela para aprobar.")}, ["curso"]),
        ("¿cuánto necesito en la próxima evaluación de redes para llegar a 15?",
         "¿qué nota necesito en EDA2 para aprobar?", "¿cuánto me falta para pasar el curso de estadística?", "¿cuánto debo sacar en gestión financiera para un 14?"),
        READ),
    ToolSpec(
        "progreso_diario",
        "Dice pasos, sueño y agua registrados en un día frente a las metas.",
        obj({"fecha": _FECHA}),
        ("¿cuánta agua llevo hoy?", "¿cuántos pasos he dado?", "¿cuánto dormí anoche?"),
        READ),
    ToolSpec(
        "estado_habitos",
        "Dice qué hábitos ya marcó Dario hoy y cuáles faltan, con sus rachas.",
        obj({}),
        ("¿qué hábitos me faltan hoy?", "¿cómo van mis hábitos?", "¿cuál es mi racha?"),
        READ),
    ToolSpec(
        "temas_examen",
        "Dice los temas (o el plan de estudio) de un examen: el próximo, el último rendido o uno por nombre.",
        obj({"curso": _CURSO,
             "evaluacion": text("examen, proyecto, práctica o un nombre (EVA 2...)."),
             "cual": {"type": "string", "enum": ["proximo", "ultimo"], "description": "ultimo = la última del curso"},
             "plan": {"type": "boolean", "description": "plan de estudio"}}),
        ("¿qué temas entran en el próximo examen?", "dime los temas del último examen", "¿qué temas entran en el examen de redes?",
         "dame el plan de estudio del parcial de EDA2"),
        READ),
    ToolSpec(
        "evaluaciones_proximas",
        "Lista las evaluaciones que se acercan (fecha y peso), de un curso o de todos.",
        obj({"curso": _CURSO}),
        ("¿qué evaluaciones se acercan?", "¿cuáles son mis próximas evaluaciones de EDA2?", "¿qué exámenes tengo pronto?"),
        READ),
    ToolSpec(
        "simular_notas",
        "Calcula con qué nota terminaría Dario un curso si saca ciertas notas en las evaluaciones que faltan.",
        obj({"curso": _CURSO, "supuestos": text("Lo que supone, tal cual: «15 en el proyecto y 15 en el ejercicio 3».")}),
        ("¿qué pasa si saco 15 en el proyecto y 15 en el ejercicio 3?", "si saco 12 en el final, ¿apruebo redes?",
         "predice mi nota de EDA2 si saco 14 en todo lo que falta"),
        READ),
    ToolSpec(
        "record_academico",
        "Analiza el récord académico de Dario: PPA, PPP de cada ciclo o nivel, tendencia y a dónde iría su PPA con este ciclo.",
        obj({"ciclo": text("Ciclo o nivel si lo dijo: «nivel 1», «2025-2», «todos».")}),
        ("¿cómo va mi récord académico?", "¿cuál es mi PPA?", "¿cómo voy en la carrera?", "¿qué tal mi ponderado?"),
        READ),
    ToolSpec(
        "lista_cursos",
        "Dice los cursos que Dario tiene registrados en UniGym.",
        obj({}),
        ("¿qué cursos tengo registrados?", "dime mis cursos", "¿cuáles son todos mis cursos?"),
        READ),
    # --------------------------------- escritura ---------------------------------
    ToolSpec(
        "crear_tarea",
        "Agrega una tarea nueva a la lista de Dario (se confirma antes de guardar).",
        obj({"titulo": text("Qué hay que hacer."), "curso": _CURSO,
             "fecha": text("Fecha límite solo si Dario la dijo (mañana, un día de la semana o fecha). Si no, omítela."),
             "prioridad": {"type": "string", "enum": ["alta", "media", "baja"]}}),
        ("agrega una tarea de EDA2 para el viernes", "anota que tengo que entregar el informe de redes el lunes",
         "crea una tarea: estudiar para el examen de gestión financiera"),
        WRITE),
    ToolSpec(
        "completar_tarea",
        "Marca una tarea como hecha (se confirma antes de guardar).",
        obj({"titulo": text("Nombre o parte del nombre de la tarea.")}, ["titulo"]),
        ("marca como hecha la tarea de EDA2", "ya terminé el informe de redes", "completa la tarea de estudiar"),
        WRITE),
    ToolSpec(
        "marcar_habito",
        "Marca un hábito de la lista como cumplido; solo si Dario dice \"hábito\" o nombra uno (se confirma antes de guardar).",
        obj({"habito": text("Nombre del hábito, p. ej. agua, leer."), "fecha": _FECHA}, ["habito"]),
        ("marca mi hábito de agua", "ya hice el hábito de leer", "marca ejercicio de hoy"),
        WRITE),
    ToolSpec(
        "registrar_peso",
        "Registra el peso corporal de Dario en kilos (se confirma antes de guardar).",
        obj({"kg": num("Peso en kilogramos."), "fecha": _FECHA}, ["kg"]),
        ("registra que peso 70.5 kg", "hoy pesé 71 kilos", "anota mi peso: 70 con 8"),
        WRITE),
    ToolSpec(
        "registrar_diario",
        "Registra o suma pasos, horas de sueño o vasos de agua tomados (se confirma antes de guardar).",
        obj({"campo": {"type": "string", "enum": ["pasos", "sueno", "agua"]},
             "valor": num("Cantidad."), "modo": {"type": "string", "enum": ["sumar", "fijar"]},
             "fecha": _FECHA}, ["campo", "valor"]),
        ("tomé un vaso de agua", "dormí 7 horas y media", "hoy di 6500 pasos"),
        WRITE),
    ToolSpec(
        "registrar_nota",
        "Anota la nota que sacó Dario en una evaluación de un curso (se confirma antes de guardar).",
        obj({"curso": _CURSO, "evaluacion": text("Nombre de la evaluación, p. ej. EVA 2, parcial."),
             "nota": num("Nota sobre 20.")}, ["curso", "evaluacion", "nota"]),
        ("saqué 15 en el parcial de redes", "anota mi nota de la EVA 2 de EDA2: 17", "en gestión financiera me sacaron 13"),
        WRITE),
)


def spec_by_name(name: str) -> ToolSpec | None:
    return next((s for s in SPECS if s.name == name), None)
