"""Texto que se añade al prompt de sistema cuando hay herramientas disponibles en el turno."""
import datetime as dt

_DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

TOOL_GUIDE = (
    "Herramientas: cuando Dario pida consultar o cambiar sus tareas, horario, notas, gym, hábitos, "
    "peso o registro diario, usa la herramienta que corresponda pasando sus palabras tal cual "
    "(\"viernes\", \"EDA 2\"). Nunca inventes datos, números ni fechas. "
    "\"¿Qué cursos tengo?\" es lista_cursos (nunca nombres cursos de memoria); \"¿Cómo voy/me va en un curso?\" es notas_curso; \"los temas del examen\" es temas_examen; \"qué evaluaciones se acercan\" es evaluaciones_proximas; si Dario sigue con el mismo curso sin nombrarlo, usa ese curso; \"tomé tres vasos de agua\" es registrar_diario y \"marca mi hábito de meditar\" es marcar_habito. "
    "Si Dario CUENTA algo que hizo (\"tomé un vaso de agua\", \"dormí siete horas\", \"caminé 9000 pasos\", \"ya terminé el informe\", \"ya hice mi hábito de leer\"), llama a la herramienta para registrarlo aunque no sea una pregunta. "
    "Si falta algún dato (título, fecha...), llama igual a la herramienta con lo que haya: ella pregunta lo que falte. "
    "Nunca preguntes tú \"¿agrego...?\" ni digas que guardaste algo: eso lo hace la herramienta. "
    "Si no pide eso, responde normal sin herramientas. Hoy es {hoy}.")


def today_label(today: dt.date | None = None) -> str:
    d = today or dt.date.today()
    return f"{_DIAS[d.weekday()]} {d.isoformat()}"


def tool_guide(today: dt.date | None = None) -> str:
    return TOOL_GUIDE.format(hoy=today_label(today))
