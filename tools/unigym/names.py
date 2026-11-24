"""Búsqueda tolerante de cursos, evaluaciones, tareas y hábitos por lo que oyó el STT.

"EDA 2", "eda dos", "EDA2" y "Estructura de Datos y Algoritmos II" deben dar el mismo curso.
Si hay dos candidatos parecidos NO se adivina: se devuelve la lista para preguntar.
"""
import difflib
import re
from dataclasses import dataclass, field

from memory.text import strip_accents

_STOP = {"de", "del", "la", "las", "el", "los", "y", "e", "en", "a", "al", "un", "una", "para", "mi", "mis",
         "curso", "tarea", "habito", "evaluacion", "examen"}
_NUM_WORDS = {"uno": "1", "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5", "seis": "6",
              "i": "1", "ii": "2", "iii": "3", "iv": "4", "primer": "1", "primera": "1", "segunda": "2"}
MIN_SCORE = 0.72
AMBIGUITY_MARGIN = 0.08


def tokens(text: str) -> list[str]:
    s = strip_accents(str(text)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", s)   # "eda2" -> "eda 2"
    return [_NUM_WORDS.get(t, t) for t in s.split()]


def compact(text: str) -> str:
    return "".join(t for t in tokens(text) if t not in _STOP)


def acronym(text: str) -> str:
    """Iniciales de las palabras significativas + números finales: 'Estructura de Datos y Algoritmos 2' -> 'eda2'."""
    toks = [t for t in tokens(text) if t not in _STOP]
    return "".join(t if t.isdigit() else t[0] for t in toks)


@dataclass
class Match:
    item: object
    score: float


@dataclass
class Resolution:
    item: object | None = None
    candidates: list = field(default_factory=list)   # Match, de mejor a peor
    ambiguous: bool = False

    @property
    def found(self) -> bool:
        return self.item is not None


def _score(query: str, labels: list[str]) -> float:
    q_tokens = [t for t in tokens(query) if t not in _STOP] or tokens(query)
    q = "".join(q_tokens)
    if not q:
        return 0.0
    best = 0.0
    for label in labels:
        if not label:
            continue
        l_tokens = [t for t in tokens(label) if t not in _STOP] or tokens(label)
        l = "".join(l_tokens)
        if not l:
            continue
        if q == l or q == acronym(label):
            s = 1.0
        elif set(q_tokens) <= set(l_tokens) and len(q) >= 3:
            s = 0.9
        elif len(q) >= 3 and (l.startswith(q) or q in l):
            s = 0.85
        else:
            s = difflib.SequenceMatcher(None, q, l).ratio()
            s = max(s, difflib.SequenceMatcher(None, q, acronym(label)).ratio() * 0.95)
        best = max(best, s)
    return best


def resolve(query, items, labels) -> Resolution:
    """`labels(item)` -> lista de textos con los que puede llamarse (nombre, id, alias...)."""
    if not query or not str(query).strip():
        return Resolution()
    scored = sorted((Match(it, _score(query, labels(it))) for it in items), key=lambda m: -m.score)
    good = [m for m in scored if m.score >= MIN_SCORE]
    if not good:
        return Resolution(candidates=scored[:3])
    if len(good) > 1 and good[0].score - good[1].score < AMBIGUITY_MARGIN and good[0].score < 1.0:
        return Resolution(candidates=good[:3], ambiguous=True)
    return Resolution(item=good[0].item, candidates=good[:3])


def course_labels(course: dict) -> list[str]:
    cid = str(course.get("id", ""))
    return [course.get("name", ""), cid.split("_", 1)[-1] if "_" in cid else cid]


_COURSE_FILLER = {"curso", "cursos", "materia", "clase", "asignatura", "agrega", "agregar", "agregame", "anota", "anotar",
                  "crea", "crear", "tarea", "tareas", "nueva", "nuevo", "pendiente", "de", "del", "el", "la", "para", "una",
                  "un", "por", "favor", "jarvis", "que", "es"}


def clean_course_query(query) -> str:
    """«agrega una tarea para el curso redes» -> «redes» (el LLM a veces pasa casi toda la frase como curso)."""
    toks = [t for t in tokens(str(query or "")) if t not in _COURSE_FILLER]
    return " ".join(toks)


# Abreviaturas que se usan al hablar de cursos. Sirven para CUALQUIER curso que se agregue (no hay que registrar nada):
# «iot» se expande a «internet de las cosas» y luego se compara con los nombres reales de los cursos.
COURSE_ABBREVIATIONS = {
    "iot": "internet de las cosas", "progra": "programacion", "prog": "programacion", "bd": "base de datos",
    "bdd": "base de datos", "ia": "inteligencia artificial", "so": "sistemas operativos", "eda": "estructura de datos",
    "poo": "programacion orientada a objetos", "gestfin": "gestion financiera", "finanzas": "financiera",
    "mate": "matematica", "mates": "matematica", "estadis": "estadistica", "conta": "contabilidad",
    "eco": "economia", "fisica": "fisica", "ciber": "ciberseguridad", "seguridad": "seguridad", "ingles": "ingles",
    "algo": "algoritmos", "procesos": "procesos", "ipn": "ingenieria de procesos de negocio", "redes": "redes",
}

# Último recurso (lo conecta tools/runtime.py con bge-m3): parecido semántico entre lo dicho y los nombres de los cursos.
# fn(query, [nombres]) -> (índice, score, segundo_score) o None.
SEMANTIC_FALLBACK = None
SEMANTIC_MIN, SEMANTIC_MARGIN = 0.60, 0.06


def expand_abbreviations(query) -> str:
    return " ".join(COURSE_ABBREVIATIONS.get(t, t) for t in tokens(str(query or "")))


def _close_short(query: str, courses: list):
    """Siglas cortas (3-4 letras) que el STT confunde: «aaa»/«aéa» -> «AEA». Solo si hay UN curso a una letra de distancia."""
    q = compact(query)
    if not (3 <= len(q) <= 4) or not q.isalpha():
        return None
    hits = []
    for c in courses:
        for label in course_labels(c):
            l = compact(label)
            if len(l) == len(q) and sum(a != b for a, b in zip(q, l)) == 1:
                hits.append(c)
                break
    return hits[0] if len(hits) == 1 else None


def resolve_course(query, state: dict) -> Resolution:
    courses = [c for c in state.get("courses", []) if isinstance(c, dict)]
    r = resolve(query, courses, course_labels)
    if r.found or r.ambiguous:
        return r
    tried = {" ".join(tokens(str(query or "")))}
    for q in (clean_course_query(query), expand_abbreviations(clean_course_query(query))):
        if q and q not in tried:
            tried.add(q)
            r2 = resolve(q, courses, course_labels)
            if r2.found or r2.ambiguous:
                return r2
    close = _close_short(clean_course_query(query), courses)       # «AAA» oído por «AEA»: una letra de diferencia
    if close is not None:
        return Resolution(item=close, candidates=[Match(close, 0.8)])
    if SEMANTIC_FALLBACK and courses and clean_course_query(query):
        try:
            hit = SEMANTIC_FALLBACK(expand_abbreviations(clean_course_query(query)), [c.get("name", "") for c in courses])
        except Exception:
            hit = None
        if hit:
            i, score, second = hit
            if score >= SEMANTIC_MIN and score - second >= SEMANTIC_MARGIN:
                return Resolution(item=courses[i], candidates=[Match(courses[i], score)])
    return r


def resolve_eval(query, course: dict) -> Resolution:
    return resolve(query, [e for e in course.get("evals", []) if isinstance(e, dict)], lambda e: [e.get("name", "")])


def resolve_task(query, state: dict, only_pending: bool = True) -> Resolution:
    tasks = [t for t in state.get("tasks", []) if isinstance(t, dict) and not (only_pending and t.get("done"))]
    r = resolve(query, tasks, lambda t: [t.get("title", "")])
    if r.found or r.ambiguous:
        return r
    expanded = expand_abbreviations(query)                     # «eda2 ford» -> «estructura de datos 2 ford» («Estructura de Datos II: ford»)
    if expanded and expanded != " ".join(tokens(str(query or ""))):
        r2 = resolve(expanded, tasks, lambda t: [t.get("title", "")])
        if r2.found or r2.ambiguous:
            return r2
    return r


def resolve_habit(query, state: dict) -> Resolution:
    return resolve(query, [h for h in state.get("habits", []) if isinstance(h, dict)], lambda h: [h.get("name", "")])
