"""Ejecutor de herramientas de UniGym (Fase 6.4): lectura libre, escritura con confirmación, copia y registro.

Flujo de una escritura (todo en Python; el LLM solo eligió la herramienta y los argumentos crudos):
    run("crear_tarea", args)  -> valida -> Outcome(kind="confirm", text="¿Agrego la tarea ...?")   [no se guardó nada]
    answer("sí")              -> repo.write(...) con copia previa -> Outcome(kind="done")           [se guardó]
    answer("no" / otra cosa)  -> descarta la propuesta                                              [no se guardó]
Cada paso queda en `tool_calls` (audit.py). El estado del archivo nunca se modifica sin pasar por `repo.write`.
"""
import dataclasses
import datetime as dt
import re
import time
from dataclasses import dataclass

from memory.text import strip_accents

from ..base import READ, WRITE
from .audit import AuditLog
from .confirm import ConfirmationGate, classify_answer
from .writers import is_none_answer
from .dates import find_date_phrase
from .grounding import grounding_problem, title_is_grounded
from .readers import READERS, Result, eval_spec, wants_plan
from .repo import RevisionConflict, UniGymDataError, UniGymRepository
from .specs import spec_by_name
from .writers import Proposal, WRITERS


@dataclass
class Outcome:
    text: str                    # lo que JARVIS debe decir
    kind: str                    # answer | ask | confirm | done | cancelled | error
    tool: str = ""
    ok: bool = True
    consumed: bool = True        # False: la frase NO era una respuesta a la confirmación; hay que procesarla normal
    data: dict | None = None


def _mentions(curso: str, heard: str) -> bool:
    """¿Dario nombró este curso en la frase? (palabras, siglas o abreviaturas)."""
    from . import names
    h = set(names.tokens(names.expand_abbreviations(heard))) | set(names.tokens(heard))
    words = {t for t in names.tokens(curso) if t not in names._STOP and len(t) >= 3}
    hc = names.compact(heard)
    acr = names.acronym(curso)
    return bool(words & h) or (len(acr) >= 2 and acr in {names.compact(t) for t in names.tokens(heard)} | {hc}) \
        or names.compact(curso) in hc


class UniGymTools:
    def __init__(self, repo: UniGymRepository, audit: AuditLog | None = None, today=dt.date.today,
                 gate: ConfirmationGate | None = None, clock=time.perf_counter):
        self.repo, self.audit = repo, audit
        self._today, self._clock = today, clock
        self.gate = gate or ConfirmationGate()
        self.last_tool: str | None = None
        self.last_course: str | None = None   # contexto de la charla: el último curso del que se habló
        self.last_read: tuple | None = None   # (herramienta, args) de la última lectura por curso: «¿y en redes?»
        self.offer: tuple | None = None       # («¿Quieres los temas de alguna?»): herramienta + args si Dario dice «sí»

    # ------------------------------ entrada principal ------------------------------
    @property
    def has_pending(self) -> bool:
        return self.gate.active

    def run(self, name: str, args=None, heard: str = "") -> Outcome:
        t0 = self._clock()
        args = args if isinstance(args, dict) else {}
        name, args = self._reroute(name, args, heard)
        spec = spec_by_name(name)
        if spec is None or (name not in READERS and name not in WRITERS):
            self._log(name, "rejected", heard=heard, args=args, result="herramienta desconocida")
            return Outcome("No tengo esa herramienta.", "error", name, ok=False)
        if self.gate.active and spec.level == WRITE:        # una escritura nueva reemplaza a la pendiente
            old = self.gate.pending.proposal.tool if self.gate.pending else self.gate.slot.tool
            self._log(old, "cancelled", level=WRITE, result="reemplazada por otra orden")
            self.gate.clear()
        problem = grounding_problem(args, heard)
        if problem:                                          # el LLM pasó un número que no se dijo
            self._log(name, "ask", level=spec.level, heard=heard, args=args, result=problem)
            return Outcome(problem, "ask", name, ok=False)
        if name == "crear_tarea" and heard and args.get("titulo") and not title_is_grounded(args.get("titulo"), heard):
            self._log(name, "ask", level=WRITE, heard=heard, args=args, result="título no dicho por Dario: se pregunta")
            args = dict(args, titulo="")                     # el LLM inventó el título: se pregunta en vez de guardarlo
        if name in ("temas_examen", "simular_notas", "record_academico") and heard and not args.get("_todo"):
            args = dict(args, _heard=heard)                  # lo dicho manda sobre el tipo/número que pasó el LLM
        try:
            snap = self.repo.read()
        except UniGymDataError as e:
            self._log(name, "error", level=spec.level, heard=heard, args=args, result=str(e))
            return Outcome(f"No pude leer tus datos de UniGym. {e}", "error", name, ok=False)
        ms = lambda: int((self._clock() - t0) * 1000)  # noqa: E731
        today = self._today()
        if spec.level == READ:
            return self._read_step(name, args, heard, snap, today, ms)
        return self._write_step(name, args, heard, snap, today, ms)

    _FOLLOWS = ("temas_examen", "evaluaciones_proximas", "simular_notas", "notas_curso", "nota_necesaria")
    _COURSE_TOOLS = _FOLLOWS + ("tareas_pendientes",)
    _UPCOMING = re.compile(r"\b(evaluaciones|examenes|ejercicios|practicas)\b.*\b(proxim|acerc|vien|falt|pendient|toca)|"
                           r"\b(proxim\w*|que) (evaluaciones|examenes)\b")

    _SIMULATE = re.compile(r"\b(si|y si|que pasa si|cuanto tendria si)\b.*\b(saco|sacara|sacar|obtengo|me sale|me ponen|tengo)\b.*\d"
                           r"|\b(saco|sacara)\s+\w+\s+(en|de)\b")

    def _reroute(self, name, args, heard):
        """Corrige elecciones del LLM que contradicen lo dicho (casos reales del 24-nov) y aplica el contexto."""
        said = strip_accents(heard or "").lower()
        spec = spec_by_name(name)
        if heard and spec is not None and spec.level == READ:
            kind, n, _ = eval_spec(heard)
            if self._SIMULATE.search(said) and name != "simular_notas":
                name, args = "simular_notas", {k: v for k, v in args.items() if k == "curso"}
            elif self._UPCOMING.search(said) and name != "evaluaciones_proximas":
                name, args = "evaluaciones_proximas", {k: v for k, v in args.items() if k == "curso"}
            elif kind and (n is not None or any(w in said for w in ("tema", "plan", "estudi", "entra", "viene")))\
                    and name in ("consultar_agenda", "tareas_pendientes", "rutina_gym", "nota_necesaria"):
                name, args = "temas_examen", {k: v for k, v in args.items() if k == "curso"}
        if name in self._FOLLOWS and not args.get("curso") and heard and not args.get("_todos"):
            try:                                               # «...en el último de redes»: el curso está en lo dicho
                said_courses = [c.get("name", "") for c in self.repo.read().state.get("courses", [])
                                if isinstance(c, dict) and _mentions(c.get("name", ""), heard)]
            except UniGymDataError:
                said_courses = []
            if len(said_courses) == 1:
                args = dict(args, curso=said_courses[0])
        if name in self._FOLLOWS and not args.get("curso") and self.last_course and not args.get("_todos"):
            kind, n, _ = eval_spec(heard)
            if name != "temas_examen" or kind is not None or n is not None:
                args = dict(args, curso=self.last_course)      # «el ejercicio individual 3» tras hablar de EDA2
        elif name in self._FOLLOWS and args.get("curso") and self.last_course and heard \
                and not _mentions(args["curso"], heard):
            args = dict(args, curso=self.last_course)          # el LLM sacó el curso de una charla vieja; Dario no lo nombró
        if name == "tareas_pendientes" and args.get("curso") and heard and not _mentions(args["curso"], heard):
            args = {k: v for k, v in args.items() if k != "curso"}   # «qué tareas tengo»: todas, no las del último curso
        return name, args

    def _answer_offer(self, text: str):
        """Respuesta a una oferta de JARVIS («¿Quieres los temas de alguna?»): «sí» o «el ejercicio 3». Si no, None."""
        offer, self.offer = self.offer, None
        if offer is None:
            return self._follow_up(text)
        tool, args = offer
        kind, n, _ = eval_spec(text)
        short = len((text or "").split()) <= 8
        if classify_answer(text) == "yes" or (short and (n is not None or kind) and classify_answer(text) != "no"):
            return self.run(tool, dict(args), heard=text)
        return self._follow_up(text)

    _BY_COURSE = ("notas_curso", "nota_necesaria", "evaluaciones_proximas", "tareas_pendientes", "temas_examen")
    _FILL = {"y", "en", "de", "del", "el", "la", "curso", "ahora", "que", "tal", "pues", "jarvis", "con", "para", "sobre",
             "otra", "vez", "tambien", "igual", "mismo", "lo", "a", "ver", "dime", "como", "voy", "va"}

    _NOTAS = re.compile(r"\b(como voy|que tal voy|como me va|como me esta yendo|como estoy|como ando|cuanto llevo|"
                        r"mis notas|mi promedio|las notas|notas de|notas en|promedio de|promedio en|promedio del)\b")

    def forget_context(self) -> None:
        """La charla cambió de tema (turno sin herramienta, «no estamos hablando de un curso»): se olvida el curso."""
        self.last_course, self.last_read, self.offer = None, None, None

    _DONE = re.compile(r"\b(?:marca|marcar|marcame|pon|poner|deja|dejar)\s+como\s+(?:lista|listo|hecha|hecho|completada|"
                       r"completado|terminada|terminado)\s+(?P<d>.+)$"
                       r"|\b(?:marca|marcar|marcame|marcala|pon|poner|deja|dejar)\s+(?P<a>.*?)\s*(?:como\s+)?"
                       r"(?:lista|listo|hecha|hecho|completada|completado|terminada|terminado)\b"
                       r"|\b(?:completa|completar|termina|terminar)\s+(?P<b>.*)$"
                       r"|\bya\s+(?:termine|hice|acabe|complete)\s+(?P<c>.*)$")
    _TASK_FILL = {"la", "el", "mi", "una", "tarea", "tareas", "de", "como", "por", "favor", "jarvis", "esa", "esta", "lo", "que"}

    def _direct_done(self, text: str):
        """«marca estudiar redes como lista», «marcar tarea como lista», «ya terminé el informe»: completar_tarea sin LLM.
        Caso real (24-nov): el LLM escribía «¿Marcar como listo...?» como texto y nunca llamaba a la herramienta."""
        said = strip_accents(text or "").lower().strip(" .!?¿¡")
        if "habito" in said:
            return None
        m = self._DONE.search(said)
        if not m:
            return None
        raw = next(g for g in (m.group("d"), m.group("a"), m.group("b"), m.group("c")) if g is not None)
        title = " ".join(t for t in re.split(r"[^a-z0-9]+", raw) if t and t not in self._TASK_FILL)
        if m.group("a") is None and m.group("d") is None and "tarea" not in said:      # «ya terminé de comer»: solo si nombra una tarea real
            try:
                from . import names
                r = names.resolve_task(title, self.repo.read().state)
            except UniGymDataError:
                return None
            if not (r.found or r.ambiguous):
                return None
        return self.run("completar_tarea", {"titulo": title}, heard=text)

    _GOT = re.compile(r"\b(?:saque|me saque|obtuve|me pusieron|me salio|me fue con|tuve|saque un|saque una)\s+(?:un\s+|una\s+)?"
                      r"(\d{1,2}(?:[.,]\d+)?)\b")
    _RECORD = re.compile(r"\b(record|ppa|ponderado|ppp|ranking|decimo superior|quinto superior|tercio superior|"
                         r"como voy en la carrera|como va mi carrera|mi carrera)\b")

    def _direct_grade(self, text: str):
        """«ya di el último examen, saqué 20»: Dario CUENTA una nota (pasado, no «si saco») -> propuesta de registrar_nota.
        Caso real (24-nov): el LLM lo tomó como hipótesis y además salió el aviso de datos inventados."""
        said = strip_accents(text or "").lower()
        m = self._GOT.search(said)
        if not m or re.search(r"\bsi\s+(saco|sacara|sacase)\b", said):
            return None
        rest = (said[:m.start()] + " " + said[m.end():]).strip(" ,.")
        args = {"nota": m.group(1).replace(",", "."), "evaluacion": rest}
        if self.last_course:
            args["curso"] = self.last_course
        try:
            hit = [c.get("name", "") for c in self.repo.read().state.get("courses", [])
                   if isinstance(c, dict) and _mentions(c.get("name", ""), text)]
        except UniGymDataError:
            return None
        if len(hit) == 1:
            args["curso"] = hit[0]
        return self.run("registrar_nota", args, heard=text)

    _HABIT = re.compile(r"\b(?:ya\s+)?(?:hice|cumpli|complete|termine|logre|marca|marcar|marcame|anota|anotar|registra|registrar|pon|poner)\b"
                        r"[^?]*?\bhabito(?:\s+(?:de|del|para))?\s+(?P<h>[a-z0-9 ]+)$")

    def _direct_habit(self, text: str):
        """«ya hice el hábito de leer», «marca mi hábito de agua»: marcar_habito sin LLM.
        Medido en el bench (24-nov): con «ya…» el LLM felicita en texto («¡Bien hecho!») y no llama a la herramienta."""
        said = strip_accents(text or "").lower().strip(" .!¿¡")
        if len(said.split()) > 12 or "?" in (text or ""):
            return None
        m = self._HABIT.search(said)
        if not m:
            return None
        rest = re.sub(r"\s+como\s+(?:hecho|hecha|listo|lista|cumplido|completado|completa)$", "", m.group("h"))
        name = " ".join(t for t in rest.split() if t not in ("hoy", "ya", "por", "favor", "jarvis"))
        if not name:
            return None
        return self.run("marcar_habito", {"habito": name}, heard=text)

    _COURSE_LIST = re.compile(r"\b(?:mis|todos? (?:los|mis)|los|las|cuales|que|cuantos|lista (?:de )?(?:los )?|dime (?:los |mis )?|"
                              r"muestrame (?:los |mis )?)\s*cursos\b|\bcursos (?:registrados|inscritos|matriculados|que tengo|tengo)\b"
                              r"|\b(?:estoy|estas|esta) (?:inscrit[oa]|matriculad[oa])\b")
    _COURSE_OTHER = re.compile(r"\b(nota|notas|promedio|tarea|tareas|examen|examenes|evaluacion|evaluaciones|temas|agenda|record|"
                               r"ppa|ppp|agrega|agregar|crea|crear|nuevo|nueva)\b")
    _EXAM = re.compile(r"\b(examen|examenes|parcial|final|temas)\b")

    def _direct_courses(self, text: str):
        """«¿qué cursos tengo?», «todos los cursos en los que estoy inscrito»: lista REAL de cursos, sin LLM.
        Caso real (24-nov): el LLM contestó «estás inscrito en…» con 4 de los 7 cursos y se inventó uno.
        Y «el examen del curso AEA» (con el STT oyendo «aéa»/«AAA»): el curso se busca en los datos, no en la memoria del LLM."""
        said = strip_accents(text or "").lower().strip(" .!?¿¡")
        if not said or len(said.split()) > 16:
            return None
        if self._COURSE_LIST.search(said) and not self._COURSE_OTHER.search(said):
            return self.run("lista_cursos", {}, heard=text)
        if self._EXAM.search(said) and re.search(r"\bcurso\b", said) and not self._SIMULATE.search(said) \
                and not self._UPCOMING.search(said):
            tail = re.split(r"\bcurso\b", said)[-1]
            try:
                from . import names
                core = names.clean_course_query(re.sub(r"\b(llamado|llamada|que se llama|de nombre)\b", " ", tail))
                r = names.resolve_course(core, self.repo.read().state) if core else None
            except UniGymDataError:
                return None
            if r is not None and r.found:
                return self.run("temas_examen", {"curso": r.item["name"], "_heard": text}, heard=text)
        return None

    def _direct(self, text: str):
        """Preguntas de notas («¿cómo voy en iot?», «mis notas de redes») se resuelven en Python, sin LLM.

        Caso real (24-nov): el LLM respondió «¿cómo voy en iot?» con notas inventadas. Con el curso nombrado (o «mis notas»
        a secas para todos) no hace falta que el LLM elija nada. Si no se reconoce el curso, sigue el camino normal."""
        done = self._direct_done(text) or self._direct_grade(text)
        if done is not None:
            return done
        listed = self._direct_courses(text)
        if listed is not None:
            return listed
        habit = self._direct_habit(text)
        if habit is not None:
            return habit
        said = strip_accents(text or "").lower()
        about_cycle = re.search(r"\b(nivel|ciclo)\s+(\d|uno|dos|tres|cuatro|cinco|seis|primer|segundo|tercer|cuarto|quinto|sexto)"
                                r"|\b(primer|segundo|tercer|cuarto|quinto|sexto)\s+(ciclo|nivel)\b|\b\d{4}\s*-\s*[12]\b"
                                r"|\b(todos los ciclos|cada ciclo|ciclo por ciclo)\b", said)
        if (self._RECORD.search(said) or (about_cycle and (self.last_tool == "record_academico" or "ppa" in said
                                                           or "ppp" in said or "como iba" in said or "como me fue" in said
                                                           or re.search(r"\b(compar\w*|academic\w*)\b", said)))) \
                and len((text or "").split()) <= 14:
            return self.run("record_academico", {"_heard": text}, heard=text)
        said = strip_accents(text or "").lower()
        m = self._NOTAS.search(said)
        if not m or len(said.split()) > 14 or re.search(r"\b(si saco|si sacara|necesito|nota necesaria)\b", said):
            return None
        rest = said[:m.start()] + " " + said[m.end():]
        try:
            from . import names
            state = self.repo.read().state
            core = names.clean_course_query(rest)
            core = " ".join(t for t in core.split() if t not in self._FILL and t not in ("notas", "promedio", "ciclo", "todo", "todos", "cursos"))
            if not core:                                    # «mis notas», «mi promedio del ciclo»: todos los cursos
                return self.run("notas_curso", {"_todos": True}, heard=text)
            r = names.resolve_course(core, state)
        except UniGymDataError:
            return None
        if r.found:
            return self.run("notas_curso", {"curso": r.item["name"]}, heard=text)
        return None

    def _follow_up(self, text: str):
        """«progra web» / «¿y en redes?» justo después de una lectura por curso: la misma consulta con otro curso, sin LLM.

        Caso real (24-nov): tras «¿cómo voy en EDA2?», Dario dijo «progra web» y el LLM inventó las notas imitando el
        formato de la respuesta anterior. Solo se activa con frases cortas que son, casi enteras, el nombre de un curso."""
        direct = self._direct(text)
        if direct is not None or not self.last_read:
            return direct
        words = [t for t in re.split(r"[^a-z0-9]+", strip_accents(text or "").lower()) if t]
        core = [t for t in words if t not in self._FILL]
        if not core or len(core) > 4 or len(words) > 7:
            return None
        try:
            from . import names
            r = names.resolve_course(" ".join(core), self.repo.read().state)
        except UniGymDataError:
            return None
        if not r.found:
            return None
        tool, args = self.last_read
        args = {k: v for k, v in args.items() if k not in ("curso", "_heard", "evaluacion", "cual")}
        return self.run(tool, dict(args, curso=r.item["name"]), heard=text)

    def _read_step(self, name, args, heard, snap, today, ms, attempts: int = 1) -> Outcome:
        r = READERS[name](snap.state, args, today)
        self.offer = None
        self.last_tool = name
        if name not in self._COURSE_TOOLS:                  # agenda, gym, hábitos...: ya no se habla de un curso
            self.last_course, self.last_read = None, None
        if r.ok and (r.data or {}).get("curso"):
            self.last_course = r.data["curso"]
            if name in self._BY_COURSE:
                self.last_read = (name, {k: v for k, v in args.items() if not k.startswith("_")})
        if r.ok and (r.data or {}).get("more"):            # «¿sigo?» / «¿te los digo?» -> «sí» da el resto
            self.offer = (name, dict({k: v for k, v in args.items() if k != "_heard"}, _todo=True,
                                     evaluacion=r.data.get("evaluacion"), curso=r.data.get("curso"),
                                     _heard=("plan" if wants_plan(args.get("_heard")) else "temas")))
        if r.ok and name == "evaluaciones_proximas" and (r.data or {}).get("evaluaciones"):
            self.offer = ("temas_examen", {"curso": r.data["curso"]} if r.data.get("curso") else {})
        self._log(name, "ask" if r.needs else "read", level=READ, heard=heard, args=args, result=r.text,
                  revision_before=snap.revision, duration_ms=ms())
        slot = (r.data or {}).get("slot") if r.needs else None
        if slot:                                            # «¿de cuál?»: la próxima frase responde (sin pasar por el LLM)
            self.gate.set_slot(name, args, slot, heard, attempts)
        return Outcome(r.text, "ask" if r.needs else "answer", name, ok=r.ok, data=r.data)

    def _write_step(self, name, args, heard, snap, today, ms, attempts: int = 1) -> Outcome:
        self.offer, self.last_read = None, None
        """Prepara la escritura: propuesta (espera el sí), pregunta por un dato que falta (espera la respuesta) o nada que hacer."""
        out = WRITERS[name](snap.state, args, today)
        if isinstance(out, Result):                         # falta un dato, o ya estaba hecho: nada que confirmar
            self._log(name, "ask" if out.needs else "read", level=WRITE, heard=heard, args=args, result=out.text,
                      revision_before=snap.revision, duration_ms=ms())
            slot = (out.data or {}).get("slot") if out.needs else None
            if slot:                                        # la próxima frase de Dario responde a esta pregunta
                self.gate.set_slot(name, args, slot, heard, attempts)
            return Outcome(out.text, "ask" if out.needs else "answer", name, ok=out.ok)
        aid = self._log(name, "proposed", level=WRITE, heard=heard, args=args, result=out.question,
                        revision_before=snap.revision, duration_ms=ms())
        self.gate.set(out, heard=heard, revision=snap.revision, audit_id=aid, args=args)
        return Outcome(out.question, "confirm", name, data=out.data)

    # ------------------------------ respuesta a la confirmación ------------------------------
    def answer(self, text: str) -> Outcome | None:
        """Interpreta la frase siguiente a una propuesta o a una pregunta por un dato. None si no hay nada pendiente.

        Si la frase NO era la respuesta (se descarta lo pendiente), igual pasa por las rutas en Python antes que por el LLM.
        Caso real (24-nov): tras «¿de cuál examen?», «¿tienes la lista de cursos?» acabó en el LLM, que inventó la lista."""
        out = self._answer_pending(text)
        if out is None or out.consumed:
            return out
        alt = self._follow_up(text)
        if alt is None:
            return out
        spec = spec_by_name(out.tool or "")
        if spec is not None and spec.level == WRITE:         # se avisa que lo que iba a guardar se descartó
            alt = dataclasses.replace(alt, text=f"{out.text} {alt.text}")
        return alt

    def _answer_pending(self, text: str) -> Outcome | None:
        if self.gate.slot is not None:
            return self._answer_slot(text)
        pend = self.gate.pending
        if pend is None:
            return self._answer_offer(text)
        prop: Proposal = pend.proposal
        expired = self.gate.expired()
        verdict = classify_answer(text)
        fix = find_date_phrase(text, self._today()) if (prop.tool == "crear_tarea" and pend.args is not None and not expired) else None
        if fix:                                              # «no, mejor para el sábado»: corrige la fecha y vuelve a proponer
            return self._correct_date(prop, pend, fix, text)
        self.gate.clear()
        if verdict == "yes" and expired:
            self._log(prop.tool, "expired", level=WRITE, heard=text, result="el sí llegó tarde")
            return Outcome("Pasó mucho tiempo y no guardé nada. Pídemelo de nuevo.", "cancelled", prop.tool)
        if verdict == "yes":
            return self._apply(prop, pend, text)
        if verdict == "no":
            self._log(prop.tool, "cancelled", level=WRITE, heard=text, result="Dario dijo que no")
            return Outcome("Cancelado, no guardé nada.", "cancelled", prop.tool)
        self._log(prop.tool, "cancelled", level=WRITE, heard=text, result="otra frase en lugar de sí/no")
        return Outcome("Descarté lo que estaba pendiente.", "cancelled", prop.tool, consumed=False)

    # ------------------------------ respuesta a una pregunta por un dato ------------------------------
    _CANCEL = {"cancela", "cancelar", "cancelalo", "olvidalo", "detente", "alto", "dejalo", "mejor"}
    _QUESTION = {"que", "cual", "cuales", "cuanto", "cuantos", "como", "cuando", "dime", "muestrame", "dame", "donde"}

    def _answer_slot(self, text: str) -> Outcome:
        from memory.text import strip_accents
        sp = self.gate.slot
        toks = [t.strip(".,;:!?¡¿\"'()") for t in strip_accents(text or "").lower().split()]
        toks = [t for t in toks if t]
        if self.gate.slot_expired():                       # pasó mucho tiempo: se olvida la pregunta y se procesa normal
            self.gate.clear()
            self._log(sp.tool, "expired", level=WRITE, heard=text, result="la respuesta llegó tarde")
            return Outcome("Descarté la tarea que estaba armando.", "cancelled", sp.tool, consumed=False)
        if any(t in self._CANCEL for t in toks) or (sp.slot == "titulo" and classify_answer(text) == "no"):
            self.gate.clear()
            self._log(sp.tool, "cancelled", level=WRITE, heard=text, result="Dario canceló")
            return Outcome("Cancelado, no guardé nada.", "cancelled", sp.tool)
        if not toks or "?" in text or "¿" in text or toks[0] in self._QUESTION:   # es otra orden, no la respuesta
            self.gate.clear()
            self._log(sp.tool, "cancelled", level=WRITE, heard=text, result="otra frase en lugar de la respuesta")
            return Outcome("Descarté lo que estaba armando.", "cancelled", sp.tool, consumed=False)
        if sp.tool == "completar_tarea":                    # «marca la tarea ford como hecha»: es una orden entera, no solo el título
            self.gate.clear()
            whole = self._direct_done(text)
            if whole is not None:
                return whole
            self.gate.set_slot(sp.tool, sp.args, sp.slot, sp.heard, sp.attempts)
        args = dict(sp.args)
        value = text.strip()
        if sp.slot == "curso" and is_none_answer(value):
            args["curso"], args["sin_curso"] = "ninguno", True
        else:
            args[sp.slot] = value
        t0 = self._clock()
        try:
            snap = self.repo.read()
        except UniGymDataError as e:
            self.gate.clear()
            return Outcome(f"No pude leer tus datos de UniGym. {e}", "error", sp.tool, ok=False)
        again = sp.attempts + 1
        self.gate.clear()
        is_read = sp.tool in READERS
        if is_read:
            from .readers import wants_plan
            from . import names
            said = [c.get("name", "") for c in snap.state.get("courses", []) if isinstance(c, dict) and _mentions(c.get("name", ""), value)]
            if len(said) == 1:                               # «de redes»: corrige el curso
                args["curso"] = said[0]
                self.last_course = said[0]
            if sp.tool == "simular_notas":
                from .readers import _NUM_RE
                old = args.get("_heard") or ""
                if re.search(r"\b" + _NUM_RE + r"\b", strip_accents(value).lower()):
                    args["_heard"] = value                   # dijo de nuevo nota y evaluación: manda lo nuevo
                else:                                        # «la eva 3» / «de redes»: la nota anterior + lo nuevo
                    m = re.search(r"\b" + _NUM_RE + r"\b", strip_accents(old).lower())
                    args["_heard"] = f"{m.group(1)} en {value}" if m and eval_spec(value) != (None, None, False) else f"{old} {value}".strip()
            else:
                keep = " (plan)" if wants_plan(args.get("_heard")) else ""   # «dame un plan...» se conserva tras «¿de cuál?»
                args["_heard"] = value + keep                # la respuesta («el tercero») es lo que manda ahora
        step = self._read_step if is_read else self._write_step
        out = step(sp.tool, args, sp.heard or text, snap, self._today(), lambda: int((self._clock() - t0) * 1000), attempts=again)
        if out.kind == "ask" and again > 3 and self.gate.slot is not None:   # no se entiende: mejor parar
            self.gate.clear()
            return Outcome("No logro entenderte; lo dejo ahí. Pídemelo de nuevo." if is_read else
                           "No logro entenderte. Cancelé la tarea; pídemela de nuevo.", "cancelled", sp.tool)
        return out

    def _correct_date(self, prop: Proposal, pend, phrase: str, heard: str) -> Outcome:
        t0 = self._clock()
        self.gate.clear()
        self._log(prop.tool, "corrected", level=WRITE, heard=heard, result=f"fecha corregida a «{phrase}»")
        try:
            snap = self.repo.read()
        except UniGymDataError as e:
            return Outcome(f"No pude leer tus datos de UniGym. {e}", "error", prop.tool, ok=False)
        args = dict(pend.args, fecha=phrase)
        return self._write_step(prop.tool, args, pend.heard or heard, snap, self._today(),
                                lambda: int((self._clock() - t0) * 1000))

    def _apply(self, prop: Proposal, pend, heard: str) -> Outcome:
        t0 = self._clock()
        try:
            snap = self.repo.write(lambda s: prop.apply(s), by="jarvis", reason=f"{prop.tool}: {prop.question}")
        except (UniGymDataError, RevisionConflict) as e:
            self._log(prop.tool, "error", level=WRITE, heard=heard, result=str(e), revision_before=pend.revision)
            return Outcome(f"No pude guardarlo: {e}", "error", prop.tool, ok=False)
        self._log(prop.tool, "confirmed", level=WRITE, heard=heard, result=prop.done, revision_before=pend.revision,
                  revision_after=snap.revision, duration_ms=int((self._clock() - t0) * 1000))
        return Outcome(prop.done, "done", prop.tool, data={"revision": snap.revision})

    # ------------------------------ utilidades ------------------------------
    def _log(self, tool, status, **kw):
        if self.audit:
            return self.audit.record(tool, status, **kw)
        return None
