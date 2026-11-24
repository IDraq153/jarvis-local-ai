"""Capa de datos de UniGym (Fase 6, subfase 6.1): un archivo JSON compartido con la web.

Fuente de verdad: `data/unigym.json` (ignorado por git). Lo leen y escriben dos clientes:
las herramientas de JARVIS (directo, sin depender de ningún servidor) y el servidor que
sirve la web (`GET/PUT /api/state`, subfase 6.3). Por eso cada escritura es:

    bloqueo -> releer -> comprobar revisión -> mutar -> validar -> copia previa -> escritura atómica

- Revisión: entero que sube en cada escritura. Quien escribe puede exigir `expected_revision`
  (la web lo hace: si JARVIS escribió mientras tanto, recibe RevisionConflict y se refresca).
- Copia previa: `backups/unigym-AAAAMMDD-HHMMSS-rN.json` antes de cada escritura (últimas `keep`).
- Atómica: se escribe un temporal y se reemplaza con os.replace; un corte de luz no deja el
  archivo a medias.
- La interfaz `UniGymRepository` es lo único que conocen las herramientas, así la Fase 7
  puede pasar a SQLite sin reescribirlas.
"""
import copy
import datetime as dt
import json
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

APP_ID = "uni-gym-tracker"   # igual que APP_ID de js/core/backup.js
FORMAT = 1


class UniGymDataError(RuntimeError):
    """Datos ausentes, dañados o con forma inválida."""


class RevisionConflict(UniGymDataError):
    """Alguien escribió después de que leíste: vuelve a leer y reintenta."""

    def __init__(self, expected: int, current: int):
        super().__init__(f"Conflicto de revisión: esperaba {expected}, el archivo está en {current}.")
        self.expected, self.current = expected, current


class LockTimeout(UniGymDataError):
    """No se pudo tomar el bloqueo de escritura a tiempo."""


@dataclass
class Snapshot:
    state: dict
    revision: int
    updated_at: str = ""
    by: str = ""


# ------------------------------- Forma del estado -------------------------------

_LISTS = ("courses", "tasks", "habits")
_DICTS = ("settings", "dailyLogs", "schedule", "gym", "record")


def _as_dict(v):
    return v if isinstance(v, dict) else {}


def _as_list(v):
    return v if isinstance(v, list) else []


def normalize_state(raw, strict: bool = False) -> dict:
    """Garantiza que existan los contenedores que esperan las herramientas.

    No cambia valores ni descarta claves desconocidas: lo que la web guardó se conserva
    tal cual (la web hace su propia normalización completa al cargar).
    Con `strict=True` un contenedor presente con el tipo equivocado (p. ej. `tasks: "x"`) lanza
    UniGymDataError en vez de convertirse en vacío: así un dato mal formado nunca borra en silencio
    lo que había. Se usa en todo lo que se va a escribir; la lectura es tolerante.
    """
    def L(v, where):
        if strict and v is not None and not isinstance(v, list):
            raise UniGymDataError(f"'{where}' debe ser una lista.")
        return _as_list(v)

    def D(v, where):
        if strict and v is not None and not isinstance(v, dict):
            raise UniGymDataError(f"'{where}' debe ser un objeto.")
        return _as_dict(v)

    if strict and not isinstance(raw, dict):
        raise UniGymDataError("El estado debe ser un objeto.")
    s = dict(raw) if isinstance(raw, dict) else {}
    for k in _LISTS:
        s[k] = L(s.get(k), k)
    for k in _DICTS:
        s[k] = D(s.get(k), k)
    s["settings"]["goals"] = D(s["settings"].get("goals"), "settings.goals")
    s["settings"]["pomodoro"] = D(s["settings"].get("pomodoro"), "settings.pomodoro")
    s["schedule"]["blocks"] = L(s["schedule"].get("blocks"), "schedule.blocks")
    g = s["gym"]
    for k in ("routines", "sessions", "bodyWeight"):
        g[k] = L(g.get(k), f"gym.{k}")
    g.setdefault("activeSession", None)
    s["record"]["cycles"] = L(s["record"].get("cycles"), "record.cycles")
    for c in s["courses"]:
        if isinstance(c, dict):
            c["evals"] = L(c.get("evals"), "courses[].evals")
    for r in g["routines"]:
        if isinstance(r, dict):
            r["days"] = L(r.get("days"), "gym.routines[].days")
            r["exercises"] = L(r.get("exercises"), "gym.routines[].exercises")
    for h in s["habits"]:
        if isinstance(h, dict):
            h["log"] = D(h.get("log"), "habits[].log")
    for x in g["sessions"]:
        if isinstance(x, dict):
            x["entries"] = L(x.get("entries"), "gym.sessions[].entries")
    return s


def validate_state(state) -> None:
    """Rechaza un estado que rompería la web (tipos equivocados). Lanza UniGymDataError."""
    if not isinstance(state, dict):
        raise UniGymDataError("El estado debe ser un objeto.")
    for k in _LISTS:
        if not isinstance(state.get(k), list):
            raise UniGymDataError(f"'{k}' debe ser una lista.")
    for k in _DICTS:
        if not isinstance(state.get(k), dict):
            raise UniGymDataError(f"'{k}' debe ser un objeto.")
    for t in state["tasks"]:
        if not isinstance(t, dict) or not isinstance(t.get("title"), str) or not t["title"].strip():
            raise UniGymDataError("Cada tarea necesita un título de texto.")
        if t.get("due") not in (None, "") and not _is_iso_date(t["due"]):
            raise UniGymDataError(f"Fecha límite inválida en una tarea: {t.get('due')!r}.")
    for w in state["gym"].get("bodyWeight", []):
        if not isinstance(w, dict) or not isinstance(w.get("kg"), (int, float)) or not _is_iso_date(w.get("date")):
            raise UniGymDataError("Cada peso corporal necesita fecha y kg numéricos.")
    for d in state["dailyLogs"]:
        if not _is_iso_date(d):
            raise UniGymDataError(f"Clave de registro diario inválida: {d!r}.")
    try:
        json.dumps(state, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise UniGymDataError(f"El estado no se puede guardar como JSON: {e}") from e


def _is_iso_date(v) -> bool:
    try:
        return isinstance(v, str) and len(v) == 10 and dt.date.fromisoformat(v).isoformat() == v
    except ValueError:
        return False


def extract_state(payload, strict: bool = False) -> dict:
    """Acepta un respaldo de la web (`{app, version, exportedAt, data}`), un archivo del
    repositorio (`{..., revision, data}`) o el estado a secas; devuelve el estado normalizado."""
    if not isinstance(payload, dict):
        raise UniGymDataError("El archivo no tiene la forma de un respaldo de UniGym Tracker.")
    if isinstance(payload.get("data"), dict) and payload.get("app") in (None, APP_ID):
        payload = payload["data"]
    if not any(k in payload for k in ("courses", "tasks", "gym", "schedule", "settings")):
        raise UniGymDataError("El archivo no tiene la forma de un respaldo de UniGym Tracker.")
    return normalize_state(payload, strict=strict)


# ---------------------------------- Interfaz ----------------------------------

class UniGymRepository(ABC):
    @abstractmethod
    def read(self) -> Snapshot:
        """Estado actual (copia que puedes modificar sin efecto) y su revisión."""

    @abstractmethod
    def write(self, mutator, *, expected_revision: int | None = None, by: str = "jarvis",
              reason: str = "") -> Snapshot:
        """Aplica `mutator(state)` y guarda. Si devuelve un dict, ese dict reemplaza al estado.
        Si lanza, no se guarda nada. Con `expected_revision` distinto de la actual: RevisionConflict."""


# ----------------------------- Implementación en archivo -----------------------------

class JsonFileRepository(UniGymRepository):
    def __init__(self, path, backups_dir=None, keep: int = 30, lock_timeout: float = 5.0,
                 stale_lock_s: float = 30.0, now=None):
        self.path = Path(path)
        self.backups_dir = Path(backups_dir) if backups_dir else self.path.parent / "backups"
        self.keep = keep
        self.lock_timeout = lock_timeout
        self.stale_lock_s = stale_lock_s
        self._now = now or dt.datetime.now
        self._lock_path = self.path.with_name(self.path.name + ".lock")

    # ---- creación / migración ----
    @classmethod
    def from_backup(cls, source, dest, overwrite: bool = False, **kw) -> "JsonFileRepository":
        """Crea `dest` desde un respaldo exportado por la web (migración única)."""
        repo = cls(dest, **kw)
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise UniGymDataError(f"No pude leer el respaldo: {e}") from e
        state = extract_state(payload, strict=True)
        validate_state(state)
        with repo._lock():
            if repo.path.exists():
                if not overwrite:
                    raise UniGymDataError(f"{repo.path.name} ya existe; usa overwrite para reemplazarlo (se hace copia).")
                repo._backup(repo._load_envelope())
            repo._save(state, 1, by="import", reason="importado desde respaldo")
        return repo

    # ---- lectura ----
    def read(self) -> Snapshot:
        env = self._load_envelope()
        return Snapshot(copy.deepcopy(env["data"]), env["revision"], env.get("updatedAt", ""), env.get("by", ""))

    def exists(self) -> bool:
        return self.path.exists()

    def _load_envelope(self) -> dict:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise UniGymDataError(
                f"No hay datos de UniGym en {self.path}. Exporta un respaldo desde la web y créalo con "
                "scripts/unigym_import.py.") from None
        except (OSError, ValueError) as e:
            raise UniGymDataError(f"{self.path.name} está dañado ({e}). Hay copias en {self.backups_dir}.") from e
        if not isinstance(raw, dict) or not isinstance(raw.get("data"), dict):
            raise UniGymDataError(f"{self.path.name} no tiene el formato esperado.")
        rev = raw.get("revision")
        raw["revision"] = rev if isinstance(rev, int) and rev >= 1 else 1
        raw["data"] = normalize_state(raw["data"])
        return raw

    # ---- escritura ----
    def write(self, mutator, *, expected_revision=None, by="jarvis", reason="") -> Snapshot:
        with self._lock():
            env = self._load_envelope()
            current = env["revision"]
            if expected_revision is not None and expected_revision != current:
                raise RevisionConflict(expected_revision, current)
            draft = copy.deepcopy(env["data"])
            result = mutator(draft)
            if result is not None and not isinstance(result, dict):
                raise UniGymDataError("El mutador debe devolver None o un estado (objeto).")
            new_state = normalize_state(result if isinstance(result, dict) else draft, strict=True)
            validate_state(new_state)
            self._backup(env)
            return self._save(new_state, current + 1, by=by, reason=reason)

    def _save(self, state: dict, revision: int, by: str, reason: str) -> Snapshot:
        updated = self._now().isoformat(timespec="seconds")
        env = {"app": APP_ID, "format": FORMAT, "revision": revision, "updatedAt": updated,
               "by": by, "reason": reason, "data": state}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(env, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(6):   # en Windows el reemplazo falla un instante si el servidor lee el archivo
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05)
        return Snapshot(copy.deepcopy(state), revision, updated, by)

    def _backup(self, env: dict) -> Path:
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._now().strftime("%Y%m%d-%H%M%S")
        dest = self.backups_dir / f"unigym-{stamp}-r{env['revision']}.json"
        dest.write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
        files = sorted(self.backups_dir.glob("unigym-*.json"), key=lambda p: (p.stat().st_mtime, p.name))
        for old in files[: max(0, len(files) - self.keep)]:
            try:
                old.unlink()
            except OSError:
                pass
        return dest

    # ---- bloqueo entre procesos ----
    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self._lock_path) > self.stale_lock_s:
                        os.remove(self._lock_path)   # dueño muerto: bloqueo huérfano
                        continue
                except FileNotFoundError:
                    continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise LockTimeout("Otro proceso está escribiendo los datos de UniGym; reintenta en un momento.")
                time.sleep(0.02)
        try:
            yield
        finally:
            try:
                os.remove(self._lock_path)
            except FileNotFoundError:
                pass
