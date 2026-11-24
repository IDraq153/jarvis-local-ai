"""Rutinas por frase (Fase 6, herramienta 2): "Jarvis, estoy en casa".

No pasa por el LLM: la frase se reconoce en Python (igual que los comandos de memoria), así es instantánea y no se
dispara por error con frases parecidas ("estoy en casa de mi tío" NO la activa). Hace tres cosas:
  1. dice «Bienvenido, señor.» + lo más urgente de UniGym (si hay) + una frase para motivar el día;
  2. cuando termina de hablar, abre el video configurado en Firefox (la URL sale de config.yaml, nunca del LLM ni de lo que
     se oyó, y solo se aceptan hosts de la lista `allowed_hosts`);
  3. deja registro en `tool_calls`.
"""
import datetime as dt
import os
import random
import shutil
import time
import subprocess
import webbrowser
from dataclasses import dataclass
from urllib.parse import urlparse

from memory.text import strip_accents

PHRASES = (
    "Hoy no necesitas ganas, necesitas el primer paso.",
    "Lo que haces hoy es lo que tu yo de mañana va a agradecer.",
    "La constancia le gana al talento cuando el talento no es constante.",
    "Un avance pequeño hoy vale más que un plan perfecto mañana.",
    "No tienes que hacerlo todo hoy, solo lo siguiente que importa.",
    "Cada línea de código, cada página, cada serie del gym te acerca a donde quieres estar.",
    "La disciplina es hacer lo que toca incluso cuando no apetece.",
    "Empieza ahora, ajusta en el camino.",
    "Estás más cerca de lo que ayer parecía lejos.",
    "Hazlo difícil hoy para que mañana sea fácil.",
    "Tu futuro profesional se construye con las horas que nadie ve.",
    "Respira, ordena tus prioridades y ataca una a la vez.",
    "Un buen día no se espera, se arma tarea por tarea.",
    "Lo importante no es ir rápido, es no detenerse.",
    "Hoy es un buen día para mejorar un uno por ciento.",
    "El cansancio pasa, lo que dejas sin hacer se queda.",
    "Si fue fácil, aprendiste poco; si costó, estás creciendo.",
    "Confía en el proceso: estás entrenando mente y cuerpo a la vez.",
    "Haz que hoy cuente.",
    "Paso a paso también se llega lejos.",
)

# Frases completas (sin tildes, sin "jarvis"): solo estas activan la rutina.
_CASA = {
    "estoy en casa", "ya estoy en casa", "ya llegue", "ya llegue a casa", "llegue a casa", "estoy en casa ya",
    "por fin estoy en casa", "ya estoy aqui en casa", "llegue", "estoy de vuelta en casa", "ya estoy de vuelta",
}
_FILLER = {"jarvis", "oye", "hola", "senor"}


def _key(text: str) -> str:
    s = strip_accents(text or "").lower()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    return " ".join(t for t in s.split() if t not in _FILLER)


def is_home_phrase(text: str) -> bool:
    if "?" in (text or "") or "¿" in (text or ""):      # "¿estoy en casa?" es una pregunta, no el aviso
        return False
    return _key(text) in _CASA


def firefox_path() -> str | None:
    """Ruta de Firefox: PATH, o las instalaciones habituales de Windows."""
    found = shutil.which("firefox")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), r"C:\Program Files"):
        if base:
            cand = os.path.join(base, "Mozilla Firefox", "firefox.exe")
            if os.path.exists(cand):
                return cand
    return None


def open_in_firefox(url: str) -> str:
    """Abre `url` en Firefox (ventana nueva). Si no hay Firefox usa el navegador predeterminado. Devuelve cuál usó."""
    exe = firefox_path()
    if exe:
        subprocess.Popen([exe, "-new-window", url])      # lista de argumentos: nada pasa por un intérprete de comandos
        return "firefox"
    webbrowser.open(url, new=1)
    return "predeterminado"


def valid_url(url: str, allowed_hosts) -> bool:
    try:
        u = urlparse(url)
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    ok = [h.lower() for h in (allowed_hosts or [])]
    return u.scheme == "https" and any(host == h or host.endswith("." + h) for h in ok)


@dataclass
class RoutineResult:
    speech: str                 # lo que JARVIS dice
    after: object               # callable: abre el video (se llama cuando JARVIS ya terminó de hablar)
    tool: str = "estoy_en_casa"
    with_speech: bool = True    # True: el video se abre a la vez que JARVIS habla; False: al terminar de hablar


class RoutineRunner:
    def __init__(self, cfg: dict | None, repo=None, audit=None, today=dt.date.today, opener=open_in_firefox,
                 rng: random.Random | None = None, sleep=time.sleep):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.url = cfg.get("url") or ""
        self.allowed = cfg.get("allowed_hosts") or ["youtube.com", "youtu.be"]
        self.repo, self.audit, self._today, self.opener = repo, audit, today, opener
        self.rng = rng or random.Random()
        self.with_speech = bool(cfg.get("open_with_speech", True))
        self.delay_s = float(cfg.get("open_delay_s", 1.0))   # margen extra tras la última palabra antes de abrir el video
        self._sleep = sleep
        self.last_opened = None

    def match(self, text: str) -> RoutineResult | None:
        if not self.enabled or not is_home_phrase(text):
            return None
        parts = ["Bienvenido, señor."]
        urgent = self._urgent()
        if urgent:
            parts.append(f"Lo más urgente: {urgent}.")
        parts.append(self.rng.choice(PHRASES))
        speech = " ".join(parts)
        self._log("said", text, speech)
        return RoutineResult(speech, lambda: self._open(text), with_speech=self.with_speech)

    # ------------------------------------------------------------------
    def _urgent(self) -> str:
        """La tarea pendiente más urgente (misma ordenación que tareas_pendientes), o ''."""
        if self.repo is None:
            return ""
        try:
            from .unigym.readers import tareas_pendientes
            r = tareas_pendientes(self.repo.read().state, {}, self._today())
        except Exception:
            return ""
        titles = (r.data or {}).get("titulos") or []
        return str(titles[0]) if titles else ""

    def _open(self, heard: str) -> str:
        if not self.url or not valid_url(self.url, self.allowed):
            self._log("rejected", heard, "URL no válida o no permitida en config.yaml (routines.casa.url)")
            return "sin video"
        if self.delay_s > 0 and not self.with_speech:
            self._sleep(self.delay_s)
        try:
            used = self.opener(self.url)
        except Exception as e:  # abrir el navegador nunca debe tumbar a JARVIS
            self._log("error", heard, f"no pude abrir el navegador: {e}")
            return "error"
        self.last_opened = used
        self._log("opened", heard, f"video abierto en {used}")
        return used

    def _log(self, status, heard, result):
        if self.audit:
            try:
                self.audit.record("estoy_en_casa", status, level="write", heard=heard, result=result)
            except Exception:
                pass
