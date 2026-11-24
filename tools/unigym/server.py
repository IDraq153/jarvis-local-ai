"""Servidor local de UniGym (Fase 6.3): sirve la web y expone el archivo de datos compartido.

    GET  /api/revision  -> {"revision": N}                     (barato: la web lo consulta cada ~2 s)
    GET  /api/state     -> {"revision", "updatedAt", "by", "data"}
    PUT  /api/state     <- {"baseRevision": N, "data": {...}}  -> 200 {"revision"} | 409 {"revision", "error"}

Solo escucha en 127.0.0.1. Defensas contra otras páginas abiertas en tu navegador (CSRF / DNS rebinding):
el PUT exige la cabecera X-UniGym (los navegadores no la permiten entre sitios sin un permiso CORS que
este servidor nunca concede), Content-Type application/json y un Host de localhost. Los estáticos salen
solo de `web_dir`: `data/` (tus datos y copias) nunca se sirve.
"""
import json
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .repo import RevisionConflict, UniGymDataError, UniGymRepository, extract_state, validate_state

MAX_BODY = 8 * 1024 * 1024
_HOST_OK = re.compile(r"^(localhost|127\.0\.0\.1)(:\d+)?$", re.I)


class Handler(SimpleHTTPRequestHandler):
    repo: UniGymRepository = None

    def __init__(self, *a, repo=None, **kw):
        self.repo = repo
        super().__init__(*a, **kw)

    # ---- utilidades ----
    def log_message(self, fmt, *args):   # silencio: la consola de JARVIS no necesita cada GET
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        return bool(_HOST_OK.match(self.headers.get("Host", "")))

    # ---- rutas ----
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/api/revision", "/api/state"):
            if not self._host_ok():
                return self._json(403, {"error": "host no permitido"})
            try:
                snap = self.repo.read()
            except UniGymDataError as e:
                return self._json(404, {"error": str(e)})
            if path == "/api/revision":
                return self._json(200, {"revision": snap.revision})
            return self._json(200, {"revision": snap.revision, "updatedAt": snap.updated_at, "by": snap.by,
                                    "data": snap.state})
        if path.startswith("/api/"):
            return self._json(404, {"error": "no existe"})
        return super().do_GET()

    def do_PUT(self):
        if self.path.split("?", 1)[0] != "/api/state":
            return self._json(404, {"error": "no existe"})
        if not self._host_ok() or self.headers.get("X-UniGym") != "1":
            return self._json(403, {"error": "petición no permitida"})
        if "application/json" not in self.headers.get("Content-Type", ""):
            return self._json(415, {"error": "se espera application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            return self._json(413, {"error": "cuerpo vacío o demasiado grande"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            base = payload["baseRevision"]
            if not isinstance(base, int) or isinstance(base, bool):
                raise ValueError("baseRevision debe ser entero")
            state = extract_state({"data": payload["data"]}, strict=True)
            validate_state(state)
        except (ValueError, KeyError, TypeError, UniGymDataError) as e:
            return self._json(400, {"error": f"datos inválidos: {e}"})
        try:
            snap = self.repo.write(lambda _old: state, expected_revision=base, by="web", reason="guardado desde la web")
        except RevisionConflict as e:
            return self._json(409, {"error": str(e), "revision": e.current})
        except UniGymDataError as e:
            return self._json(500, {"error": str(e)})
        return self._json(200, {"revision": snap.revision})

    def do_POST(self):
        self._json(405, {"error": "método no permitido"})

    do_DELETE = do_PATCH = do_POST


def make_server(repo: UniGymRepository, web_dir, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("El servidor de UniGym solo puede escuchar en localhost.")
    handler = partial(Handler, repo=repo, directory=str(Path(web_dir).resolve()))
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def serve_in_thread(srv: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return t
