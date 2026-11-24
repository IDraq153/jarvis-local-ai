"""Pruebas del servidor de UniGym (Fase 6.3) por HTTP real en localhost, con datos inventados."""
import http.client
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_unigym_repo import backup_envelope, synthetic_state
from tools.unigym.repo import JsonFileRepository
from tools.unigym.server import make_server, serve_in_thread


class ServerBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        (d / "web").mkdir()
        (d / "web" / "index.html").write_text("<h1>web</h1>", encoding="utf-8")
        (d / "secreto.txt").write_text("NO SE SIRVE", encoding="utf-8")
        src = d / "r.json"
        src.write_text(json.dumps(backup_envelope()), encoding="utf-8")
        self.path = d / "data" / "unigym.json"
        self.repo = JsonFileRepository.from_backup(src, self.path)
        self.srv = make_server(self.repo, d / "web", port=0)
        self.port = self.srv.server_address[1]
        serve_in_thread(self.srv)

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self._tmp.cleanup()

    def req(self, method, path, body=None, headers=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = dict(headers or {})
        if host:
            h["Host"] = host
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, dict(r.getheaders()), data

    def put(self, state, base, **kw):
        body = json.dumps({"baseRevision": base, "data": state}).encode()
        h = {"Content-Type": "application/json", "X-UniGym": "1"} | kw.pop("headers", {})
        return self.req("PUT", "/api/state", body, h, **kw)


class Reading(ServerBase):
    def test_serves_web_and_state(self):
        s, h, b = self.req("GET", "/")
        self.assertEqual((s, b), (200, b"<h1>web</h1>"))
        self.assertEqual(h["Cache-Control"], "no-store")
        s, _, b = self.req("GET", "/api/revision")
        self.assertEqual((s, json.loads(b)), (200, {"revision": 1}))
        s, _, b = self.req("GET", "/api/state")
        j = json.loads(b)
        self.assertEqual((s, j["revision"], j["by"]), (200, 1, "import"))
        self.assertEqual(j["data"]["tasks"][0]["title"], "Tarea uno")

    def test_does_not_serve_files_outside_web_dir(self):
        for p in ("/../secreto.txt", "/%2e%2e/secreto.txt", "/..%2fsecreto.txt", "/data/unigym.json"):
            s, _, b = self.req("GET", p)
            self.assertNotEqual(b, b"NO SE SIRVE", p)
            self.assertEqual(s, 404, p)

    def test_no_data_file_gives_404_so_web_falls_back(self):
        self.path.unlink()
        s, _, b = self.req("GET", "/api/state")
        self.assertEqual(s, 404)
        self.assertIn("error", json.loads(b))

    def test_no_cors_headers_ever(self):
        _, h, _ = self.req("GET", "/api/state")
        self.assertNotIn("Access-Control-Allow-Origin", h)
        s, _, _ = self.req("OPTIONS", "/api/state")
        self.assertEqual(s, 501)

    def test_foreign_host_header_rejected(self):
        for host in ("evil.example.com", "localhost.evil.com", "127.0.0.1.evil.com"):
            self.assertEqual(self.req("GET", "/api/state", host=host)[0], 403, host)
        self.assertEqual(self.req("GET", "/api/state", host=f"localhost:{self.port}")[0], 200)


class Writing(ServerBase):
    def test_put_saves_and_bumps_revision(self):
        st = synthetic_state()
        st["tasks"].append({"id": "t2", "title": "Desde la web", "due": "", "done": False})
        s, _, b = self.put(st, 1)
        self.assertEqual((s, json.loads(b)), (200, {"revision": 2}))
        snap = self.repo.read()
        self.assertEqual((snap.revision, snap.by), (2, "web"))
        self.assertEqual(len(snap.state["tasks"]), 2)

    def test_stale_revision_gets_409_with_current(self):
        self.repo.write(lambda s: s["settings"].update(x=1))            # JARVIS escribe: rev 2
        s, _, b = self.put(synthetic_state(), 1)
        self.assertEqual((s, json.loads(b)["revision"]), (409, 2))
        self.assertEqual(self.repo.read().state["settings"]["x"], 1)    # lo de JARVIS sigue ahí

    def test_cross_site_requests_are_rejected(self):
        body = json.dumps({"baseRevision": 1, "data": synthetic_state()}).encode()
        self.assertEqual(self.req("PUT", "/api/state", body, {"Content-Type": "application/json"})[0], 403)   # sin X-UniGym
        self.assertEqual(self.put(synthetic_state(), 1, host="evil.com")[0], 403)
        self.assertEqual(self.req("PUT", "/api/state", body, {"Content-Type": "text/plain", "X-UniGym": "1"})[0], 415)
        self.assertEqual(self.repo.read().revision, 1)

    def test_bad_payloads_rejected_and_nothing_written(self):
        h = {"Content-Type": "application/json", "X-UniGym": "1"}
        bad_state = synthetic_state()
        bad_state["tasks"] = "no es lista"
        cases = [b"no json", b"{}", json.dumps({"baseRevision": "1", "data": synthetic_state()}).encode(),
                 json.dumps({"baseRevision": True, "data": synthetic_state()}).encode(),
                 json.dumps({"baseRevision": 1, "data": [1]}).encode(),
                 json.dumps({"baseRevision": 1, "data": bad_state}).encode()]
        for body in cases:
            s, _, _ = self.req("PUT", "/api/state", body, h)
            self.assertEqual(s, 400, body[:40])
        self.assertEqual(self.req("PUT", "/api/state", b"", h)[0], 413)
        self.assertEqual(self.repo.read().revision, 1)

    def test_other_methods_and_paths(self):
        self.assertEqual(self.req("POST", "/api/state", b"{}")[0], 405)
        self.assertEqual(self.req("DELETE", "/api/state")[0], 405)
        self.assertEqual(self.put(synthetic_state(), 1)[0], 200)
        self.assertEqual(self.req("PUT", "/otra", b"{}", {"X-UniGym": "1"})[0], 404)

    def test_refuses_non_local_bind(self):
        with self.assertRaises(ValueError):
            make_server(self.repo, ".", host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
