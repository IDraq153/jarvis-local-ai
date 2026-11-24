"""Sincronización web <-> JARVIS de punta a punta: store.js REAL (Node) contra el servidor REAL (Python).

Se omite si no hay Node o no se encuentra la web (../uni-gym-tracker). Datos inventados.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.test_unigym_repo import backup_envelope
from tools.unigym.repo import JsonFileRepository
from tools.unigym.server import make_server, serve_in_thread

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(os.environ.get("UNIGYM_WEB", ROOT.parent / "uni-gym-tracker"))
SCRIPT = ROOT / "tests" / "js" / "sync_e2e.mjs"


@unittest.skipUnless(shutil.which("node") and (WEB / "js" / "core" / "store.js").exists(), "falta Node o la web")
class WebSync(unittest.TestCase):
    def run_node(self, mode, with_file=True):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            repo = JsonFileRepository(d / "data" / "unigym.json")
            if with_file:
                (d / "r.json").write_text(json.dumps(backup_envelope()), encoding="utf-8")
                repo = JsonFileRepository.from_backup(d / "r.json", d / "data" / "unigym.json")
            srv = make_server(repo, WEB, port=0)
            serve_in_thread(srv)
            try:
                env = dict(os.environ, UNIGYM_WEB=str(WEB), PORT=str(srv.server_address[1]), MODE=mode)
                out = subprocess.run(["node", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
            finally:
                srv.shutdown()
                srv.server_close()
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(f"OK {mode}", out.stdout)

    def test_shared_file_mode(self):
        self.run_node("normal")

    def test_falls_back_to_localstorage_without_data_file(self):
        self.run_node("nofile", with_file=False)


if __name__ == "__main__":
    unittest.main()
