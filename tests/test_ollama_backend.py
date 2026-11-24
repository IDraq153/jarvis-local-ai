"""Prueba el adaptador contra un servidor Ollama simulado (sin GPU ni modelos).
Ejecutar: python -m unittest discover -s tests -v
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from llm import LLMError, load_backend


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] == "missing":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        for w in ["Hola", ", ", "Dario"]:
            self.wfile.write((json.dumps({"message": {"content": w}, "done": False}) + "\n").encode())
        self.wfile.write((json.dumps({"message": {"content": ""}, "done": True,
                                      "eval_count": 30, "eval_duration": 1_000_000_000}) + "\n").encode())


class TestOllama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.host = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_stream_y_tps(self):
        llm = load_backend({"model": "x", "host": self.host, "candidates": ["a"],
                            "options": {"temperature": 0.1}})
        self.assertEqual(list(llm.stream([{"role": "user", "content": "hi"}])), ["Hola", ", ", "Dario"])
        self.assertEqual(llm.tokens_per_second(), 30.0)

    def test_modelo_ausente(self):
        with self.assertRaisesRegex(LLMError, "ollama pull missing"):
            list(load_backend({"model": "missing", "host": self.host}).stream([]))

    def test_servidor_caido(self):
        with self.assertRaisesRegex(LLMError, "conectar"):
            list(load_backend({"model": "x", "host": "http://127.0.0.1:9"}).stream([]))

    def test_sin_modelo(self):
        with self.assertRaises(LLMError):
            load_backend({"model": None})


if __name__ == "__main__":
    unittest.main()
