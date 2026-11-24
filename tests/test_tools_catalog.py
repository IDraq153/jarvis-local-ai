"""Pruebas del catálogo de herramientas y del selector (Fase 6.0), sin Ollama."""
import unittest

import numpy as np

from tools import READ, WRITE, ToolSelector, approx_tokens
from tools.prompt import tool_guide
from tools.unigym import SPECS, spec_by_name


class FakeEmbedder:
    """Embedding por bolsa de palabras con hash: textos con palabras comunes se parecen."""
    model = "fake"

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace("¿", "").replace("?", "").split():
                out[i, hash(w) % 64] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-12)


class CatalogTests(unittest.TestCase):
    def test_names_unique_and_snake_case(self):
        names = [s.name for s in SPECS]
        self.assertEqual(len(names), len(set(names)))
        for n in names:
            self.assertRegex(n, r"^[a-z]+(_[a-z]+)*$")

    def test_schema_shape(self):
        for s in SPECS:
            f = s.ollama_schema()["function"]
            self.assertEqual(f["name"], s.name)
            self.assertEqual(f["parameters"]["type"], "object")
            self.assertTrue(f["description"])
            for req in f["parameters"].get("required", []):
                self.assertIn(req, f["parameters"]["properties"])
            self.assertGreaterEqual(len(s.examples), 3, s.name)

    def test_levels(self):
        reads = {s.name for s in SPECS if s.level == READ}
        writes = {s.name for s in SPECS if s.level == WRITE}
        self.assertEqual(reads | writes, {s.name for s in SPECS})
        self.assertIn("crear_tarea", writes)
        self.assertIn("consultar_agenda", reads)
        for s in SPECS:
            if s.level == WRITE:
                self.assertIn("confirm", s.description.lower().replace("ó", "o"), s.name)

    def test_token_budget_of_top_k(self):
        """Cualquier 4 herramientas deben caber en ~650 tokens del contexto de 4096."""
        costs = sorted((approx_tokens(s.ollama_schema()) for s in SPECS), reverse=True)
        self.assertLess(sum(costs[:4]), 700)
        self.assertGreater(sum(costs), 1000)  # con todas el contexto se resentiría: por eso se selecciona

    def test_spec_by_name(self):
        self.assertIs(spec_by_name("crear_tarea"), next(s for s in SPECS if s.name == "crear_tarea"))
        self.assertIsNone(spec_by_name("no_existe"))

    def test_tool_guide_has_today(self):
        import datetime as dt
        self.assertIn("viernes 2026-11-27", tool_guide(dt.date(2026, 11, 27)))


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.sel = ToolSelector(SPECS, FakeEmbedder(), top_k=3, min_score=0.2)

    def test_exact_example_ranks_first(self):
        for spec in SPECS:
            top = self.sel.rank(spec.examples[0])[0]
            self.assertEqual(top.spec.name, spec.name)
            self.assertGreater(top.score, 0.99)

    def test_select_respects_top_k_and_threshold(self):
        picked = self.sel.select("¿qué tengo hoy?")
        self.assertLessEqual(len(picked), 3)
        self.assertEqual(picked[0].name, "consultar_agenda")

    def test_nothing_selected_for_unrelated_text(self):
        sel = ToolSelector(SPECS, FakeEmbedder(), top_k=3, min_score=0.9)
        self.assertEqual(sel.select("xyzzy plugh"), [])


if __name__ == "__main__":
    unittest.main()
