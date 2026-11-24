"""Pruebas de la comprobación de números dichos (Fase 6.4)."""
import unittest

from tests.test_unigym_executor import FlowBase
from tools.unigym.grounding import candidates, grounding_problem


class Candidates(unittest.TestCase):
    def test_formas_de_decir_un_numero(self):
        for heard, want in (("peso 70.5 kilos", 70.5), ("peso 70,5", 70.5), ("peso 70 con 8", 70.8), ("peso 70 punto 5", 70.5),
                            ("dormí 7 y media", 7.5), ("saqué 15", 15), ("9000 pasos", 9000)):
            vals, _ = candidates(heard)
            self.assertTrue(any(abs(v - want) < 1e-9 for v in vals), (heard, vals))

    def test_numeros_en_palabras(self):
        for heard in ("tomé un vaso de agua", "dormí siete horas y media", "saqué quince", "pesé setenta kilos", "tomé veintiún vasos"):
            self.assertTrue(candidates(heard)[1], heard)
        self.assertFalse(candidates("anota que tengo que entregar el informe de redes el lunes")[1])


class Problem(unittest.TestCase):
    def test_numero_inventado_se_rechaza(self):
        # el caso real del bench: nadie dijo 18
        self.assertIn("ningún número", grounding_problem({"curso": "redes", "nota": 18}, "Anota que tengo que entregar el informe de redes el lunes"))
        self.assertIn("seguro", grounding_problem({"kg": 75}, "registra que peso 70.5 kilos"))

    def test_numero_dicho_pasa(self):
        for args, heard in (({"kg": 70.5}, "registra que peso 70.5 kg"), ({"kg": 70.8}, "peso 70 con 8"), ({"valor": 1}, "tomé un vaso de agua"),
                            ({"valor": 7.5}, "dormí siete horas y media"), ({"nota": 15, "curso": "redes"}, "saqué 15 en el parcial de redes"),
                            ({"meta": 15}, "¿cuánto necesito en redes para llegar a 15?"), ({"kg": "70,5"}, "peso 70,5")):
            self.assertIsNone(grounding_problem(args, heard), (args, heard))

    def test_sin_numeros_en_args_o_sin_frase_no_hay_problema(self):
        self.assertIsNone(grounding_problem({"curso": "redes"}, "¿cómo voy en redes?"))
        self.assertIsNone(grounding_problem({"nota": 18}, ""))             # llamada programática
        self.assertIsNone(grounding_problem({"nota": None}, "hola"))
        self.assertIsNone(grounding_problem("no es dict", "hola 3"))


class ConElEjecutor(FlowBase):
    def test_nota_inventada_no_llega_a_proponerse(self):
        out = self.tools.run("registrar_nota", {"curso": "redes", "evaluacion": "informe", "nota": 18},
                             heard="Anota que tengo que entregar el informe de redes el lunes")
        self.assertEqual((out.kind, out.ok), ("ask", False))
        self.assertFalse(self.tools.has_pending)
        self.assertEqual(self.repo.read().revision, 1)

    def test_peso_dicho_con_con(self):
        out = self.tools.run("registrar_peso", {"kg": 70.8}, heard="Registra que peso 70 con 8")
        self.assertEqual(out.kind, "confirm")
        self.assertIn("70,8", out.text)


if __name__ == "__main__":
    unittest.main()
