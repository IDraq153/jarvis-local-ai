"""Pruebas de la memoria persistente (Fase 5) sin Ollama ni GPU.
Ejecutar: python -m unittest discover -s tests -v
"""
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from llm import LLMError
from memory import Memory, MemoryConfig, MemoryStore
from memory.commands import parse_memory_command
from memory.embeddings import EmbedError, Embedder, normalize_rows
from memory.extract import parse_extraction
from memory.text import looks_secret


class FakeEmbedder(Embedder):
    """Bolsa de palabras con hash -> vector de 64 dimensiones (el coseno ~ palabras en común)."""
    def __init__(self, model="fake-a", fail=False):
        self.model, self.fail, self.calls = model, fail, 0

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise EmbedError("Ollama caído")
        from memory.text import content_words
        m = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in content_words(t):
                m[i, zlib.crc32(w.encode()) % 64] += 1.0
        return normalize_rows(m)


class FakeLLM:
    def __init__(self, reply='{"resumen": "Hablaron de redes.", "hechos": ["Dario estudia redes en Lima"]}',
                 fail=False):
        self.reply, self.fail, self.seen = reply, fail, []

    def chat(self, messages, **kw):
        self.seen.append(messages)
        if self.fail:
            raise LLMError("servidor caído")
        return self.reply


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m.db"
        self.warnings = []
        self.now = 1_000_000.0
        self.mem = self.make()
        self.addCleanup(self.tmp.cleanup)

    def make(self, embedder=None, cfg=None):
        m = Memory(MemoryStore(self.path), embedder or FakeEmbedder(), cfg or MemoryConfig(min_fact_score=0.3,
                   min_episode_score=0.3), clock=lambda: self.now, warn=self.warnings.append)
        m.start_session()
        return m


class TestFacts(Base):
    def test_remember_and_retrieve(self):
        self.mem.remember("Dario estudia Ingeniería de Sistemas en la Universidad de Lima")
        self.mem.remember("Dario tiene examen de redes el viernes")
        msgs = self.mem.augment([{"role": "system", "content": "SYS"}, {"role": "user", "content": "x"}],
                                "¿cuándo es mi examen de redes?")
        self.assertIn("examen de redes", msgs[0]["content"])
        self.assertTrue(msgs[0]["content"].startswith("SYS"))

    def test_augment_does_not_mutate_input(self):
        self.mem.remember("Dario tiene examen de redes el viernes")
        original = [{"role": "system", "content": "SYS"}]
        out = self.mem.augment(original, "examen de redes cuando")
        self.assertEqual(original[0]["content"], "SYS")
        self.assertNotEqual(out[0]["content"], "SYS")

    def test_irrelevant_query_adds_nothing(self):
        self.mem.remember("Dario tiene examen de redes el viernes")
        msgs = [{"role": "system", "content": "SYS"}]
        self.assertEqual(self.mem.augment(msgs, "cuéntame sobre dinosaurios gigantes"), msgs)

    def test_near_duplicate_updates_instead_of_duplicating(self):
        self.mem.remember("Dario vive en Lima Peru", source="auto")
        status, f = self.mem.remember("Dario vive en Lima Peru", source="auto")
        self.assertEqual(status, "updated")
        self.assertEqual(len(self.mem.facts()), 1)

    def test_secrets_are_refused(self):
        for t in ("mi contraseña es hunter2", "mi dni es 12345678", "tarjeta 4111 1111 1111 1111"):
            self.assertEqual(self.mem.remember(t)[0], "secret", t)
        self.assertEqual(self.mem.facts(), [])
        self.assertFalse(looks_secret("examen el 2026-11-24 a las 8"))

    def test_forget_matching_and_by_id(self):
        self.mem.remember("Dario tiene examen de redes el viernes")
        self.mem.remember("Dario prefiere el modelo qwen3 en Ollama")
        gone = self.mem.forget_matching("el examen de redes")
        self.assertIn("examen", gone.text)
        self.assertEqual(len(self.mem.facts()), 1)
        self.assertTrue(self.mem.forget_id(self.mem.facts()[0].id))
        self.assertIsNone(self.mem.forget_matching("algo inexistente raro"))

    def test_persists_across_sessions(self):
        self.mem.remember("Dario tiene examen de redes el viernes")
        self.mem.store.close()
        mem2 = self.make()
        self.assertEqual(len(mem2.facts()), 1)
        out = mem2.augment([{"role": "system", "content": "S"}], "examen de redes cuando")
        self.assertIn("examen de redes", out[0]["content"])


class TestEpisodes(Base):
    def test_past_session_recalled_but_current_excluded(self):
        self.mem.record_turn("hablemos del proyecto cybershield de estructuras", "Claro, cuéntame del proyecto.")
        self.mem.close_session(FakeLLM('{"resumen": "", "hechos": []}'))
        self.now += 3 * 86400
        mem2 = self.make()
        mem2.record_turn("sigamos con la tarea de contabilidad financiera", "Vale.")
        out = mem2.augment([{"role": "system", "content": "S"}], "el proyecto cybershield de estructuras")
        text = out[0]["content"]
        self.assertIn("cybershield", text)
        self.assertIn("hace 3 días", text)
        self.assertNotIn("contabilidad", text)  # turno de la sesión en curso: ya está en el historial

    def test_summary_of_last_session_is_injected(self):
        self.mem.record_turn("hablemos de redes ccna", "ok")
        self.mem.record_turn("repasemos subnetting ccna", "ok")
        res = self.mem.close_session(FakeLLM())
        self.assertEqual(res["nuevos"], 1)
        self.now += 86400
        mem2 = self.make()
        out = mem2.augment([{"role": "system", "content": "S"}], "hola")
        self.assertIn("Hablaron de redes", out[0]["content"])
        self.assertIn("ayer", out[0]["content"])

    def test_close_session_skips_short_and_survives_llm_errors(self):
        self.mem.record_turn("un solo turno aquí", "ok")
        llm = FakeLLM()
        self.assertIsNone(self.mem.close_session(llm))
        self.assertEqual(llm.seen, [])
        mem2 = self.make()
        mem2.record_turn("primer turno de prueba", "ok")
        mem2.record_turn("segundo turno de prueba", "ok")
        self.assertIsNone(mem2.close_session(FakeLLM(fail=True)))
        self.assertTrue(self.warnings)


class TestDegradation(Base):
    def test_embedder_down_falls_back_to_words_and_reindexes_later(self):
        down = self.make(FakeEmbedder(fail=True))
        down.remember("Dario tiene examen de redes el viernes")
        out = down.augment([{"role": "system", "content": "S"}], "examen de redes")
        self.assertIn("examen de redes", out[0]["content"])
        self.assertEqual(len(self.warnings), 1)       # avisa una sola vez
        up = self.make(FakeEmbedder())                # al volver Ollama, se reindexa solo
        self.assertEqual(up.store.pending_embeddings("facts", "fake-a"), [])
        self.assertIn("examen", up.augment([{"role": "system", "content": "S"}], "examen de redes")[0]["content"])

    def test_model_change_reindexes(self):
        self.mem.remember("Dario tiene examen de redes el viernes")
        self.mem.record_turn("hablemos de redes", "ok")
        self.mem.store.close()
        mem_b = self.make(FakeEmbedder(model="fake-b"))
        self.assertEqual(mem_b.store.pending_embeddings("facts", "fake-b"), [])
        self.assertEqual(mem_b.store.pending_embeddings("turns", "fake-b"), [])


class TestCommands(Base):
    def test_parse(self):
        self.assertEqual(parse_memory_command("Recuerda que mi examen es el viernes."),
                         ("remember", "mi examen es el viernes."))
        self.assertEqual(parse_memory_command("Jarvis, recuerda que vivo en Lima"), ("remember", "vivo en Lima"))
        self.assertEqual(parse_memory_command("olvida lo de mi examen"), ("forget", "mi examen"))
        self.assertEqual(parse_memory_command("olvida lo del examen"), ("forget", "lo del examen"))
        self.assertEqual(parse_memory_command("¿Qué sabes de mí?"), ("list", ""))
        self.assertIsNone(parse_memory_command("recuerdas qué dije ayer"))
        self.assertIsNone(parse_memory_command("recuerda que"))
        self.assertIsNone(parse_memory_command("hola jarvis"))

    def test_handle_command_flow(self):
        self.assertEqual(self.mem.handle_command("hola"), None)
        self.assertIn("Todavía no sé", self.mem.handle_command("qué sabes de mí"))
        self.assertIn("Anotado", self.mem.handle_command("recuerda que mi examen de redes es el viernes"))
        self.assertIn("una cosa", self.mem.handle_command("qué sabes de mí"))
        self.assertIn("Olvidado", self.mem.handle_command("olvida lo del examen de redes"))
        self.assertIn("No encuentro", self.mem.handle_command("olvida lo del examen de redes"))
        self.assertIn("contraseñas", self.mem.handle_command("recuerda que mi contraseña es hunter2"))


class TestPinned(Base):
    SYS = [{"role": "system", "content": "SYS"}]

    def test_name_is_applied_every_turn_even_when_unrelated(self):
        self.assertIn("Tony", self.mem.handle_command("llámame Tony"))
        for q in ("hola", "cuéntame algo sobre subnetting ccna", "sí"):
            out = self.mem.augment(self.SYS, q)[0]["content"]
            self.assertIn("«Tony»", out, q)
            self.assertIn("Instrucciones de Dario", out)

    def test_new_name_replaces_old(self):
        self.mem.handle_command("llámame Tony")
        self.mem.handle_command("quiero que me llames Stark")
        out = self.mem.augment(self.SYS, "hola")[0]["content"]
        self.assertIn("Stark", out)
        self.assertNotIn("Tony", out)
        self.assertEqual(len([f for f in self.mem.facts() if f.source == "name"]), 1)

    def test_standing_instructions_are_pinned_plain_facts_are_not(self):
        self.mem.remember("responde siempre en frases cortas")
        self.mem.remember("Dario tiene examen de redes el viernes")
        pinned = [f.text for f in self.mem.store.pinned_facts()]
        self.assertEqual(pinned, ["responde siempre en frases cortas"])
        self.assertNotIn("examen", self.mem.augment(self.SYS, "hola")[0]["content"])

    def test_migration_pins_old_instructions(self):
        import sqlite3
        db = sqlite3.connect(self.path)
        db.execute("ALTER TABLE facts DROP COLUMN pinned")
        db.execute("INSERT INTO facts(text, source, created, updated) VALUES ('Dario quiere que lo llamen Tony','auto',1,1)")
        db.execute("INSERT INTO facts(text, source, created, updated) VALUES ('Dario estudia redes','auto',1,1)")
        db.commit(); db.close()
        self.mem.store.close()
        store = MemoryStore(self.path)
        self.assertEqual([f.text for f in store.pinned_facts()], ["Dario quiere que lo llamen Tony"])

    def test_name_parse(self):
        from memory.commands import parse_memory_command as parse
        self.assertEqual(parse("Jarvis, llámame Tony."), ("name", "Tony"))
        self.assertEqual(parse("llámame como Tony Stark"), ("name", "Tony Stark"))
        self.assertEqual(parse("quiero que me llames jefe"), ("name", "jefe"))
        self.assertIsNone(parse("llámame cuando termines de compilar el proyecto"))


class TestExtraction(unittest.TestCase):
    def test_parse_tolerates_fences_and_noise(self):
        raw = 'Claro:\n```json\n{"resumen": " Hablaron  de X ", "hechos": ["Dario usa Ollama a diario", "corto", 5, "Dario usa Ollama a diario"]}\n```'
        s, f = parse_extraction(raw)
        self.assertEqual(s, "Hablaron de X")
        self.assertEqual(f, ["Dario usa Ollama a diario"])

    def test_parse_garbage_and_secrets(self):
        self.assertEqual(parse_extraction("no es json"), ("", []))
        self.assertEqual(parse_extraction("{roto"), ("", []))
        self.assertEqual(parse_extraction("[1, 2]"), ("", []))
        _, f = parse_extraction('{"resumen": "", "hechos": ["La contraseña de Dario es abc123 en el banco"]}')
        self.assertEqual(f, [])


class TestVoiceLoopIntegration(Base):
    def test_memory_command_turn_and_recording(self):
        import numpy as np
        from core.voice_loop import VoiceLoop
        from stt.base import Transcript

        class LLM:
            def __init__(self): self.seen = []
            def stream(self, messages, **kw):
                self.seen.append(messages)
                yield "Hola Dario."

        def make_loop(text, llm):
            return VoiceLoop(listen=lambda: np.ones(1600, dtype=np.float32),
                             transcribe=lambda a: Transcript(text, "es", 1.0, 0.5), llm=llm, speaker=None,
                             history=[{"role": "system", "content": "SYS"}], memory=self.mem)

        llm = LLM()
        t = make_loop("Recuerda que mi examen de redes es el viernes", llm).run_turn()
        self.assertEqual(t.kind, "memory")
        self.assertIn("Anotado", t.reply)
        self.assertEqual(llm.seen, [])                       # la orden no llega al LLM
        loop = make_loop("cuándo es mi examen de redes", llm)
        t = loop.run_turn()
        self.assertEqual(t.kind, "reply")
        self.assertIn("examen de redes", llm.seen[0][0]["content"])   # recuerdo inyectado
        self.assertEqual(loop.history[0]["content"], "SYS")           # historial intacto
        self.assertEqual(self.mem.store.count("turns"), 1)            # turno guardado


if __name__ == "__main__":
    unittest.main()
