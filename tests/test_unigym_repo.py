"""Pruebas de la capa de datos de UniGym (Fase 6.1). Solo datos sintéticos."""
import datetime as dt
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tools.unigym.repo import (APP_ID, JsonFileRepository, LockTimeout, RevisionConflict, UniGymDataError,
                               extract_state, normalize_state, validate_state)


def synthetic_state():
    return {
        "version": 1,
        "settings": {"theme": "dark", "userName": "Test", "goals": {"steps": 8000, "sleep": 8, "water": 8},
                     "campoNuevoDeLaWeb": {"x": 1}},
        "courses": [{"id": "cur_a", "name": "Curso A", "credits": 3, "evals": []}],
        "gym": {"routines": [], "sessions": [], "activeSession": None, "bodyWeight": []},
        "dailyLogs": {}, "schedule": {"blocks": []},
        "tasks": [{"id": "tar_1", "title": "Tarea uno", "category": "Universidad", "priority": "media",
                   "due": "2026-11-27", "done": False, "createdAt": 1}],
        "habits": [{"id": "hab_1", "name": "Agua", "log": {}}],
        "record": {"career": "", "ranking": "", "cycles": []},
        "claveDesconocida": [1, 2, 3],
    }


def backup_envelope():
    return {"app": APP_ID, "version": 1, "exportedAt": "2026-11-24T00:00:00Z", "data": synthetic_state()}


class RepoBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.src = self.dir / "respaldo.json"
        self.src.write_text(json.dumps(backup_envelope()), encoding="utf-8")
        self.path = self.dir / "data" / "unigym.json"
        self.repo = JsonFileRepository.from_backup(self.src, self.path, keep=5)

    def tearDown(self):
        self._tmp.cleanup()

    def backups(self):
        return sorted((self.dir / "data" / "backups").glob("unigym-*.json"))


class ImportAndRead(RepoBase):
    def test_import_backup_sets_revision_one_and_keeps_unknown_keys(self):
        snap = self.repo.read()
        self.assertEqual(snap.revision, 1)
        self.assertEqual(snap.state["claveDesconocida"], [1, 2, 3])
        self.assertEqual(snap.state["settings"]["campoNuevoDeLaWeb"], {"x": 1})
        self.assertEqual(snap.by, "import")

    def test_import_refuses_to_overwrite_unless_asked(self):
        with self.assertRaises(UniGymDataError):
            JsonFileRepository.from_backup(self.src, self.path)
        JsonFileRepository.from_backup(self.src, self.path, overwrite=True)
        self.assertEqual(len(self.backups()), 1)  # se respaldó lo anterior

    def test_import_accepts_raw_state_and_rejects_junk(self):
        raw = self.dir / "raw.json"
        raw.write_text(json.dumps(synthetic_state()), encoding="utf-8")
        JsonFileRepository.from_backup(raw, self.dir / "otro.json")
        junk = self.dir / "junk.json"
        junk.write_text(json.dumps({"hola": 1}), encoding="utf-8")
        with self.assertRaises(UniGymDataError):
            JsonFileRepository.from_backup(junk, self.dir / "x.json")
        junk.write_text("no es json", encoding="utf-8")
        with self.assertRaises(UniGymDataError):
            JsonFileRepository.from_backup(junk, self.dir / "x.json")

    def test_read_returns_independent_copy(self):
        snap = self.repo.read()
        snap.state["tasks"].clear()
        self.assertEqual(len(self.repo.read().state["tasks"]), 1)

    def test_missing_file_has_helpful_error(self):
        with self.assertRaises(UniGymDataError) as cm:
            JsonFileRepository(self.dir / "nada.json").read()
        self.assertIn("respaldo", str(cm.exception))

    def test_corrupt_file_is_reported_not_crashed(self):
        self.path.write_text("{no json", encoding="utf-8")
        with self.assertRaises(UniGymDataError) as cm:
            self.repo.read()
        self.assertIn("dañado", str(cm.exception))


class Writing(RepoBase):
    def test_write_bumps_revision_and_persists(self):
        snap = self.repo.write(lambda s: s["tasks"].append(
            {"id": "tar_2", "title": "Nueva", "category": "Personal", "priority": "alta", "due": "", "done": False,
             "createdAt": 2}), reason="prueba")
        self.assertEqual(snap.revision, 2)
        again = self.repo.read()
        self.assertEqual([t["id"] for t in again.state["tasks"]], ["tar_1", "tar_2"])
        self.assertEqual(again.state["claveDesconocida"], [1, 2, 3])   # lo que no conoce, se conserva
        env = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual((env["app"], env["revision"], env["by"], env["reason"]), (APP_ID, 2, "jarvis", "prueba"))

    def test_mutator_may_return_a_whole_new_state(self):
        new = synthetic_state()
        new["tasks"] = []
        self.assertEqual(self.repo.write(lambda s: new, by="web").revision, 2)
        self.assertEqual(self.repo.read().state["tasks"], [])

    def test_backup_holds_previous_state(self):
        self.repo.write(lambda s: s["tasks"].clear())
        (b,) = self.backups()
        old = json.loads(b.read_text(encoding="utf-8"))
        self.assertEqual(old["revision"], 1)
        self.assertEqual(len(old["data"]["tasks"]), 1)

    def test_backups_are_pruned_to_keep(self):
        for i in range(9):
            self.repo.write(lambda s, i=i: s["settings"].update(n=i))
        self.assertEqual(len(self.backups()), 5)
        self.assertTrue(self.backups()[-1].name.endswith("-r9.json"))

    def test_expected_revision_conflict_leaves_file_untouched(self):
        self.repo.write(lambda s: s["settings"].update(a=1))            # rev 2
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(RevisionConflict) as cm:
            self.repo.write(lambda s: s["settings"].update(a=2), expected_revision=1)
        self.assertEqual((cm.exception.expected, cm.exception.current), (1, 2))
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.repo.write(lambda s: None, expected_revision=2).revision, 3)

    def test_failing_mutator_writes_nothing(self):
        before = self.path.read_text(encoding="utf-8")

        def boom(s):
            s["tasks"].clear()
            raise ValueError("fallo")
        with self.assertRaises(ValueError):
            self.repo.write(boom)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.backups(), [])

    def test_invalid_result_is_rejected_before_writing(self):
        before = self.path.read_text(encoding="utf-8")
        bad = [
            lambda s: s["tasks"].append({"id": "x", "title": ""}),
            lambda s: s["tasks"].append({"id": "x", "title": "ok", "due": "viernes"}),
            lambda s: s["gym"]["bodyWeight"].append({"id": "p", "date": "2026-11-24", "kg": "70"}),
            lambda s: s["dailyLogs"].update({"ayer": {}}),
            lambda s: s["tasks"].append({"id": "x", "title": "ok", "extra": {1, 2}}),   # no serializable
            lambda s: s.update(tasks="no es lista"),                  # no debe convertirse en [] y borrar tareas
            lambda s: s["gym"].update(bodyWeight={"a": 1}),
            lambda s: s["courses"][0].update(evals="x"),
            lambda s: s["habits"][0].update(log=[1]),
            lambda s: [1, 2],
        ]
        for mut in bad:
            with self.assertRaises(UniGymDataError):
                self.repo.write(mut)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual(len(self.repo.read().state["tasks"]), 1)

    def test_no_temp_or_lock_files_left_behind(self):
        self.repo.write(lambda s: None)
        names = {p.name for p in (self.dir / "data").iterdir()}
        self.assertEqual(names, {"unigym.json", "backups"})


class Locking(RepoBase):
    def test_live_lock_times_out(self):
        lock = self.path.with_name("unigym.json.lock")
        lock.write_text("999")
        repo = JsonFileRepository(self.path, lock_timeout=0.2, stale_lock_s=60)
        with self.assertRaises(LockTimeout):
            repo.write(lambda s: None)
        self.assertEqual(self.repo.read().revision, 1)

    def test_stale_lock_is_taken_over(self):
        lock = self.path.with_name("unigym.json.lock")
        lock.write_text("999")
        old = time.time() - 120
        os.utime(lock, (old, old))
        repo = JsonFileRepository(self.path, lock_timeout=1, stale_lock_s=30)
        self.assertEqual(repo.write(lambda s: None).revision, 2)

    def test_concurrent_writers_lose_no_updates(self):
        def bump(s):
            s["settings"]["contador"] = s["settings"].get("contador", 0) + 1

        def worker():
            repo = JsonFileRepository(self.path, keep=3, lock_timeout=20)   # un objeto por hilo, como dos procesos
            for _ in range(15):
                repo.write(bump)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        snap = self.repo.read()
        self.assertEqual(snap.state["settings"]["contador"], 60)
        self.assertEqual(snap.revision, 61)


class Shape(unittest.TestCase):
    def test_normalize_fills_containers_and_keeps_extras(self):
        s = normalize_state({"courses": [{"name": "X"}], "habits": [{"name": "h"}], "foo": 1})
        self.assertEqual(s["courses"][0]["evals"], [])
        self.assertEqual(s["habits"][0]["log"], {})
        self.assertEqual((s["tasks"], s["dailyLogs"], s["gym"]["bodyWeight"]), ([], {}, []))
        self.assertEqual(s["foo"], 1)
        validate_state(s)

    def test_extract_state_variants(self):
        self.assertIn("tasks", extract_state(backup_envelope()))
        self.assertIn("tasks", extract_state(synthetic_state()))
        with self.assertRaises(UniGymDataError):
            extract_state([1, 2])


if __name__ == "__main__":
    unittest.main()
