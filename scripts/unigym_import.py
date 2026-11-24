"""Crea data/unigym.json (fuente de verdad de UniGym) desde un respaldo exportado por la web.

Uso:  python scripts/unigym_import.py RUTA_DEL_RESPALDO.json [--overwrite]
Valida el contenido antes de crear nada y solo imprime recuentos, nunca tus datos.
Si data/unigym.json ya existe, no lo toca salvo con --overwrite (y entonces guarda copia previa).
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.unigym.repo import JsonFileRepository, UniGymDataError  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backup")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    path = ROOT / ((cfg.get("unigym") or {}).get("data_path", "data/unigym.json"))
    try:
        repo = JsonFileRepository.from_backup(a.backup, path, overwrite=a.overwrite)
    except UniGymDataError as e:
        sys.exit(f"No se creó nada: {e}")
    s = repo.read().state
    print(f"Listo: {path}")
    print(f"  {len(s['courses'])} cursos, {len(s['tasks'])} tareas, {len(s['habits'])} hábitos, "
          f"{len(s['schedule']['blocks'])} bloques de horario, {len(s['gym']['sessions'])} sesiones de gym, "
          f"{len(s['gym']['bodyWeight'])} pesos, {len(s['dailyLogs'])} días de registro.")


if __name__ == "__main__":
    main()
