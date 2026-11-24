"""Sirve UniGym Tracker y su archivo de datos compartido con JARVIS (reemplaza a `python -m http.server 8080`).

Uso:  python unigym_serve.py [--port 8080] [--web RUTA]     y abre http://localhost:8080
Los datos viven en data/unigym.json (créalo una vez con scripts/unigym_import.py). Si el archivo no existe,
la web funciona como antes, con localStorage.
"""
import argparse
import sys
from pathlib import Path

import yaml

from tools.unigym.repo import JsonFileRepository
from tools.unigym.server import make_server

ROOT = Path(__file__).resolve().parent


def main():
    cfg = (yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}).get("unigym") or {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=cfg.get("port", 8080))
    ap.add_argument("--web", default=str((ROOT / cfg.get("web_dir", "../uni-gym-tracker")).resolve()))
    a = ap.parse_args()
    web = Path(a.web)
    if not (web / "index.html").exists():
        sys.exit(f"No encuentro la web en {web}. Usa --web o unigym.web_dir en config.yaml.")
    repo = JsonFileRepository(ROOT / cfg.get("data_path", "data/unigym.json"), keep=cfg.get("backups_keep", 30))
    print("Datos:", "data/unigym.json" if repo.exists() else "NO hay data/unigym.json: la web usará localStorage")
    try:
        srv = make_server(repo, web, port=a.port)
    except OSError as e:
        sys.exit(f"No pude abrir el puerto {a.port} ({e}). ¿Sigue abierto 'python -m http.server 8080'? Ciérralo.")
    print(f"UniGym en http://localhost:{a.port}  (Ctrl+C para cerrar)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrado.")


if __name__ == "__main__":
    main()
