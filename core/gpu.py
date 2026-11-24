"""Consulta de VRAM vía nvidia-smi (sin importar torch)."""
import subprocess


def _query(field: str) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def vram_used_mb() -> int | None:
    return _query("memory.used")


def vram_total_mb() -> int | None:
    return _query("memory.total")
