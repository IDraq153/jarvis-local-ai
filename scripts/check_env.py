"""Verifica el entorno de JARVIS: Python, GPU/CUDA y herramientas externas."""
import platform
import shutil
import subprocess
import sys


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception as e:
        return f"no disponible ({e.__class__.__name__})"


print(f"Python      : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
print(f"git         : {run(['git', '--version'])}")
print(f"ollama      : {run(['ollama', '--version']) if shutil.which('ollama') else 'no instalado'}")
print(f"nvidia-smi  : {run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']) if shutil.which('nvidia-smi') else 'no encontrado'}")

try:
    import torch
    print(f"torch       : {torch.__version__} (CUDA build: {torch.version.cuda})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU CUDA    : OK -> {p.name}, {p.total_memory / 1024**3:.1f} GB, capability sm_{p.major}{p.minor}")
        x = torch.randn(1024, 1024, device="cuda")
        print(f"Prueba GPU  : OK (matmul suma = {(x @ x).sum().item():.1f})")
    else:
        print("GPU CUDA    : NO disponible (torch instalado sin CUDA o driver incompatible)")
except ImportError:
    print("torch       : no instalado todavía")
