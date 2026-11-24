# Fase 0: base y hardware

## Hardware (24-nov-2026)
- GPU: NVIDIA GeForce RTX 5060, 8 GB VRAM (Blackwell, sm_120), driver 621.24
- CPU: AMD Ryzen 7 5700G (8 núcleos / 16 hilos)
- RAM: 15.9 GB
- Disco: solo C:, 74.5 GB libres
- SO: Windows 11 Pro

## Decisiones
- Python del proyecto: 3.12 en un venv (`.venv`). El sistema tiene 3.13, pero varias librerías de voz (Kokoro, Coqui TTS) aún no soportan 3.13.
- PyTorch con CUDA 12.8 (`cu128`): la RTX 5060 (Blackwell) no funciona con builds más viejos.
- No instalar CUDA Toolkit completo: Ollama y los wheels de PyTorch traen su runtime. Ahorra varios GB.
- Presupuesto de VRAM (8 GB): LLM 7B-8B cuantizado Q4 (~5 GB) + STT pequeño. TTS ligeros (Piper, Kokoro) en CPU o GPU. XTTS y F5-TTS no conviven con el LLM: se prueban solos en el laboratorio de voces.
- Presupuesto de disco: máximo ~35 GB para modelos; dejar el resto libre.

## Checklist
- [x] Python 3.12 instalado
- [x] venv `.venv` creado y activo
- [x] PyTorch cu128 instalado y `scripts/check_env.py` muestra GPU OK
- [x] Ollama instalado (`ollama --version`)
- [x] Repo en GitHub sin coautores no deseados

## Verificación final (24-nov-2026)
Python 3.12.10, torch 2.11.0+cu128, GPU CUDA OK (RTX 5060, sm_120, 7.9 GB), Ollama 0.40.18, git 2.54.0.
