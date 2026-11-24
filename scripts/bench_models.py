"""Compara modelos de Ollama: velocidad, VRAM y respuestas de muestra.

Uso: python scripts/bench_models.py [modelo1 modelo2 ...]
Sin argumentos usa llm.candidates de config.yaml.
Los resultados se acumulan en docs/fase1_bench.json y se regenera docs/fase1_bench.md
tras CADA modelo, así que se pueden correr de uno en uno sin perder lo anterior.
"""
import json
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.persona import SYSTEM_PROMPT  # noqa: E402

HOST = "http://127.0.0.1:11434"
JSON_PATH = ROOT / "docs" / "fase1_bench.json"
MD_PATH = ROOT / "docs" / "fase1_bench.md"
PROMPTS = [
    "Preséntate en dos frases.",
    "Explícame en tres líneas qué es una VLAN, para alguien que cursa CCNA 1.",
    "Dame un plan de estudio de 3 pasos para el examen de redes de mañana.",
]


def installed():
    tags = requests.get(f"{HOST}/api/tags", timeout=10).json().get("models", [])
    return {m["name"] for m in tags}


def vram_gb(model):
    for m in requests.get(f"{HOST}/api/ps", timeout=10).json().get("models", []):
        if m["name"] == model or m["name"].split(":")[0] == model.split(":")[0]:
            return m.get("size_vram", 0) / 1024**3, m.get("size", 0) / 1024**3
    return None, None


def bench(model):
    tps_list, load, answers = [], 0.0, []
    for i, p in enumerate(PROMPTS, 1):
        print(f"   prompt {i}/{len(PROMPTS)}...", end=" ", flush=True)
        body = {"model": model, "stream": False, "keep_alive": "5m",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": p}]}
        if model.startswith("qwen3"):
            body["think"] = False  # sin modo 'pensar': latencia comparable para voz
        r = requests.post(f"{HOST}/api/chat", json=body, timeout=900)
        r.raise_for_status()
        d = r.json()
        tps = d["eval_count"] / (d["eval_duration"] / 1e9)
        tps_list.append(tps)
        if i == 1:
            load = d.get("load_duration", 0) / 1e9
        answers.append({"prompt": p, "answer": d["message"]["content"].strip()})
        print(f"{tps:.1f} tok/s")
    v, total = vram_gb(model)
    requests.post(f"{HOST}/api/chat", json={"model": model, "keep_alive": 0, "messages": []}, timeout=60)
    return {"tps": sum(tps_list) / len(tps_list), "load_s": load,
            "vram_gb": v, "size_gb": total, "answers": answers}


def write_md(results):
    f = lambda x: "?" if x is None else f"{x:.2f}"
    out = ["# Fase 1: benchmark de modelos\n",
           "| Modelo | tok/s (media) | Carga (s) | VRAM usada (GB) | Tamaño cargado (GB) |",
           "|---|---|---|---|---|"]
    for m, r in results.items():
        out.append(f"| {m} | {r['tps']:.1f} | {r['load_s']:.1f} | {f(r['vram_gb'])} | {f(r['size_gb'])} |")
    for m, r in results.items():
        out.append(f"\n## {m}\n")
        for a in r["answers"]:
            out.append(f"**{a['prompt']}**\n\n{a['answer']}\n")
    MD_PATH.write_text("\n".join(out), encoding="utf-8")


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    models = sys.argv[1:] or cfg["llm"].get("candidates", [])
    try:
        have = installed()
    except requests.RequestException:
        sys.exit("No puedo conectar con Ollama en 127.0.0.1:11434. ¿Está abierto?")
    results = json.loads(JSON_PATH.read_text(encoding="utf-8")) if JSON_PATH.exists() else {}
    try:
        for m in models:
            print(f"== {m}")
            if m not in have:
                print(f"   NO descargado. Corre: ollama pull {m}")
                continue
            t0 = time.time()
            try:
                results[m] = bench(m)
            except requests.RequestException as e:
                print(f"   fallo: {e}")
                continue
            print(f"   media {results[m]['tps']:.1f} tok/s, VRAM {results[m]['vram_gb']} GB "
                  f"({time.time() - t0:.0f} s)")
            JSON_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            write_md(results)
    except KeyboardInterrupt:
        print("\nInterrumpido. Lo ya medido quedó guardado.")
    print(f"\nTabla en {MD_PATH}")


if __name__ == "__main__":
    main()
