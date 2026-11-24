"""Mide modelos de embeddings para la memoria (Fase 5): latencia, acierto y umbral sugerido.

Uso: python scripts/bench_embeddings.py [modelo ...]   (por defecto los de config.yaml)
Antes: ollama pull <modelo>. Corre en CPU igual que JARVIS (memory.embedder.device).
Por cada consulta compara el parecido con su hecho correcto frente al mejor hecho
equivocado; el umbral sugerido (min_fact_score) separa ambos grupos.
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.embeddings import EmbedError, OllamaEmbedder  # noqa: E402

FACTS = [
    "Dario estudia Ingeniería de Sistemas en la Universidad de Lima",
    "Dario quiere especializarse en ciberseguridad y redes",
    "Dario cursa CCNA 1 con Cisco Networking Academy",
    "Dario busca prácticas preprofesionales",
    "Dario usa una RTX 5060 con 8 GB de VRAM",
    "Dario prefiere respuestas breves cuando JARVIS habla en voz alta",
    "Dario tiene examen de redes el viernes",
    "Dario trabaja en un almacén y elabora PECOSAs",
    "Dario hace gimnasio tres veces por semana",
    "Dario está haciendo un mod de Minecraft con una isla de fiestas",
]
# (consulta, índice del hecho correcto) con otras palabras, como se habla de verdad
QUERIES = [
    ("¿qué carrera estoy estudiando?", 0), ("quiero dedicarme a la seguridad informática", 1),
    ("¿en qué curso de Cisco estoy?", 2), ("necesito un trabajo para practicar", 3),
    ("¿cuánta memoria de video tiene mi tarjeta gráfica?", 4), ("no me hables tanto cuando uses la voz", 5),
    ("¿cuándo es mi evaluación de networking?", 6), ("¿qué hago en mi empleo con la mercadería?", 7),
    ("¿cuántos días entreno?", 8), ("cuéntame de mi proyecto de videojuego con fiestas", 9),
]
OFF_TOPIC = ["¿cuál es la capital de Francia?", "explícame cómo funciona un motor diésel",
             "dime un chiste sobre pingüinos"]


def main():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
    ecfg = cfg["memory"]["embedder"]
    models = sys.argv[1:] or [ecfg["model"], "embeddinggemma", "nomic-embed-text"]
    for model in models:
        emb = OllamaEmbedder(model=model, host=ecfg["host"], device=ecfg.get("device", "cpu"))
        try:
            emb.embed(["calentamiento"])
            f = emb.embed(FACTS)
            lat = []
            for q, _ in QUERIES:
                t = time.perf_counter()
                emb.embed([q])
                lat.append((time.perf_counter() - t) * 1000)
            q = emb.embed([x for x, _ in QUERIES])
            off = emb.embed(OFF_TOPIC)
        except EmbedError as e:
            print(f"{model}: {e}\n")
            continue
        sims = q @ f.T
        right = [sims[i, j] for i, (_, j) in enumerate(QUERIES)]
        wrong = [max(s for k, s in enumerate(sims[i]) if k != j) for i, (_, j) in enumerate(QUERIES)]
        hits = sum(int(np.argmax(sims[i]) == j) for i, (_, j) in enumerate(QUERIES))
        off_top = [float((off @ f.T)[i].max()) for i in range(len(OFF_TOPIC))]
        lo, hi = min(right), max(wrong + off_top)
        print(f"{model}: dim {f.shape[1]}, latencia mediana {statistics.median(lat):.0f} ms (p. máx {max(lat):.0f}), "
              f"acierto top-1 {hits}/{len(QUERIES)}")
        print(f"  parecido con el hecho correcto: mín {lo:.2f}, mediana {statistics.median(right):.2f}")
        print(f"  mejor hecho equivocado / consulta sin relación: máx {hi:.2f}")
        if lo > hi:
            print(f"  umbral sugerido min_fact_score: {(lo + hi) / 2:.2f} (separa los grupos)\n")
        else:
            print(f"  los grupos se solapan: sin umbral limpio. Sugerido min_fact_score: {max(lo - 0.02, 0):.2f} "
                  f"(justo bajo el peor acierto; algún hecho de más es mejor que olvidar uno útil)\n")


if __name__ == "__main__":
    main()
