"""Mide el tool calling de qwen3:8b con num_ctx 4096 (Fase 6, subfase 6.0).

Uso (en Windows, con Ollama abierto y, para medir VRAM real, LLM + voz cargables):
    python scripts/bench_tools.py [--model qwen3:8b] [--sizes 3,5,8,13] [--repeats 1]
                                  [--skip-llm] [--skip-selector] [--out docs/fase6_bench_tools]

Qué mide:
1. LLM: con N herramientas en el prompt (siempre incluye la correcta; el resto, distractores
   fijos por semilla) -> ¿elige la herramienta correcta? ¿argumentos razonables? ¿llama una
   herramienta cuando no debe (charla normal)? tokens de prompt y latencia de la vuelta 1.
2. Segunda vuelta: tras devolver el resultado de la herramienta, tiempo al primer token y total.
3. Selector (bge-m3 en CPU): ¿la herramienta correcta cae en el top-k? latencia, y umbral
   sugerido que separa frases de herramienta de charla normal.
4. VRAM antes/después (nvidia-smi).
Escribe docs/fase6_bench_tools.json y .md. No usa datos personales: todo son frases sintéticas.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import gpu  # noqa: E402
from core.persona import SYSTEM_PROMPT  # noqa: E402
from memory.embeddings import EmbedError, load_embedder  # noqa: E402
from memory.text import strip_accents  # noqa: E402
from tools import ToolSelector, approx_tokens  # noqa: E402
from tools.prompt import tool_guide  # noqa: E402
from tools.unigym import SPECS  # noqa: E402

# (frase, herramienta esperada o None, {argumento: [subcadenas aceptables, sin tildes, en minúsculas]})
CASES = [
    ("¿Qué tengo hoy?", "consultar_agenda", {}),
    ("Jarvis, dime qué tengo mañana", "consultar_agenda", {"fecha": ["manana"]}),
    ("¿Qué tareas me faltan por entregar?", "tareas_pendientes", {}),
    ("¿Qué tareas pendientes tengo de EDA 2?", "tareas_pendientes", {"curso": ["eda"]}),
    ("¿Qué rutina de gym me toca hoy?", "rutina_gym", {}),
    ("¿Qué toca entrenar el sábado?", "rutina_gym", {"fecha": ["sabado"]}),
    ("¿Cómo voy en redes?", "notas_curso", {"curso": ["redes"]}),
    ("¿Cuál es mi promedio en EDA2?", "notas_curso", {"curso": ["eda"]}),
    ("¿Cuánto necesito en la próxima evaluación de redes para llegar a 15?", "nota_necesaria", {"curso": ["redes"], "meta": ["15"]}),
    ("¿Qué nota necesito en EDA 2 para aprobar?", "nota_necesaria", {"curso": ["eda"]}),
    ("¿Cuánta agua llevo hoy?", "progreso_diario", {}),
    ("¿Cuántos pasos he dado hoy?", "progreso_diario", {}),
    ("¿Qué hábitos me faltan hoy?", "estado_habitos", {}),
    ("Agrega una tarea de EDA2 para el viernes", "crear_tarea", {"fecha": ["viernes"], "curso": ["eda"]}),
    ("Anota que tengo que entregar el informe de redes el lunes", "crear_tarea", {"fecha": ["lunes"]}),
    ("Marca como hecha la tarea de EDA2", "completar_tarea", {"titulo": ["eda"]}),
    ("Ya terminé el informe de redes", "completar_tarea", {"titulo": ["informe"]}),
    ("Marca mi hábito de agua", "marcar_habito", {"habito": ["agua"]}),
    ("Ya hice el hábito de leer", "marcar_habito", {"habito": ["leer"]}),
    ("Registra que peso 70.5 kilos", "registrar_peso", {"kg": ["70.5"]}),
    ("Hoy pesé 71 kilos", "registrar_peso", {"kg": ["71"]}),
    ("Tomé un vaso de agua", "registrar_diario", {"campo": ["agua"]}),
    ("Dormí siete horas y media", "registrar_diario", {"campo": ["sueno"]}),
    ("Saqué 15 en el parcial de redes", "registrar_nota", {"curso": ["redes"], "nota": ["15"]}),
    # charla normal: no debe llamar herramientas
    ("Hola Jarvis, ¿cómo estás?", None, {}),
    ("Explícame qué es una VLAN en dos frases", None, {}),
    ("Cuéntame un chiste corto", None, {}),
    ("¿Cuál es la capital de Francia?", None, {}),
    ("Recuérdame cómo se llama el protocolo que traduce IP a MAC", None, {}),
    ("Gracias, eso es todo", None, {}),
]


# Frases NUEVAS: ninguna está entre los ejemplos de specs.py, así que el selector no las "conoce" de antemano.
HELD_OUT = [
    ("Dime qué clases tengo esta semana el miércoles", "consultar_agenda", {"fecha": ["miercoles"]}),
    ("¿Tengo algo programado para hoy?", "consultar_agenda", {}),
    ("¿Cuáles son mis pendientes?", "tareas_pendientes", {}),
    ("¿Qué me falta entregar de redes?", "tareas_pendientes", {"curso": ["redes"]}),
    ("¿Qué ejercicios hago hoy en el gimnasio?", "rutina_gym", {}),
    ("¿Hoy es día de pierna o de pecho?", "rutina_gym", {}),
    ("¿Cómo me está yendo en gestión financiera?", "notas_curso", {"curso": ["gestion"]}),
    ("Léeme mis notas del ciclo", "notas_curso", {}),
    ("¿Cuánto me falta sacar en el final de EDA2 para sacar 16?", "nota_necesaria", {"curso": ["eda"], "meta": ["16"]}),
    ("Si quiero 14 en redes, ¿qué nota me hace falta?", "nota_necesaria", {"curso": ["redes"], "meta": ["14"]}),
    ("¿Ya cumplí mi meta de pasos?", "progreso_diario", {}),
    ("¿Cuánto llevo de agua y de sueño hoy?", "progreso_diario", {}),
    ("¿Qué hábitos ya cumplí?", "estado_habitos", {}),
    ("Pon una tarea para el miércoles: estudiar para el parcial de redes", "crear_tarea", {"fecha": ["miercoles"]}),
    ("Recuérdame entregar el laboratorio de EDA2 el 30", "crear_tarea", {}),
    ("Termina la tarea del informe", "completar_tarea", {"titulo": ["informe"]}),
    ("Ya cumplí con el hábito de ejercicio", "marcar_habito", {"habito": ["ejercicio"]}),
    ("Anota mi peso de hoy: 69.8", "registrar_peso", {"kg": ["69.8"]}),
    ("Llevo 4 vasos de agua", "registrar_diario", {"campo": ["agua"]}),
    ("Hoy caminé 9000 pasos", "registrar_diario", {"campo": ["pasos"]}),
    ("En la EVA 2 de redes me pusieron 16", "registrar_nota", {"curso": ["redes"], "nota": ["16"]}),
    # charla normal cercana al dominio: no debe llamar herramientas
    ("Explícame cómo funciona el protocolo TCP", None, {}),
    ("¿Cuántas horas debería dormir una persona?", None, {}),
    ("Dame consejos para organizar mi tiempo de estudio", None, {}),
    ("¿Qué es la progresión de carga en el gimnasio?", None, {}),
    ("¿Qué significa tener una buena nota en una universidad?", None, {}),
    ("Cuéntame algo interesante sobre ciberseguridad", None, {}),
]

FAKE_MEMORY = ("Lo que sabes de Dario: estudia Ingeniería de Sistemas, quiere especializarse en ciberseguridad "
               "y redes, prefiere respuestas breves cuando hablas en voz alta, cursa CCNA 1, busca prácticas, "
               "usa una RTX 5060 de 8 GB, trabaja en un almacén, hace gym tres veces por semana. " * 2)

CANNED_RESULT = {
    "consultar_agenda": "Hoy: clase de Redes 08:00-10:00 en A-204; tareas: ninguna. (resultado de prueba)",
    "tareas_pendientes": "Tienes 2 tareas pendientes: informe de redes (viernes) y práctica de EDA2 (lunes). (resultado de prueba)",
    "rutina_gym": "Hoy te toca Pecho y tríceps: press banca 4 por 8 a 10; fondos 3 por 12. (resultado de prueba)",
    "notas_curso": "En Redes llevas 15 de promedio con 2 evaluaciones calificadas. Falta el 40 por ciento. (resultado de prueba)",
    "nota_necesaria": "Para llegar a 15 en Redes necesitas promediar 18 en lo que falta: EVA 2 (25%) y EVA 3 (25%). (resultado de prueba)",
    "progreso_diario": "Registro de hoy: Pasos: 5200 de 8000; Sueño: 7,5 horas de 8; Agua: 3 vasos de 8. (resultado de prueba)",
    "estado_habitos": "Hoy llevas 1 de 2 hábitos. Te faltan: Leer. (resultado de prueba)",
}


def norm(s) -> str:
    return strip_accents(str(s)).lower()


ALL_CASES = CASES + HELD_OUT


def system_message(with_memory: bool) -> dict:
    parts = [SYSTEM_PROMPT, tool_guide()]
    if with_memory:
        parts.append(FAKE_MEMORY)
    return {"role": "system", "content": "\n\n".join(parts)}


def chat(cfg, messages, tools=None, stream=False):
    llm = cfg["llm"]
    payload = {"model": llm["model"], "messages": messages, "stream": stream, "think": False,
               "keep_alive": llm.get("keep_alive", "10m"),
               "options": {"num_ctx": llm["options"]["num_ctx"], "temperature": 0}}
    if tools:
        payload["tools"] = tools
    return requests.post(f"{llm['host'].rstrip('/')}/api/chat", json=payload, timeout=300, stream=stream)


def pick_tools(case_idx, expected, n, seed):
    names = [s.name for s in SPECS]
    rng = random.Random(seed * 1000 + case_idx)
    chosen = [expected] if expected else []
    pool = [x for x in names if x != expected]
    rng.shuffle(pool)
    chosen += pool[: max(0, n - len(chosen))]
    rng.shuffle(chosen)
    by = {s.name: s for s in SPECS}
    return [by[x] for x in chosen]


def args_ok(args: dict, expect: dict) -> bool:
    for key, options in expect.items():
        if not any(o in norm(args.get(key, "")) for o in options):
            return False
    return True


def run_llm(cfg, sizes, repeats, with_memory):
    sysmsg = system_message(with_memory)
    rows = []
    chat(cfg, [sysmsg, {"role": "user", "content": "Hola"}]).json()  # calentamiento (carga el modelo)
    for n in sizes:
        res = {"n": n, "ok_tool": 0, "ok_args": 0, "pos": 0, "neg": 0, "false_pos": 0, "no_call": 0,
               "prompt_tokens": [], "latency_s": [], "schema_tokens": [], "failures": []}
        for rep in range(repeats):
            for i, (text, tool, expect) in enumerate(ALL_CASES):
                specs = pick_tools(i, tool, n, seed=rep)
                schemas = [s.ollama_schema() for s in specs]
                t0 = time.perf_counter()
                r = chat(cfg, [sysmsg, {"role": "user", "content": text}], schemas).json()
                dt = time.perf_counter() - t0
                msg = r.get("message", {})
                calls = msg.get("tool_calls") or []
                res["prompt_tokens"].append(r.get("prompt_eval_count", 0))
                res["latency_s"].append(dt)
                res["schema_tokens"].append(approx_tokens(schemas))
                if tool is None:
                    res["neg"] += 1
                    res["false_pos"] += bool(calls)
                    if calls:
                        res["failures"].append({"text": text, "expected": None, "got": calls[0]["function"]["name"]})
                else:
                    res["pos"] += 1
                    if not calls:
                        res["no_call"] += 1
                        res["failures"].append({"text": text, "expected": tool, "got": None})
                        continue
                    fn = calls[0]["function"]
                    good = fn["name"] == tool
                    res["ok_tool"] += good
                    ok_a = good and args_ok(fn.get("arguments") or {}, expect)
                    res["ok_args"] += ok_a
                    if not ok_a:
                        res["failures"].append({"text": text, "expected": tool, "got": fn["name"],
                                                "args": fn.get("arguments")})
        rows.append(res)
        print(f"  N={n:2d}: herramienta {res['ok_tool']}/{res['pos']}, args {res['ok_args']}/{res['pos']}, "
              f"falsos positivos {res['false_pos']}/{res['neg']}, mediana {statistics.median(res['latency_s']):.2f} s")
    return rows


def run_second_turn(cfg, n, with_memory):
    """Tras la herramienta: tiempo al primer token y total de la respuesta redactada por el LLM."""
    sysmsg = system_message(with_memory)
    out = []
    for text, tool, _ in ALL_CASES:
        if tool not in CANNED_RESULT:
            continue
        specs = pick_tools(0, tool, n, seed=0)
        schemas = [s.ollama_schema() for s in specs]
        r = chat(cfg, [sysmsg, {"role": "user", "content": text}], schemas).json()
        calls = r.get("message", {}).get("tool_calls") or []
        if not calls:
            continue
        msgs = [sysmsg, {"role": "user", "content": text}, r["message"],
                {"role": "tool", "tool_name": calls[0]["function"]["name"], "content": CANNED_RESULT[tool]}]
        t0, ttft, words = time.perf_counter(), None, 0
        with chat(cfg, msgs, schemas, stream=True) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                piece = d.get("message", {}).get("content", "")
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                words += len(piece.split())
        out.append({"text": text, "ttft_s": ttft or 0.0, "total_s": time.perf_counter() - t0, "words": words})
    return out


def run_selector(cfg):
    sel_cfg = (cfg.get("tools") or {}).get("selector", {})
    embedder = load_embedder(cfg["memory"]["embedder"])
    selector = ToolSelector(SPECS, embedder, top_k=max(5, sel_cfg.get("top_k", 4)))
    t0 = time.perf_counter()
    nt = selector.index()
    index_s = time.perf_counter() - t0
    known = {norm(e) for sp in SPECS for e in sp.examples}
    groups = {"conocidas": [], "nuevas": []}
    lat = []
    for text, tool, _ in ALL_CASES:
        t0 = time.perf_counter()
        ranked = selector.rank(text)
        lat.append(time.perf_counter() - t0)
        names = [x.spec.name for x in ranked]
        groups["conocidas" if norm(text) in known else "nuevas"].append(
            (text, tool, names, ranked[0].score if tool is None else next(x.score for x in ranked if x.spec.name == tool)))
    out = {"texts_indexed": nt, "index_s": index_s, "median_ms": statistics.median(lat) * 1000, "max_ms": max(lat) * 1000}
    for g, rows in groups.items():
        pos = [r for r in rows if r[1]]
        neg = [r for r in rows if not r[1]]
        out[g] = {"positives": len(pos), "negatives": len(neg),
                  "recall_at": {str(k): sum(r[1] in r[2][:k] for r in pos) for k in (3, 4, 5)},
                  "pos_score_min": min((r[3] for r in pos), default=None),
                  "neg_score_max": max((r[3] for r in neg), default=None),
                  "misses_top4": [r[0] for r in pos if r[1] not in r[2][:4]],
                  "neg_over_055": [r[0] for r in neg if r[3] > 0.55]}
    n = out["nuevas"]
    if n["pos_score_min"] is not None and n["neg_score_max"] is not None:
        out["suggested_min_score"] = round((n["pos_score_min"] + n["neg_score_max"]) / 2, 2)
    return out


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f} %)" if b else "-"


def write_report(path: Path, data: dict):
    path.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    L = [f"# Fase 6, 6.0: bench de tool calling ({data['model']}, num_ctx {data['num_ctx']})", "",
         f"Memoria simulada en el prompt: {'sí' if data['with_memory'] else 'no'}. "
         f"VRAM usada: {data['vram_before']} MB antes, {data['vram_after']} MB después (de {data['vram_total']} MB).", ""]
    if data.get("llm"):
        L += ["## Vuelta 1 (el LLM elige herramienta)", "",
              "| N herram. | tokens esquemas (aprox.) | tokens prompt (mediana) | herramienta correcta | args correctos | falsos positivos | latencia mediana | p95 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in data["llm"]:
            lat = sorted(r["latency_s"])
            p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
            L.append(f"| {r['n']} | {statistics.median(r['schema_tokens']):.0f} | {statistics.median(r['prompt_tokens']):.0f} | "
                     f"{pct(r['ok_tool'], r['pos'])} | {pct(r['ok_args'], r['pos'])} | {pct(r['false_pos'], r['neg'])} | "
                     f"{statistics.median(lat):.2f} s | {p95:.2f} s |")
        L.append("")
    if data.get("second_turn"):
        st = data["second_turn"]
        L += ["## Vuelta 2 (el LLM redacta con el resultado de la herramienta)", "",
              f"Muestras: {len(st['rows'])}. Primer token: mediana {st['ttft_median_s']:.2f} s. "
              f"Total: mediana {st['total_median_s']:.2f} s. Alternativa con plantilla de Python: ~0 s.", ""]
    if data.get("llm"):
        L += ["## Fallos de la vuelta 1", ""]
        for r in data["llm"]:
            L.append(f"- N={r['n']}: " + ("; ".join(f"«{f['text']}» esperaba {f['expected']}, dio {f['got']} {f.get('args', '')}"
                                                 for f in r["failures"]) or "sin fallos"))
        L.append("")
    if data.get("selector"):
        s = data["selector"]
        L += ["## Selector por embeddings (bge-m3, CPU)", "",
              f"Textos indexados: {s['texts_indexed']} (en {s['index_s']:.2f} s, una sola vez). "
              f"Latencia por consulta: mediana {s['median_ms']:.0f} ms, máx. {s['max_ms']:.0f} ms.", ""]
        for g, title in (("conocidas", "Frases que SÍ están en los ejemplos (optimista: el selector las conoce)"),
                         ("nuevas", "Frases NUEVAS (la medida que importa)")):
            x = s[g]
            L += [f"### {title}", "", f"Positivas: {x['positives']}, charla normal: {x['negatives']}.", "",
                  "| top-k | herramienta correcta dentro |", "|---|---|"]
            L += [f"| {k} | {pct(v, x['positives'])} |" for k, v in x["recall_at"].items()]
            L += ["", f"Parecido mínimo de una frase de herramienta: {x['pos_score_min']}; máximo de charla normal: {x['neg_score_max']}.",
                  f"Fuera del top-4: {x['misses_top4'] or 'ninguna'}. Charla con parecido > 0.55: {x['neg_over_055'] or 'ninguna'}.", ""]
        L += [f"Umbral sugerido con frases nuevas (`tools.selector.min_score`): {s.get('suggested_min_score')}.", ""]
    path.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--sizes", default="3,5,8,13")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-selector", action="store_true")
    ap.add_argument("--sin-memoria", action="store_true", help="prompt sin el bloque de memoria simulado")
    ap.add_argument("--out", default=str(ROOT / "docs" / "fase6_bench_tools"))
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if args.model:
        cfg["llm"]["model"] = args.model
    with_memory = not args.sin_memoria
    sizes = [int(x) for x in args.sizes.split(",")]
    data = {"model": cfg["llm"]["model"], "num_ctx": cfg["llm"]["options"]["num_ctx"], "with_memory": with_memory,
            "vram_before": gpu.vram_used_mb(), "vram_total": gpu.vram_total_mb()}
    try:
        if not args.skip_llm:
            print(f"Vuelta 1 con {data['model']}...")
            data["llm"] = run_llm(cfg, sizes, args.repeats, with_memory)
            print("Vuelta 2...")
            rows = run_second_turn(cfg, 5, with_memory)
            if rows:
                data["second_turn"] = {"rows": rows, "ttft_median_s": statistics.median(r["ttft_s"] for r in rows),
                                       "total_median_s": statistics.median(r["total_s"] for r in rows)}
        if not args.skip_selector:
            print("Selector por embeddings...")
            data["selector"] = run_selector(cfg)
    except (requests.RequestException, EmbedError) as e:
        sys.exit(f"No pude hablar con Ollama: {e}")
    data["vram_after"] = gpu.vram_used_mb()
    write_report(Path(args.out), data)
    print(f"Listo: {args.out}.md y .json")


if __name__ == "__main__":
    main()
