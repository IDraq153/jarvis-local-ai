"""Anti-alucinación de números (Fase 6.4): un número que el LLM pasa a una herramienta tiene que haberse DICHO.

En el bench, con 13 herramientas, qwen3:8b convirtió «anota que tengo que entregar el informe de redes el lunes» en
registrar_nota con nota 18: un número que nadie dijo. La confirmación por voz lo habría atrapado, pero es mejor no
llegar ahí. Regla: si Dario no dijo ningún número (ni en dígitos ni en palabras), se rechaza el argumento numérico; si
dijo dígitos y ninguno coincide con el valor (ni como «70 con 8» = 70,8 ni «7 y media» = 7,5), también.
Cuando solo hay números en palabras no se puede verificar y se deja pasar (la confirmación repite el valor).
"""
import re

from memory.text import strip_accents

NUMERIC_KEYS = ("kg", "valor", "nota", "meta")
_WORDS = {"cero", "un", "una", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once",
          "doce", "trece", "catorce", "quince", "dieciseis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veinti",
          "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa", "cien", "ciento", "doscientos",
          "trescientos", "quinientos", "mil", "medio", "media", "par", "docena"}
_DECIMAL = {"con", "coma", "punto"}
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w.,]+", strip_accents(text or "").lower()) if t]


def candidates(heard: str) -> tuple[list[float], bool]:
    """(valores numéricos que se pueden formar con los dígitos dichos, ¿hay números en palabras?)"""
    toks = _tokens(heard)
    vals, words = [], False
    nums = []                                  # (índice, valor, texto)
    for i, t in enumerate(toks):
        t = t.strip(".,")
        if _NUM.fullmatch(t):
            nums.append((i, float(t.replace(",", ".")), t))
            vals.append(float(t.replace(",", ".")))
        elif t.startswith(tuple(_WORDS)) and (t in _WORDS or t.rstrip("s") in _WORDS or t.startswith("veinti")):
            words = True
    for (i, a, _), (j, b, bt) in zip(nums, nums[1:]):      # "70 con 8" -> 70,8
        if j == i + 2 and toks[i + 1] in _DECIMAL and "." not in bt and "," not in bt:
            vals.append(a + b / (10 ** len(bt)))
    for i, a, _ in nums:                                   # "7 y media" -> 7,5
        if toks[i + 1:i + 3] == ["y", "media"] or toks[i + 1:i + 3] == ["y", "medio"]:
            vals.append(a + 0.5)
    return vals, words


def to_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def grounding_problem(args: dict, heard: str) -> str | None:
    """Pregunta para Dario si algún argumento numérico no se oyó; None si todo está respaldado (o no hay frase)."""
    if not heard or not isinstance(args, dict):
        return None                                   # llamada programática: no hay frase que comprobar
    vals, words = candidates(heard)
    for key in NUMERIC_KEYS:
        if args.get(key) is None:
            continue
        x = to_float(args[key])
        if x is None:
            continue                                  # el validador de cada herramienta lo rechaza
        if not vals and not words:
            return "No te oí ningún número. ¿Cuánto es?"
        if vals and not words and not any(abs(x - v) < 0.011 for v in vals):
            return "No estoy seguro del número que dijiste. ¿Me lo repites?"
    return None


_TITLE_STOP = {"de", "del", "la", "las", "el", "los", "y", "en", "a", "al", "un", "una", "para", "mi", "mis", "con", "por", "que", "tarea"}


def title_is_grounded(titulo, heard: str) -> bool:
    """¿El título que pasó el LLM sale de lo que Dario dijo? Si no ("estudiar" cuando solo dijo «tarea para el curso redes»)
    el LLM lo inventó y hay que preguntar. Compara por raíz (5 letras) para tolerar «estudiar/estudiaré»."""
    toks = [t for t in re.split(r"[^\w]+", strip_accents(str(titulo or "")).lower()) if t and t not in _TITLE_STOP]
    if not toks:
        return True
    said = [t for t in re.split(r"[^\w]+", strip_accents(heard or "").lower()) if t]
    return any(t == h or (len(t) >= 5 and len(h) >= 5 and t[:5] == h[:5]) for t in toks for h in said)
