"""Métricas de reconocimiento: tasa de error de palabras (WER)."""
import re
import unicodedata


_UNI = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once",
        "doce", "trece", "catorce", "quince", "dieciseis", "diecisiete", "dieciocho", "diecinueve"]
_DEC = {2: "veinte", 3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta",
        8: "ochenta", 9: "noventa"}


def _num_words(n: int) -> list[str]:
    """0-99 en palabras (para que '8' y 'ocho' cuenten como lo mismo)."""
    if n < 20:
        return [_UNI[n]]
    d, u = divmod(n, 10)
    if d == 2 and u:
        return ["veinti" + _UNI[u]]
    return [_DEC[d]] + (["y", _UNI[u]] if u else [])


def normalize(text: str) -> list[str]:
    """Minúsculas, sin puntuación ni tildes: compara palabras, no ortografía."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    out = []
    for tok in re.findall(r"[a-z0-9]+", text):
        out.extend(_num_words(int(tok)) if tok.isdigit() and int(tok) < 100 else [tok])
    return out


def wer(reference: str, hypothesis: str) -> float:
    """WER = (sustituciones + inserciones + borrados) / palabras de la referencia."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)
