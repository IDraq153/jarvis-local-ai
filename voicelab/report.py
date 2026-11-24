"""Genera voicelab/samples/index.html: tabla de métricas y un reproductor por frase y voz."""
from html import escape
from pathlib import Path


def build_index(results: dict, samples_dir: Path) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid, r in sorted(results.items(), key=lambda kv: (kv[1]["backend"], kv[0])):
        ps = r["phrases"]
        avg = lambda k: sum(p[k] for p in ps) / len(ps)
        players = "".join(
            f"<div class='ph'><small>{i}. {escape(p['text'][:60])}…</small>"
            f"<audio controls preload='none' src='{escape(p['wav'])}'></audio>"
            f"<small>1.er audio {p['first_audio_s']} s · RTF {p['rtf']}</small></div>"
            for i, p in enumerate(ps, 1))
        rows.append(
            f"<tr><td><b>{escape(r['label'])}</b><br><small>{escape(vid)}</small></td>"
            f"<td>{escape(r['backend'])}</td><td>{r['load_s']} s</td>"
            f"<td>{avg('first_audio_s'):.2f} s</td><td>{avg('rtf'):.2f}</td>"
            f"<td>{'+' + str(r['vram_delta_mb']) + ' MB' if r.get('vram_delta_mb') is not None else 'n/d'}</td>"
            f"<td>{players}</td></tr>")
    page = f"""<!doctype html><meta charset="utf-8"><title>JARVIS · laboratorio de voces</title>
<style>body{{font:15px system-ui;margin:2rem;background:#111;color:#eee}}
table{{border-collapse:collapse}}td,th{{border:1px solid #333;padding:.5rem;vertical-align:top}}
th{{background:#1c1c1c}}.ph{{margin-bottom:.6rem}}audio{{display:block;width:260px}}small{{color:#999}}</style>
<h1>Laboratorio de voces</h1>
<p>RTF &lt; 1 = más rápido que tiempo real. VRAM = pico durante carga y síntesis menos la VRAM base (el LLM cuenta en la base si estaba cargado).</p>
<table><tr><th>Voz</th><th>Motor</th><th>Carga</th><th>1.er audio (media)</th><th>RTF (media)</th><th>VRAM</th><th>Muestras</th></tr>
{''.join(rows)}</table>"""
    out = samples_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out
