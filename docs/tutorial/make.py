#!/usr/bin/env python3
"""Builds the illustrated tutorial PDFs, one per language (the same method as LegatorOCR's
docs/guide-src: one HTML per language, printed to A4 PDF by a headless Chromium/Chrome).

  python3 docs/tutorial/make.py [lang ...]        # default: all eight languages
  python3 docs/tutorial/make.py --prepare SHOTS   # turn raw PNG screenshots in SHOTS/<lang>/ into img/<lang>/*.webp

Text: docs/tutorial/lang_<xx>.py (same keys in every language; a test of the structure is run first).
Images: docs/tutorial/img/<lang>/*.webp, real captures of the real window (see the honesty box).
Output: docs/tutorial/Tutorial-Colibri-Companion.<xx>.pdf
Needs: Python 3, Pillow (only for --prepare), and google-chrome or chromium on PATH.
Fonts: DejaVu Sans / DejaVu Sans Mono (they cover all eight languages, including Hungarian
double-acute o and u).
"""
import html
import importlib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LANGS = ["en", "ro", "fr", "it", "es", "pt", "de", "hu"]
e = html.escape

CSS = """
@page{size:A4;margin:16mm 15mm 18mm 15mm;@bottom-center{content:counter(page) " / " counter(pages);font:8pt "DejaVu Sans",sans-serif;color:#555}}
body{font-family:"DejaVu Sans","Liberation Sans",sans-serif;font-size:10pt;line-height:1.45;color:#111}
h1{font-size:23pt;margin:0 0 2mm;color:#0b6b57}
.sub{font-size:12.5pt;color:#444;margin-bottom:2mm}
.ver{font-size:8.5pt;color:#555;margin:0 0 6mm}
h2{font-size:14pt;margin:9mm 0 3mm;border-bottom:1px solid #bbb;padding-bottom:1mm;break-after:avoid;color:#0b6b57}
h3{font-size:11pt;margin:5mm 0 2mm;break-after:avoid}
p{margin:2mm 0}
ul{margin:2mm 0 2mm 5mm;padding:0}li{margin:1.2mm 0}
.toc{columns:2;column-gap:8mm;margin:0 0 4mm 0;padding:0}.toc li{list-style:none;font-size:9.5pt;break-inside:avoid}
figure{margin:4mm 0;break-inside:avoid}
figure img{display:block;width:100%;height:auto;border:1px solid #bbb;margin:0 auto}
figure.crop img{width:100%}
figcaption{font-size:8.5pt;color:#444;text-align:center;margin-top:1.5mm}
.step{display:flex;gap:3mm;margin:2mm 0;break-inside:avoid}
.step b{flex:0 0 7mm;height:7mm;line-height:7mm;border-radius:50%;background:#0b6b57;color:#fff;text-align:center;font-size:9pt}
.step span{flex:1}
.attn{border-left:4px solid #d08a00;background:#fff6e0;padding:2mm 3mm;margin:3mm 0;break-inside:avoid}
.note{font-size:9pt;color:#333}
table{border-collapse:collapse;width:100%;margin:3mm 0;break-inside:avoid;font-size:9pt}
th,td{border:1px solid #bbb;padding:1.8mm 2.2mm;vertical-align:top;text-align:left}th{background:#eee}
code{font-family:"DejaVu Sans Mono",monospace;font-size:8.6pt;background:#f1f1f1;padding:0 1mm}
footer.end{margin-top:8mm;border-top:1px solid #bbb;padding-top:2mm;font-size:8pt;color:#555}
"""

# which screenshot goes where: key -> (file, crop height in px or None, English-only?)
IMG = {
    "first": ("00-first", None, False),
    "choose": ("01-choose-model", 665, False),
    "licence": ("01b-license", None, False),
    "estimate": ("02-models-estimate", None, False),
    "installing": ("03-installing", 420, False),
    "ready": ("04-ready", None, False),
    "chat": ("05-chat", None, False),
    "docs": ("08-documents", None, False),
    "mcp": ("09-mcp", None, False),
    "auto": ("10-automations", None, False),
    "tools": ("07-resources-tools", None, False),
    "armed": ("07c-capture-armed", None, True),
    "confirm": ("07d-autotune-confirm", None, True),
}


def version():
    m = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M)
    return m.group(1) if m else "?"


def img(lang, key, cap, num):
    name, crop, en_only = IMG[key]
    d = lang if not en_only else "en"
    p = HERE / "img" / d / f"{name}.webp"
    if not p.exists():
        raise SystemExit(f"missing image {p}")
    cls = ' class="crop"' if crop else ""
    return f'<figure{cls}><img src="{p.as_uri()}" alt=""><figcaption>{e(cap)}</figcaption></figure>'


def pair(a, b):
    return a + b


def steps(items):
    return "".join(f'<div class="step"><b>{i}</b><span>{e(t)}</span></div>' for i, t in enumerate(items, 1))


def bullets(items):
    return "<ul>" + "".join(f"<li>{e(t)}</li>" for t in items) + "</ul>"


def table(head, rows):
    h = "".join(f"<th>{e(c)}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def page(lang, ver):
    T = importlib.import_module(f"lang_{lang}").T
    F = T["fig"]
    n = [0]

    def cap(text):
        n[0] += 1
        return f"{F} {n[0]}. {text}"

    def fig(key, text):
        return img(lang, key, cap(text), n[0])

    heads = [T[k] for k in ("honest_h", "s1_h", "s2_h", "s3_h", "s4_h", "s5_h", "s6_h", "s7_h", "s8_h", "s9_h", "s10_h", "s11_h", "s12_h", "s13_h")]
    toc = "".join(f"<li>{e(h)}</li>" for h in heads)
    o = [f'<!doctype html><html lang="{T["html_lang"]}"><head><meta charset="utf-8"><title>{e(T["title"])} ({ver})</title><style>{CSS}</style></head><body>']
    o.append(f'<h1>{e(T["title"])}</h1><div class="sub">{e(T["sub"])}</div><div class="ver">{e(T["ver"].replace("{VERSION}", ver))}</div>')
    o.append(f'<h3>{e(T["toc"])}</h3><ul class="toc">{toc}</ul>')
    o.append(f'<h2>{e(T["honest_h"])}</h2><div class="attn"><p>{e(T["honest_p"])}</p>{bullets(T["honest_li"])}</div>')
    o.append(f'<h2>{e(T["s1_h"])}</h2><p>{e(T["s1_p"])}</p>')
    o.append(f'<h2>{e(T["s2_h"])}</h2>{steps(T["s2_steps"])}{fig("first", T["s2_cap"])}')
    o.append(f'<h2>{e(T["s3_h"])}</h2><p>{e(T["s3_p"])}</p>{steps(T["s3_steps"])}{fig("choose", T["s3_cap"])}')
    o.append(f'<h2>{e(T["s4_h"])}</h2><p>{e(T["s4_p"])}</p>{bullets(T["s4_li"])}{fig("estimate", T["s4_cap"])}')
    o.append(f'<h2>{e(T["s5_h"])}</h2>{steps(T["s5_steps"])}{pair(fig("licence", T["s5_cap1"]), fig("installing", T["s5_cap2"]))}')
    o.append(f'<h2>{e(T["s6_h"])}</h2><p>{e(T["s6_p"])}</p>{bullets(T["s6_li"])}{pair(fig("ready", T["s6_cap1"]), fig("chat", T["s6_cap2"]))}')
    o.append(f'<h2>{e(T["s7_h"])}</h2><p>{e(T["s7_p"])}</p>{bullets(T["s7_li"])}')
    o.append(pair(fig("docs", T["s7_cap1"]), fig("mcp", T["s7_cap2"])))
    o.append(fig("auto", T["s7_cap3"]))
    o.append(f'<h2>{e(T["s8_h"])}</h2><p>{e(T["s8_p"])}</p>{steps(T["s8_steps"])}<p class="note">{e(T["s8_note"])}</p>')
    o.append(pair(fig("tools", T["s8_cap1"]), fig("armed", T["s8_cap2"])))
    o.append(f'<h2>{e(T["s9_h"])}</h2><p>{e(T["s9_p"])}</p>{steps(T["s9_steps"])}<p class="note">{e(T["s9_note"])}</p>{fig("confirm", T["s9_cap"])}')
    o.append(f'<h2>{e(T["s10_h"])}</h2><p>{e(T["s10_p"])}</p>{table(T["s10_head"], T["s10_rows"])}<p class="note">{e(T["s10_note"])}</p><p>{e(T["s10_agents"])}</p>')
    o.append(f'<h2>{e(T["s11_h"])}</h2><p>{e(T["s11_p"])}</p>{bullets(T["s11_li"])}')
    o.append(f'<h2>{e(T["s12_h"])}</h2>{table(T["s12_head"], T["s12_rows"])}')
    o.append(f'<h2>{e(T["s13_h"])}</h2><p>{e(T["s13_p"])}</p>{table(T["s13_head"], T["s13_rows"])}<p class="note">{e(T["s13_note"])}</p>')
    o.append(f'<footer class="end">{e(T["foot"])}</footer></body></html>')
    return "".join(o)


def chrome():
    for c in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if shutil.which(c):
            return c
    raise SystemExit("no Chrome/Chromium found")


def prepare(src):
    from PIL import Image
    src = pathlib.Path(src)
    for lang in LANGS:
        out = HERE / "img" / lang
        out.mkdir(parents=True, exist_ok=True)
        need = {v[0] for v in IMG.values()}
        for f in sorted((src / lang).glob("*.png")):
            if f.stem not in need:
                continue
            im = Image.open(f).convert("RGB")
            name = f.stem
            crop = {k: v[1] for k, v in IMG.items()}
            for key, (nm, ch, _) in IMG.items():
                if nm == name and ch:
                    im = im.crop((0, 0, im.width, ch))
            if im.width > 1100:
                im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
            im.save(out / f"{name}.webp", quality=82, method=6)
            print("wrote", out / f"{name}.webp")


def main(argv):
    if argv[:1] == ["--prepare"]:
        return prepare(argv[1])
    sys.path.insert(0, str(HERE))
    langs = argv or LANGS
    ver = version()
    exe = chrome()
    for lang in langs:
        doc = page(lang, ver)
        with tempfile.TemporaryDirectory() as td:
            h = pathlib.Path(td) / f"t-{lang}.html"
            h.write_text(doc, encoding="utf-8")
            out = HERE / f"Tutorial-Colibri-Companion.{lang}.pdf"
            subprocess.run([exe, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={out}", h.as_uri()], check=True, capture_output=True, timeout=180)
        print("wrote", out.name, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1:])
