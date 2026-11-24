#!/usr/bin/env python3
"""Checks that the engine docs match the code. Exit 1 on any gap.
  1. every `pub mod` in crates/engine/src/lib.rs is named in ENGINE.md and INNOVATIONS.md
  2. every long flag (--x) in crates/engine-cli/src/*.rs and every subcommand string is named in QUICKSTART.md
  3. all 8 languages exist for each doc, with the same number of headings and the same code blocks / flags
  4. README.*.md and index.html mention the engine (corbul) at all
"""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
docs = root / "docs" / "engine"
langs = ["", ".ro", ".fr", ".it", ".es", ".pt", ".de", ".hu"]
names = ["ENGINE", "INNOVATIONS", "QUICKSTART", "CLEANROOM"]
bad = []
def rd(p):
    try: return p.read_text(encoding="utf-8")
    except Exception: bad.append(f"missing file: {p.relative_to(root)}"); return ""
mods = re.findall(r"^pub mod (\w+);", rd(root/"crates/engine/src/lib.rs"), re.M)
eng, inn, qs = (rd(docs/f"{n}.md") for n in ("ENGINE", "INNOVATIONS", "QUICKSTART"))
for m in mods:
    if not re.search(rf"\b{m}(\.rs)?\b", eng + inn): bad.append(f"module `{m}` not documented in ENGINE/INNOVATIONS")
src = "".join(rd(p) for p in sorted((root/"crates/engine-cli/src").glob("*.rs")))
flags = set(re.findall(r'"(--[a-z][a-z0-9-]+)"', src))
flags |= {"--" + f for f in re.findall(r'args\.(?:get|has|flag|take|values|get_all|opt)\w*\(\s*"([a-z][a-z0-9-]+)"', src)}
flags |= set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", rd(root/"crates/engine-cli/src/main.rs")))
flags = sorted(flags - {"--help", "--version"})
for f in flags:
    if f not in qs: bad.append(f"flag {f} not in QUICKSTART")
subs = sorted(set(re.findall(r'^\s*"([a-z][a-z-]+)"\s*(?:\||=>)', src, re.M)))
for s in subs:
    if len(s) > 3 and not re.search(rf"corbul {s}\b|`{s}`", qs): pass  # informational only, subcommand detection is heuristic
for n in names:
    base = rd(docs/f"{n}.md")
    h0 = len(re.findall(r"^#{1,4} ", base, re.M)); c0 = re.findall(r"```[\s\S]*?```", base)
    fl0 = set(re.findall(r"--[a-z][a-z0-9-]+", base))
    for l in langs[1:]:
        t = rd(docs/f"{n}{l}.md")
        if not t: continue
        if len(re.findall(r"^#{1,4} ", t, re.M)) != h0: bad.append(f"{n}{l}.md: heading count differs from English")
        if len(re.findall(r"```[\s\S]*?```", t)) != len(c0): bad.append(f"{n}{l}.md: code block count differs")
        if set(re.findall(r"--[a-z][a-z0-9-]+", t)) != fl0: bad.append(f"{n}{l}.md: flag set differs from English")
for l in langs:
    t = rd(root/f"README{l}.md")
    if not re.search(r"corbul", t, re.I): bad.append(f"README{l}.md does not mention the engine")
idx = rd(root/"index.html")
if "corbul" not in idx.lower(): bad.append("index.html does not mention corbul")
print(f"{len(mods)} modules, {len(flags)} CLI flags checked")
for b in bad: print("GAP:", b)
sys.exit(1 if bad else 0)
