#!/usr/bin/env python3
"""Checks that the engine docs match the code. Exit 1 on any gap.
  --english-only   skip translation, README.<lang> and per-language checks
Checks: every `pub mod` of lib.rs in ENGINE.md+INNOVATIONS.md; every CLI flag (args.rs FLAGS,
every args.get/flag/all name, every --flag of the help text), every subcommand of the main.rs
dispatch, every env var read through std::env::var/var_os (CLI + library, excluding tests of
the host app) in QUICKSTART/ENGINE; no hard-coded test counts; README and index.html have a
corbul section; translations keep headings, code blocks and flag set of English."""
import re, sys, pathlib
english = "--english-only" in sys.argv
root = pathlib.Path(__file__).resolve().parent.parent
docs = root / "docs" / "engine"
langs = ["", ".ro", ".fr", ".it", ".es", ".pt", ".de", ".hu"]
names = ["ENGINE", "INNOVATIONS", "QUICKSTART", "CLEANROOM", "CHANGES-SECURITY"]
bad = []
def rd(p):
    try: return p.read_text(encoding="utf-8")
    except Exception: bad.append(f"missing file: {p.relative_to(root)}"); return ""
def cat(d): return "".join(rd(p) for p in sorted(d.glob("*.rs")))
mods = re.findall(r"^pub mod (\w+);", rd(root/"crates/engine/src/lib.rs"), re.M)
eng, inn, qs, cl, cs = (rd(docs/f"{n}.md") for n in names)
for m in mods:
    if not re.search(rf"\b{m}\b", eng) or not re.search(rf"\b{m}\b", inn + eng): bad.append(f"module `{m}` not documented")
cli = cat(root/"crates/engine-cli/src"); mainrs = rd(root/"crates/engine-cli/src/main.rs")
argsrs = rd(root/"crates/engine-cli/src/args.rs")
flags = set("--" + x for x in re.findall(r'"([a-z][a-z-]*)"', re.search(r"FLAGS: &\[&str\] = &\[(.*?)\];", argsrs, re.S).group(1)))
flags |= set("--" + x for x in re.findall(r'\b(?:args|a)\.(?:get|flag|all|num|has|opt|str|list|take)(?:::<[^>]+>)?\(\s*"([a-z][a-z0-9-]+)"', cli))
flags |= set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", mainrs))
flags -= {"--help", "--version"}
for f in sorted(flags):
    if not re.search(re.escape(f) + r"(?![\w-])", qs): bad.append(f"flag {f} not in QUICKSTART")
disp = re.search(r"match .*?\{(.*?)\n    \}", mainrs, re.S)
subs = set(re.findall(r'"([a-z][a-z-]+)"(?:\s*\|\s*"[a-z-]+")*\s*=>', mainrs))
for s in sorted(subs):
    if s not in ("help", "version") and not re.search(rf"corbul {re.escape(s)}\b", qs): bad.append(f"subcommand {s} has no `corbul {s}` example in QUICKSTART")
envs = set(re.findall(r'env::var(?:_os)?\(\s*"([A-Z][A-Z0-9_]+)"', "".join(rd(p) for p in (root/"crates").rglob("*.rs") if "/engine" in str(p) and "crates/app" not in str(p))))
envs |= set(re.findall(r'"(CORBUL_[A-Z_]+)"', cli))
for e in sorted(envs):
    if e not in qs + eng: bad.append(f"env var {e} not documented in QUICKSTART/ENGINE")
for n, t in zip(names, (eng, inn, qs, cl, cs)):
    for m in re.finditer(r"\b(\d{3,5}) (?:unit |integration )?tests\b", t): bad.append(f"{n}.md: hard-coded test count `{m.group(0)}`")
for n in names:
    base = rd(docs/f"{n}.md")
    h0 = len(re.findall(r"^#{1,4} ", base, re.M)); c0 = re.findall(r"```[\s\S]*?```", base)
    fl0 = set(re.findall(r"--[a-z][a-z0-9-]+", base))
    if english: continue
    for l in langs[1:]:
        t = rd(docs/f"{n}{l}.md")
        if not t: continue
        if len(re.findall(r"^#{1,4} ", t, re.M)) != h0: bad.append(f"{n}{l}.md: heading count differs from English")
        if len(re.findall(r"```[\s\S]*?```", t)) != len(c0): bad.append(f"{n}{l}.md: code block count differs")
        if set(re.findall(r"--[a-z][a-z0-9-]+", t)) != fl0: bad.append(f"{n}{l}.md: flag set differs from English")
for l in ([""] if english else langs):
    t = rd(root/f"README{l}.md")
    if not re.search(r"^#{2,3} .*corbul", t, re.I | re.M): bad.append(f"README{l}.md has no corbul section")
idx = rd(root/"index.html")
if not re.search(r'id="corbul"', idx): bad.append("index.html has no #corbul section")
print(f"{len(mods)} modules, {len(flags)} flags, {len(subs)} subcommands, {len(envs)} env vars checked")
for b in bad: print("GAP:", b)
sys.exit(1 if bad else 0)
