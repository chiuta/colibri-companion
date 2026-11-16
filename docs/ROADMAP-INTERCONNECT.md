# Interconnection roadmap: from a chat window to a capable assistant

Goal: connect the local model to everything that multiplies what it can do, while the person
stays in charge. Every step keeps three rules: **local first, nothing leaves without a yes,
every action can be seen and undone.**

## 0. The safety frame (built first, used by everything below)
- **Capability levels** (see `platforms.rs`): Read → Search → Write → Notify → Act. A connection
  is created at one level and can never exceed it.
- **Approval card** for every write, send and action: shows exactly what will happen, with
  *Allow once / Allow for this automation / Refuse*. Reading needs no card.
- **Untrusted content** (web pages, issues, e-mails, documents) is fenced and labelled as data.
  A tool call that follows untrusted content always asks again. This is the main defence against
  prompt injection.
- **Budgets and brakes**: steps, model calls, minutes and bytes per run; one *Stop everything* key.
- **Audit log** of every request that left the computer and every action taken, in a plain file.
- **Secrets**: tokens live in memory by default; saving them is an explicit choice.

## 1. Tools for the model (the multiplier)
Colibri exposes tool calling on most engines. The program offers the model a small, honest toolbox:
web page, search, repository file, folder search, calculator, Python in a sandbox, table query,
note search, save file, webhook. The model asks, the program checks level and approval, runs it,
and returns the result as data.

## 2. MCP: one plug for thousands of tools
Speak the Model Context Protocol as a client (and keep `coli mcp` as a server). Any MCP server
(files, databases, browsers, calendars, design tools, home devices) becomes available under the
same approval rules. This is the single biggest jump in reach for the least code.

## 3. Memory and knowledge that grow with you
- Local index (BM25 now, embeddings later) over folders, notes, saved chats, code, mail archives.
- Answers cite the passage they used; "show me where" opens it.
- Offline libraries: Wikipedia through Kiwix, arXiv and Wikidata snapshots, manuals, laws.
- The *constellation*: memories as linked stars (tags, people, projects) you can see and edit.

## 4. Senses and voice (all local)
- Hearing: whisper.cpp for dictation and meeting transcripts.
- Speaking: Piper or the system voice, per language of the interface.
- Seeing: models that take pictures (colibri lists which), plus OCR (Tesseract) for scans and
  photos of paper, plus the camera or a screenshot on demand.
- Making: Qwen-Image through colibri, ComfyUI or Stable Diffusion for images; audio and video tools
  as they appear.

## 5. Many minds
- **Duel**: blind A/B with Elo, to learn which model is best *for you*.
- **Council**: several models answer, a judge model merges and flags disagreement.
- **Router**: a small fast model for easy questions, the big one for hard ones, chosen by a
  decision model (colibri ships Laya, GLiNER2.5-Decide, Clef).
- **Specialists**: code, translation, vision, each picked automatically, each replaceable.

## 6. Exactness: let the model check itself
- Run the Python it writes in a sandbox and show the real output.
- Unit tests for code it produces, run before it is offered.
- Self-consistency (ask twice, compare), citations required for facts, a "show your sources"
  switch, and a small evaluation harness with confidence intervals (as LocalMind does).

## 7. Agents and automations
Saved chains of steps (read, ask, check, save, notify) with triggers: by hand, on a timetable,
when a folder changes, when a webhook arrives. Templates: morning digest, inbox triage draft,
repository watcher, meeting notes, research brief, weekly review. Each step shows its level.

## 8. Devices and the physical world (last, and most careful)
Home Assistant, MQTT, serial devices, 3D printers, cameras. Read-only by default; actions through
an allow-list; never locks, alarms or anything safety-critical.

## 9. Several computers, one brain
Share a model across the home network (a strong PC serves, a laptop uses it), sync prompts,
memory and settings between machines (encrypted, peer to peer, no cloud), and split work across
GPUs where the engine supports it.

## 10. Make it effortless
First-run guide that picks a model for the hardware, a *Simple mode* with five buttons, one-click
connections from the catalogue, "what can I do with this?" suggestions, a visible "what the model
can see right now" panel, and everything in eight languages.

## Order of work
1. Folders and big files, prompt library, answer styles (done).
2. Online access, GitHub/GitLab/Gitea/web pages, the catalogue (in progress).
3. Approval cards, audit log, budgets (the safety frame).
4. Automations with webhooks and notifications.
5. Duel, council, router.
6. Tools for the model, then MCP client.
7. Index and memory, voice and OCR.
8. Devices, multi-machine.

What is *not* promised: nothing here has been tested against real colibri engines with tool calling
yet, and each external service changes its API over time; entries marked "planned" in the
catalogue are plans, not features.
