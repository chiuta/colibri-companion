# Accessibility

Colibri Companion is meant to be usable by as many people as possible. Everything below is
reachable from **Accessibility** in the top bar (or `Ctrl+Shift+A`).

## What is there

| Need | What the app does |
| --- | --- |
| Low vision | Text size from 80% to 300%; a high-contrast look; the Atkinson Hyperlegible typeface (SIL OFL); more space between lines and buttons; a thick outline around the item the keyboard is on |
| Blindness | The interface is exposed to screen readers through AccessKit; every answer can be read aloud with the computer's own voice (button, `Ctrl+Shift+S`, or automatically) |
| Colour blindness | A blue/orange palette in place of green/red in all three looks |
| Hands and movement | Buttons up to 64 points tall; the whole app works with the keyboard alone; no dragging or fine movement is ever required |
| Hearing | The app never makes a sound and never asks you to listen; the taskbar can flash when an answer is ready |
| Dyslexia, attention, cognitive load | Readable typeface, looser lines, no animation, simple menus, and a *plain language* answer style (short sentences, everyday words) |

Seven presets (low vision, blind, hard-to-control movements, dyslexia, easier reading, colour
blindness, hearing) set sensible bundles in one click; every option can then be changed on its own.

## How it is checked

* Every pair of text and background colour used by the interface, in all three looks and with the
  colour-blind palette on or off, must reach contrast 4.5 : 1 (WCAG AA for normal text). This is a unit
  test (`theme::tests::every_look_is_readable`), so a colour change that breaks it fails the build.
* Good/bad colours in the colour-blind palette must also differ in brightness, so they stay
  distinguishable in greyscale.
* The text sent to the speech program travels on standard input or in an environment variable, never
  inside the command line, so an answer can never be taken for an option (unit tested).

## What has **not** been verified

* The app has not been tried with a real screen reader (NVDA, JAWS, Narrator, VoiceOver, Orca). The
  accessibility tree comes from egui/AccessKit; icon-only buttons may be announced poorly. Reports are welcome.
* Reading aloud uses `espeak-ng`/`espeak`/`spd-say` on Linux, `say` on macOS and System.Speech through
  PowerShell on Windows. Only the command construction is tested; speech itself was not heard here. On macOS
  and Windows the voice language is the one chosen in the system.
* Speech *input* is left to the operating system's dictation (Windows `Win+H`, macOS `Fn` twice).
  Offline dictation inside the app (whisper.cpp) is planned in `ROADMAP-INTERCONNECT.md`.
* Switch-access and eye-tracking scanning are not implemented; operating-system tools work on top of the keyboard support.
