# Datei des Unterhaltungsverlaufs

Der Unterhaltungsverlauf ist eine einfache JSON-Datei, damit Colibri Companion, CartaColibri und jedes Skript ihn austauschen können. In der App: Seitenleiste, „Verlauf: Export / Import“.

```json
{
  "format": "colibri-chat-history",
  "version": 1,
  "app": "colibri-companion",
  "exported": 1790000000,
  "conversations": [
    {
      "id": "c1790000000-000000001",
      "title": "…",
      "created": 1790000000,
      "updated": 1790000500,
      "messages": [
        { "role": "user", "content": "…" },
        { "role": "assistant", "content": "…", "reasoning": "…", "sources": ["…"] }
      ]
    }
  ]
}
```

## Regeln für Schreibende und Lesende

- `format` muss `colibri-chat-history` sein; `version` ist `1`. Ein Leser lehnt eine höhere Version ab.
- Zeiten sind Sekunden seit 1970 (UTC). `id` enthält nur Buchstaben, Ziffern, `-` und `_` (höchstens 80); alles andere wird entfernt.
- `role` ist `user`, `assistant` oder `system`. Optionale Nachrichtenfelder: `reasoning`, `display`, `sources` (Liste von Texten), `stopped` (true), `images` (nur `data:image/…`-URLs, und nur beim Export „mit Bildern“).
- Unbekannte Felder werden ignoriert, andere Programme können also eigene hinzufügen.
- Der Import fügt neue `id` hinzu und ersetzt eine bekannte `id` nur, wenn `updated` in der Datei neuer ist. Unterhaltungen ohne Nachrichten werden übersprungen.
- Grenzen beim Lesen: Datei 512 MB, 100.000 Unterhaltungen, 100.000 Nachrichten pro Unterhaltung.

Eine Verlaufsdatei auf das Fenster zu ziehen importiert sie.
