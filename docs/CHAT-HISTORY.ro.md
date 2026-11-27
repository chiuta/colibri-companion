# Fișierul istoricului de chat

Istoricul conversațiilor este un singur fișier JSON simplu, ca Colibri Companion, CartaColibri și orice script să-l poată schimba între ele. În aplicație: bara din stânga, „Istoric: export / import”.

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

## Reguli pentru cine scrie și cine citește

- `format` trebuie să fie `colibri-chat-history`; `version` este `1`. Un cititor refuză o versiune mai mare.
- Timpii sunt secunde de la 1970 (UTC). `id` conține doar litere, cifre, `-` și `_` (cel mult 80); restul se elimină.
- `role` este `user`, `assistant` sau `system`. Câmpuri opționale ale mesajului: `reasoning`, `display`, `sources` (listă de texte), `stopped` (true), `images` (doar URL-uri `data:image/…`, și doar la exportul „cu imagini”).
- Câmpurile necunoscute se ignoră, deci alte programe își pot adăuga ale lor.
- Importul adaugă id-urile noi și înlocuiește un id cunoscut doar dacă `updated` din fișier e mai nou. Conversațiile fără mesaje se sar.
- Limite la citire: fișier 128 MB, 20.000 de conversații, 100.000 de mesaje pe conversație.

Tragerea unui fișier de istoric peste fereastră îl importă.
