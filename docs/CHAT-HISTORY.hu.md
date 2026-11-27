# A beszélgetési előzmények fájlja

A beszélgetési előzmények egyetlen egyszerű JSON-fájl, hogy a Colibri Companion, a CartaColibri és bármely szkript cserélhesse. Az alkalmazásban: az oldalsáv, „Előzmények: export / import”.

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

## Szabályok az író és az olvasó félnek

- A `format` értéke `colibri-chat-history`, a `version` értéke `1`. Az olvasó elutasítja a nagyobb verziót.
- Az idők 1970 óta eltelt másodpercek (UTC). Az `id` csak betűt, számjegyet, `-` és `_` jelet tartalmaz (legfeljebb 80); a többit elhagyjuk.
- A `role` értéke `user`, `assistant` vagy `system`. Az üzenet választható mezői: `reasoning`, `display`, `sources` (szövegek listája), `stopped` (true), `images` (csak `data:image/…` URL-ek, és csak a „képekkel” exportnál).
- Az ismeretlen mezőket figyelmen kívül hagyjuk, így más programok hozzáadhatják a sajátjaikat.
- Az import hozzáadja az új `id`-kat, és ismert `id`-t csak akkor cserél, ha a fájl `updated` értéke újabb. Az üzenet nélküli beszélgetéseket kihagyja.
- Olvasáskor a korlátok: 128 MB fájl, 20 000 beszélgetés, beszélgetésenként 100 000 üzenet.

Egy előzményfájl az ablakra húzva importálódik.
