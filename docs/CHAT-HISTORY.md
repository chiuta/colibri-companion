# Chat history file

The chat history is one plain JSON file, so that Colibri Companion, CartaColibri and any script can exchange it. In the app: the sidebar, “History: export / import”.

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

## Rules for writers and readers

- `format` must be `colibri-chat-history`; `version` is `1`. A reader refuses a larger version.
- Times are seconds since 1970 (UTC). `id` uses only letters, digits, `-` and `_` (up to 80); anything else is dropped.
- `role` is `user`, `assistant` or `system`. Optional message fields: `reasoning`, `display`, `sources` (list of texts), `stopped` (true), `images` (only `data:image/…` URLs, and only when exported “with pictures”).
- Unknown fields are ignored, so other programs may add their own.
- Importing adds new ids and replaces a known id only when the file’s `updated` is newer. Conversations without messages are skipped.
- Limits when reading: file 512 MB, 100,000 conversations, 100,000 messages per conversation.

Dropping a history file onto the window imports it.
