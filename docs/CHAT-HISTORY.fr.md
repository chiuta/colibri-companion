# Fichier d'historique des conversations

L'historique des conversations est un simple fichier JSON, pour que Colibri Companion, CartaColibri et n'importe quel script puissent l'échanger. Dans l'application : la barre latérale, « Historique : export / import ».

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

## Règles pour qui écrit et qui lit

- `format` doit valoir `colibri-chat-history` ; `version` vaut `1`. Un lecteur refuse une version supérieure.
- Les dates sont en secondes depuis 1970 (UTC). `id` n'utilise que lettres, chiffres, `-` et `_` (80 au plus) ; le reste est retiré.
- `role` vaut `user`, `assistant` ou `system`. Champs facultatifs d'un message : `reasoning`, `display`, `sources` (liste de textes), `stopped` (true), `images` (seulement des URL `data:image/…`, et seulement pour l'export « avec images »).
- Les champs inconnus sont ignorés ; d'autres programmes peuvent donc ajouter les leurs.
- L'import ajoute les nouveaux `id` et ne remplace un `id` connu que si `updated` du fichier est plus récent. Les conversations sans message sont ignorées.
- Limites à la lecture : fichier 128 Mo, 20 000 conversations, 100 000 messages par conversation.

Déposer un fichier d'historique sur la fenêtre l'importe.
