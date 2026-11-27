# Ficheiro do histórico de conversas

O histórico de conversas é um único ficheiro JSON simples, para que o Colibri Companion, o CartaColibri e qualquer script o possam trocar. Na aplicação: a barra lateral, «Histórico: exportar / importar».

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

## Regras para quem escreve e quem lê

- `format` tem de ser `colibri-chat-history`; `version` é `1`. Um leitor recusa uma versão maior.
- Os tempos são segundos desde 1970 (UTC). `id` usa só letras, algarismos, `-` e `_` (no máximo 80); o resto é retirado.
- `role` é `user`, `assistant` ou `system`. Campos opcionais da mensagem: `reasoning`, `display`, `sources` (lista de textos), `stopped` (true), `images` (só URLs `data:image/…`, e só na exportação «com imagens»).
- Os campos desconhecidos são ignorados, por isso outros programas podem juntar os seus.
- A importação junta os `id` novos e só substitui um `id` conhecido se `updated` do ficheiro for mais recente. As conversas sem mensagens são saltadas.
- Limites na leitura: ficheiro 128 MB, 20.000 conversas, 100.000 mensagens por conversa.

Arrastar um ficheiro de histórico para a janela importa-o.
