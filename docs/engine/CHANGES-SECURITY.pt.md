# Alterações de segurança e robustez

[English](CHANGES-SECURITY.md) · [Română](CHANGES-SECURITY.ro.md) · [Français](CHANGES-SECURITY.fr.md) · [Italiano](CHANGES-SECURITY.it.md) · [Español](CHANGES-SECURITY.es.md) · [Português](CHANGES-SECURITY.pt.md) · [Deutsch](CHANGES-SECURITY.de.md) · [Magyar](CHANGES-SECURITY.hu.md)

Outros documentos: [ENGINE.pt.md](ENGINE.pt.md) · [QUICKSTART.pt.md](QUICKSTART.pt.md)

Esta página enumera o que a revisão adversarial do motor e do `corbul` encontrou, e o que foi feito a esse respeito, para que possas avaliar até que ponto confiar numa funcionalidade. É uma lista de correções e limites, não uma certificação de segurança: nada disto foi auditado por terceiros.

## Servidor

- **Pedidos entre sites.** As chamadas que alteram estado verificam `Origin` e exigem um `Content-Type` JSON, para que uma página web que visites não possa fazer o teu navegador enviar pedidos a um `corbul serve` local. Os navegadores só são admitidos a partir das origens que indicares com `--cors`.
- **Rotas de administração.** `/api/create`, `/api/delete`, `/models/load` e `/models/unload` precisam do token mestre. Os titulares de chaves de API só as podem usar se iniciares o servidor com `--allow-admin`.
- **Segredos fora da linha de comandos.** `--token-file`, `--key-file`, `--receipt-key-file` e as variáveis `CORBUL_TOKEN` e `CORBUL_MESH_KEY` mantêm os segredos fora da lista de processos. Os simples `--token`, `--key` e `--receipt-key` continuam a funcionar e são visíveis para os outros utilizadores da máquina.
- **Limites de recursos.** `--max-kv` recusa pedidos cuja cache KV seria demasiado grande (por defeito 8g); `--max-heavy` limita os pedidos simultâneos de tokenização, modelo e verificação (por defeito 4); os limites de taxa de `--keys-file` cobrem todas as rotas que custam computação; o descodificador de corpos fragmentados é linear; `POST /v1/quotes/verify` tem tetos no tamanho do texto e das fontes.
- **Registos.** O texto dos prompts não é registado a menos que se indique `--log-prompts`; `--no-log` desativa o registo.
- **Interface web.** A página em si é pública (não contém segredos); cada chamada que faz precisa do token.
- **Limite que permanece:** o servidor fala HTTP sem cifra. Associa-o a `127.0.0.1` (o valor por defeito) ou coloca TLS à frente.

## Malha

- Handshake HMAC-SHA256 mútuo com uma chave partilhada de pelo menos 8 caracteres, um MAC em cada trama, tetos nos tamanhos de handshake e de trama.
- A descoberta UDP é opcional (`--discovery`, `--discovery-port`) e as respostas não são maiores do que a sonda, pelo que um worker não pode ser usado como amplificador.
- **Limite que permanece:** o tráfego é autenticado mas não cifrado; a chave é partilhada por todos na malha; testado apenas em loopback.

## Ficheiros, descarregamentos e fusões

- A análise de GGUF recebe o comprimento real do ficheiro (`parse_with_len`); a construção do tokenizador usa uma trie com um teto no tamanho das peças; os ficheiros truncados e corrompidos devolvem erros nos testes adversariais.
- Descarregamentos: fixação ao SHA do commit, limites de tamanho, recusa de redirecionamentos para http sem cifra, `wget` anónimo como alternativa (o token nunca lhe é passado), verificações SHA-256.
- As fusões não seguem ligações simbólicas e recusam-se a substituir uma saída a menos que se indique `--overwrite`; as formas e os intervalos são verificados antes de escrever.
- Os descodificadores de imagem aplicam `Limits` antes de reservar memória e limitam a expansão de deflate; o leitor de WAV tem limites contra ficheiros hostis.

## Descodificação e verificação

- A descodificação por gramática, expressão regular e esquema JSON tem orçamentos de trabalho; o bloqueio de citações é linear com tetos e informa `limit_hit` quando um teto o interrompe.
- Os recibos têm tetos de tamanho; as instantâneas podem ser seladas com um HMAC com chave; os recibos e as instantâneas usam cadeias SHA-256.
- Correções de exatidão encontradas na revisão: probabilidades obsoletas após top-k/top-p (agora renormalizadas), o alias `typ_p`, vírgulas decimais no tratamento de números, limites de lookahead além do contexto, a pontuação de grupo do DeepSeek-V2, a exatidão em lote quantizada (`exact_batch`).

## Não feito ou não verificado

- Sem auditoria de terceiros, sem campanha de fuzzing além dos testes adversariais do repositório, sem teste de intrusão do `corbul serve`.
- `ToolStream::push` pode voltar a percorrer o seu buffer repetidamente (quadrático no comprimento da saída até `--max-tokens`).
- O `GrammarCache` existe mas a CLI ainda não o usa.
- As licenças das dependências foram verificadas pela última vez com `cargo-deny` no commit b496ba5.
- Os workflows de CI nunca foram executados; os caminhos de código de ARM, macOS e Windows não foram executados aqui.
