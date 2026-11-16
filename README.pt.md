# Colibri Companion

[English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · **Português** · [Deutsch](README.de.md) · [Magyar](README.hu.md)

**Site: https://colibri-companion.chiuta.ro** — transferência, tutoriais em vídeo, ajuda.

Uma aplicação de secretária simples para o [colibri](https://github.com/JustVugg/colibri): escolhe um modelo adequado ao teu computador, transfere-o com um clique e conversa — o modelo corre no teu computador, sem conta e sem serviços na nuvem.

## Execução

```bash
cargo run --release -p colibri-companion            # a aplicação real
cargo run --release -p colibri-companion -- --demo  # demonstração, sem colibri
# opções: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### Os quatro passos

(1) encontra o Python e o colibri (e propõe transferir o colibri se faltar), (2) permite-te escolher um modelo (os que não cabem na RAM ficam ocultos; o recomendado está assinalado), (3) transfere e instala com progresso, pausa e retoma, (4) inicia o servidor e abre a conversa.

Conversa: respostas em fluxo, Parar instantâneo, Markdown, raciocínio recolhível, conversas guardadas, estilo (Preciso / Equilibrado / Criativo), copiar e regenerar. Temas: claro, escuro, contraste elevado.

## Como funciona

A aplicação não inclui o colibri. Executa os comandos JSON do colibri (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) e comunica com o seu gateway compatível com OpenAI em `127.0.0.1`. Nunca instala pacotes do sistema: se faltar algo, mostra-te o comando para copiar. Os teus dados (definições, conversas) ficam locais, na pasta de dados do utilizador; `COLIBRI_COMPANION_HOME` permite alterá-la.

## Licenças

- Este projeto: Apache-2.0 (`LICENSE`, `NOTICE`), como o colibri.
- Dependências: apenas licenças permissivas, verificadas com `cargo deny check licenses` (`deny.toml`); lista completa em `THIRD_PARTY_LICENSES.md`. Tipos de letra: Inter e JetBrains Mono (SIL OFL 1.1, em `licenses/`).
- Os modelos têm licenças próprias; a aplicação mostra-as e pede confirmação antes da transferência.

## Plataformas

Linux (x86-64, ARM64; Wayland e X11), Windows 10/11 e macOS (Intel e Apple Silicon) — um só código-fonte, renderização OpenGL (funciona em placas gráficas antigas e em máquinas virtuais). `.github/workflows/build.yml` compila e testa todas as variantes e produz ficheiros comprimidos prontos a distribuir. No Linux, `packaging/linux/install.sh` acrescenta a aplicação ao menu. Escala HiDPI, atalhos Ctrl/⌘+N, Ctrl/⌘+, e Esc, e suporte para leitores de ecrã (AccessKit). As compilações e os testes automáticos cobrem os três sistemas; os testes manuais em computadores reais com Windows e macOS ainda são limitados, e as etiquetas para leitores de ecrã (AccessKit) ainda não foram experimentadas com leitores reais.

**Documentos e ferramentas:** adiciona ficheiros (**Mais → Documentos**) e faz perguntas: a aplicação pesquisa-os no teu computador (palavras-chave, sem nuvem) e mostra sob cada resposta os excertos usados. **Mais → Ferramentas MCP** liga servidores MCP que tu adicionas e inicias; o modelo pode pedir uma ferramenta e cada chamada precisa da tua aprovação (funcionalidade inicial, depende de o modelo seguir o formato).

Modelos: mantém vários no computador, alterna entre eles ou adiciona mais a partir de uma só janela (**Modelos**); o mais capaz que o teu computador consegue executar é sugerido primeiro (com pouca RAM, os especialistas são lidos do SSD). Velocidade: CUDA para NVIDIA, Vulkan para outros chips gráficos, senão o processador, escolhidos automaticamente. **Recursos** mostra em direto processador, memória, placa gráfica e disco, e a prova de que nada sai do teu computador (todas as ligações abertas da aplicação e do motor, com o comando para verificar por ti mesmo). Se o colibri faltar, é descarregado por ti (sem git), e uma verificação diária (pode ser desligada) avisa quando o colibri ou esta aplicação têm uma versão nova.

**O teu computador, analisado:** em **Recursos**, a aplicação encontra sozinha as placas gráficas e diz de forma simples se o modelo cabe na memória e porque pode estar lento (swap, outros programas, placa sem uso). Só aconselha; não altera nenhuma definição sozinha.

## Idiomas

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — escolhidos nas Definições, com `--lang en|ro|fr|it|es|pt|de|hu`, ou pelo idioma do sistema no primeiro arranque. As traduções estão em `crates/app/src/i18n.rs` (a chave é o texto em inglês); um teste verifica que cada texto da interface tem todas as traduções. Foram feitas com ajuda de uma IA e não foram revistas por falantes nativos.

## Desenvolvimento

```bash
cargo test --workspace   # inclui um teste de ponta a ponta com um `coli` falso (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Novo: pesquisa nas conversas (Ctrl/⌘+K), mudar o nome (clique direito), editar a última mensagem, ficheiros de texto arrastados para a mensagem e guardar como .md.
