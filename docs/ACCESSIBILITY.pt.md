# Acessibilidade

O Colibri Companion quer poder ser usado pelo maior número possível de pessoas. Tudo o que se segue está em **Acessibilidade**, na barra de cima (ou com `Ctrl+Shift+A`).

## O que existe

| Necessidade | O que a aplicação faz |
| --- | --- |
| Baixa visão | Tamanho do texto de 80 % a 300 %; um aspeto de alto contraste; o tipo de letra Atkinson Hyperlegible (SIL OFL); mais espaço entre linhas e botões; um contorno grosso à volta do elemento onde está o teclado |
| Cegueira | A interface é exposta aos leitores de ecrã através do AccessKit; cada resposta pode ser lida em voz alta com a voz do próprio computador (botão, `Ctrl+Shift+S` ou automaticamente) |
| Daltonismo | Uma paleta azul/laranja em vez de verde/vermelho nos três aspetos |
| Mãos e movimento | Botões até 64 pontos de altura; toda a aplicação funciona só com o teclado; nunca é preciso arrastar nem fazer movimentos finos |
| Audição | A aplicação nunca faz som e nunca te pede para ouvir; a barra de tarefas pode piscar quando uma resposta está pronta Esc fecha a janela aberta, mesmo a partir de uma caixa de texto, e nenhuma janela passa do ecrã. |
| Dislexia, atenção, carga cognitiva | Tipo de letra legível, linhas mais espaçadas, sem animações, menus simples e um estilo de resposta em *linguagem simples* (frases curtas, palavras do dia a dia) |

Sete predefinições (baixa visão, cegueira, movimentos difíceis de controlar, dislexia, leitura mais fácil, daltonismo, audição) ajustam conjuntos sensatos com um clique; depois cada opção pode ser alterada em separado.

## Como é verificada

* Cada par de cores de texto e fundo usado pela interface, nos três aspetos e com a paleta para daltónicos ligada ou desligada, tem de atingir o contraste 4,5 : 1 (WCAG AA para texto normal). É um teste unitário (`theme::tests::every_look_is_readable`), por isso uma mudança de cor que o quebre faz falhar a compilação.
* As cores «bom/mau» da paleta para daltónicos também têm de diferir em luminosidade, para continuarem distinguíveis em tons de cinzento.
* O texto enviado ao programa de voz segue pela entrada padrão ou numa variável de ambiente, nunca na linha de comandos, por isso uma resposta não pode ser confundida com uma opção (testado de forma unitária).

## O que **não** foi verificado

* A aplicação não foi experimentada com um leitor de ecrã real (NVDA, JAWS, Narrator, VoiceOver, Orca). A árvore de acessibilidade vem do egui/AccessKit; os botões só com ícone podem ser anunciados mal. Os relatos são bem-vindos.
* A leitura em voz alta usa `espeak-ng`/`espeak`/`spd-say` no Linux, `say` no macOS e System.Speech através do PowerShell no Windows. Só a construção do comando é testada; a voz em si não foi ouvida aqui. No macOS e no Windows a língua da voz é a escolhida no sistema.
* A *entrada* de voz fica a cargo do ditado do sistema operativo (Windows `Win+H`, macOS `Fn` duas vezes). O ditado offline dentro da aplicação (whisper.cpp) está previsto em `ROADMAP-INTERCONNECT.md`.
* A varredura por interruptor e o seguimento do olhar não estão implementados; as ferramentas do sistema operativo funcionam por cima do suporte de teclado.
