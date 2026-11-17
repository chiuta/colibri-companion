# Roteiro da interligação: de uma janela de chat a um assistente capaz

Objetivo: ligar o modelo local a tudo o que multiplica o que ele pode fazer, enquanto a pessoa continua ao comando. Cada passo mantém três regras: **primeiro o local, nada sai sem um sim, qualquer ação pode ser vista e desfeita.**

## 0. O quadro de segurança (construído primeiro, usado por tudo o resto)
- **Níveis de capacidade** (ver `platforms.rs`): Ler → Pesquisar → Escrever → Notificar → Agir. Uma ligação é criada a um nível e nunca o pode ultrapassar.
- **Cartão de aprovação** para cada escrita, envio e ação: mostra exatamente o que vai acontecer, com *Permitir uma vez / Permitir para esta automatização / Recusar*. Ler não precisa de cartão.
- **Conteúdo não fiável** (páginas web, problemas, e-mails, documentos) é isolado e etiquetado como dados. Uma chamada a uma ferramenta que se segue a conteúdo não fiável pergunta sempre de novo. É a principal defesa contra a injeção de prompts.
- **Orçamentos e travões**: passos, chamadas ao modelo, minutos e bytes por execução; uma tecla *Parar tudo*.
- **Registo de auditoria** de cada pedido que saiu do computador e de cada ação feita, num ficheiro simples.
- **Segredos**: os tokens ficam em memória por omissão; guardá-los é uma escolha explícita.

## 1. Ferramentas para o modelo (o multiplicador)
O colibri expõe a chamada a ferramentas na maioria dos motores. O programa oferece ao modelo uma pequena caixa honesta: página web, pesquisa, ficheiro de repositório, pesquisa em pasta, calculadora, Python num ambiente isolado, consulta de tabelas, pesquisa em notas, guardar ficheiro, webhook. O modelo pede, o programa verifica nível e aprovação, executa e devolve o resultado como dados.

## 2. MCP: uma só ficha para milhares de ferramentas
Falar o Model Context Protocol como cliente (e manter `coli mcp` como servidor). Qualquer servidor MCP (ficheiros, bases de dados, navegadores, calendários, ferramentas de design, dispositivos da casa) fica disponível sob as mesmas regras de aprovação. É o maior salto de alcance com o menor código.

## 3. Memória e conhecimento que crescem contigo
- Índice local (BM25 agora, embeddings mais tarde) sobre pastas, notas, conversas guardadas, código, arquivos de correio.
- As respostas citam a passagem que usaram; «mostra-me onde» abre-a.
- Bibliotecas offline: Wikipédia através do Kiwix, instantâneos do arXiv e do Wikidata, manuais, leis.
- A *constelação*: memórias como estrelas ligadas (etiquetas, pessoas, projetos) que podes ver e editar.

## 4. Sentidos e voz (tudo local)
- Audição: whisper.cpp para ditado e transcrições de reuniões.
- Fala: Piper ou a voz do sistema, na língua da interface.
- Visão: modelos que aceitam imagens (o colibri indica quais), mais OCR (Tesseract) para digitalizações e fotografias de papel, mais a câmara ou uma captura de ecrã a pedido.
- Criação: Qwen-Image através do colibri, ComfyUI ou Stable Diffusion para imagens; ferramentas de áudio e vídeo à medida que surgirem.

## 5. Várias mentes
- **Duelo**: A/B às cegas com Elo, para descobrir qual o melhor modelo *para ti*.
- **Conselho**: vários modelos respondem, um modelo juiz junta-os e assinala os desacordos.
- **Encaminhador**: um modelo pequeno e rápido para as perguntas fáceis, o grande para as difíceis, escolhido por um modelo de decisão (o colibri traz Laya, GLiNER2.5-Decide, Clef).
- **Especialistas**: código, tradução, visão, cada um escolhido automaticamente, cada um substituível.

## 6. Exatidão: o modelo verifica-se a si próprio
- Executar num ambiente isolado o Python que escreve e mostrar a saída real.
- Testes unitários do código que produz, executados antes de o oferecer.
- Autoconsistência (perguntar duas vezes, comparar), citações obrigatórias para factos, um interruptor «mostra as tuas fontes» e um pequeno banco de avaliação com intervalos de confiança (como faz o LocalMind).

## 7. Agentes e automatizações
Cadeias guardadas de passos (ler, perguntar, verificar, guardar, notificar) com gatilhos: à mão, por horário, quando uma pasta muda, quando chega um webhook. Modelos: resumo da manhã, rascunho de triagem da caixa de entrada, vigia de repositório, notas de reunião, nota de investigação, revisão semanal. Cada passo mostra o seu nível.

## 8. Dispositivos e mundo físico (no fim, e com o maior cuidado)
Home Assistant, MQTT, dispositivos série, impressoras 3D, câmaras. Só leitura por omissão; ações através de uma lista de permitidos; nunca fechaduras, alarmes ou algo crítico para a segurança.

## 9. Vários computadores, um só cérebro
Partilhar um modelo na rede doméstica (um PC potente serve, um portátil usa), sincronizar prompts, memória e definições entre máquinas (cifrado, de par para par, sem nuvem) e dividir o trabalho por várias GPU onde o motor o permita.

## 10. Tornar tudo sem esforço
Um guia no primeiro arranque que escolhe um modelo para o hardware, um *modo simples* com cinco botões, ligações de um clique a partir do catálogo, sugestões «o que posso fazer com isto?», um painel visível «o que o modelo vê agora» e tudo em oito línguas.

## Ordem de trabalho
1. Pastas e ficheiros grandes, biblioteca de prompts, estilos de resposta (feito).
2. Acesso online, GitHub/GitLab/Gitea/páginas web, o catálogo (em curso).
3. Cartões de aprovação, registo de auditoria, orçamentos (o quadro de segurança).
4. Automatizações com webhooks e notificações.
5. Duelo, conselho, encaminhador.
6. Ferramentas para o modelo, depois cliente MCP.
7. Índice e memória, voz e OCR.
8. Dispositivos, várias máquinas.

O que *não* é prometido: nada disto foi ainda testado com motores colibri reais com chamada a ferramentas, e cada serviço externo muda de API com o tempo; as entradas marcadas «previsto» no catálogo são planos, não funcionalidades.
