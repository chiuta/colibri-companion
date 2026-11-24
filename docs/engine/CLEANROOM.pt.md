# Processo clean-room

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

Uma nota breve e honesta sobre como o motor em `crates/engine` foi escrito e o que «clean-room» significa e não significa aqui.

## As regras que seguimos

1. **Fontes.** Apenas especificações e artigos públicos: as estruturas dos ficheiros GGUF e safetensors, os campos `config.json` e `tokenizer.json` do Hugging Face, as descrições publicadas das estruturas de blocos de quantização e os artigos por trás de cada técnica (variantes de atenção, encaminhamento MoE, Mamba, RWKV, descodificação especulativa, H2O, StreamingLLM, SnapKV, TIES, DARE, marca de água de lista verde, HNSW, BM25…).
2. **Sem espreitar.** O código-fonte do colibri, do llama.cpp e de outros motores de inferência não foi lido nem copiado ao escrever o motor. Quando uma estrutura não era clara, o trabalho parou na descrição pública, e a incerteza está escrita no código e na documentação.
3. **Oráculos em vez de imitação.** Cada núcleo rápido é testado face a uma versão escalar simples escrita primeiro; cada quantizador face a blocos calculados à mão e ida e volta; os caminhos repartidos (malha, transmissão, caches) face ao caminho numa só máquina, bit a bit onde isso é prometido.
4. **Apenas licenças permissivas.** O motor depende de `serde_json` e `memmap2` e de mais nada; o crate de GPU acrescenta `wgpu` atrás de uma opção. As licenças foram verificadas com `cargo-deny` num commit anterior; as alterações posteriores ainda não foram reverificadas.
5. **A entrada hostil é normal.** Cada tamanho ou deslocamento lido de um ficheiro é verificado face ao comprimento real do ficheiro antes de qualquer alocação; testes do tipo fuzz fornecem ficheiros truncados e corrompidos e têm de receber um erro, não uma falha.

## O que não afirmamos

- Clean-room é um processo, não uma garantia legal. O modelo que escreveu o código, como qualquer modelo, foi treinado em texto público que pode incluir código aberto, e não podemos excluir que tenha visto os projetos acima. Não os consultámos e procurámos tornar isso visível mantendo no código os oráculos, os testes e a fonte de cada fórmula.
- Várias fórmulas (Mamba, RWKV-6/7, MLA, extras de imatrix e GGUF) foram escritas a partir de descrições publicadas e da memória delas, e verificadas apenas quanto à coerência interna, por não haver nenhum ponto de controlo real. Podem diferir da referência em detalhes; trata-as como não verificadas até alguém as correr em modelos reais.
- A compatibilidade com ficheiros de outros motores é um objetivo, não um facto verificado (vê a tabela em [ENGINE.pt.md](ENGINE.pt.md)).

Se encontrares um sítio onde o motor reproduz de demasiado perto o código de outra pessoa, abre uma issue; reescrevemo-lo.
