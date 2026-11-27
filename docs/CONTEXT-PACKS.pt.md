# Pacotes de contexto

Reaproveita o trabalho feito sobre um texto quando ele reaparece noutro prompt (só motor corbul; clean-room, código original).

Uma cache de prefixo só acerta se o texto comum vier primeiro. Um *pacote* é a cache KV de um texto calculado sozinho e guardado numa prateleira (armazém de instantâneos assinado, com limite LRU).

`compose` coloca os pacotes um a seguir ao outro: as linhas são copiadas e as chaves rodadas pela diferença de posição (exato); depois o início de cada pacote seguinte é recalculado em janelas crescentes até a alteração medida cair abaixo de uma tolerância. A pergunta é sempre calculada de novo.

`tolerance = 0` recalcula tudo e iguala um prefill completo. Modelos com janela deslizante, cache expulsa ou em acelerador são recusados e voltam ao prefill normal.

Não exato até ser reparado: um estado com linhas movidas indica `is_exact() == false` e nunca é escrito em disco.

Limites: o motor colibri de JustVugg não exporta caches, por isso só funciona com o corbul. Testado em modelos sintéticos pequenos; a qualidade em modelos reais ainda não foi medida.

## Linha de comandos

`corbul packs MODEL --session DIR --pack FICHEIRO… --question TEXTO [--tolerance 0.02] [--probe 8] [--max-new 64] [--check]` guarda cada ficheiro uma só vez e depois responde com os pacotes guardados, em qualquer ordem. `--check` também executa um prefill completo e mostra a diferença de pontuação e o tempo de cada um.
