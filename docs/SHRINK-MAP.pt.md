# Pacotes reduzidos e mapas de cache entre modelos

Duas extensões dos pacotes de contexto ([CONTEXT-PACKS.pt.md](CONTEXT-PACKS.pt.md), só motor corbul; clean-room, código original).

## Pacotes reduzidos

Quando os textos e a pergunta juntos não cabem na janela do modelo, os pacotes são recusados. Com a redução, são encurtados: `corbul packs --shrink auto` (só o necessário) ou `--shrink R` (mantém sempre cerca da fração `R`, de 0,1 a 1 das linhas de cada pacote). O mesmo faz o campo `"shrink": "auto"` (ou um número) de `/v1/packs/complete`; a resposta lista o que foi reduzido (`"shrunk"`).

Como: os últimos `--shrink-window` tokens de um pacote (16 por omissão) são relidos e votam nas linhas a manter (SnapKV); as primeiras linhas mantêm-se sempre (é onde a atenção se acumula); os valores que ficam são reajustados ([KV-FIT.pt.md](KV-FIT.pt.md); `--no-refit` desliga isso). As chaves mantidas são depois rodadas para as posições dos novos lugares, e o pacote passa a ser *exatamente a cache do seu texto encurtado*: emenda-se e repara-se como qualquer pacote e, com reparação completa, a resposta é igual a um simples pré-preenchimento do texto encurtado. Os primeiros `--shrink-skip` textos (1 por omissão, normalmente as instruções) ficam inteiros. A redução faz-se ao juntar os pacotes; a prateleira guarda os pacotes inteiros.

No Colibri Companion: *Documentos → Pacotes de contexto (corbul)* → *Reduzir os textos demasiado longos*.

Limites: perde-se informação, quanta depende do modelo e do texto (meça com `corbul kvfit` ou a janela *Measure cache shrinking…*); modelos com camadas de janela deslizante, caches Q8 e caches de acelerador são recusados; se os textos não couberem nem retirando todos os textos redutíveis, o pedido falha. Até agora testado só em modelos minúsculos aleatórios.

## Mapas de cache entre modelos diferentes

Um pacote escrito por um modelo não serve a outro. Se os dois partilham o tokenizador, as suas linhas de cache para os *mesmos tokens* podem ser relacionadas: `corbul map-fit SRC DST --text FICHEIRO --out MAPA` lê o texto com os dois modelos e ajusta, por camada, um mapa linear (mínimos quadrados com ridge, com a média) das linhas de chaves e valores de SRC para as de DST. As chaves comparam-se sem a posição (devolvidas a 0 e rodadas de novo na importação), por isso as tabelas rotativas e os tamanhos de cabeça podem diferir; as camadas emparelham-se por profundidade relativa. Mostra o erro nas linhas deixadas fora do ajuste ao lado do de adivinhar a média e avisa se o mapa não for melhor.

`corbul map-check SRC DST --map MAPA --text FICHEIRO --question Q` divide o texto em dois pacotes escritos por SRC, leva-os para DST pelo mapa e responde de três maneiras: o pré-preenchimento completo de DST (a verdade), a cache mapeada com a reparação habitual (`--tolerance`, 0,02 por omissão) e a cache mapeada sem reparação. Mostra as linhas recalculadas e a distância de cada resposta à verdade.

Uma linha mapeada é só um palpite: as linhas importadas ficam marcadas como aproximadas e a reparação recalcula as que forem precisas (todas com tolerância 0, o que iguala um pré-preenchimento simples). Um mau mapa custa, portanto, tempo, nunca correção. Ponto de entrada da biblioteca: `ctxpack::compose_mapped`.

Limites: os tokenizadores têm de dar os mesmos identificadores (verifica-se; senão nada é ajustado); o mapa vale para um só par de modelos (as impressões digitais verificam-se); o tamanho é `kd × kd' × camadas × 8` bytes; o ajuste guarda em memória as linhas de calibração (`--rows`, 1024 por omissão). Um mapa linear entre caches de modelos sem relação pode ser fraco; se compensa para o seu par só se sabe medindo. Ainda não está no servidor nem no Companion e nada foi medido num par real de modelos.
