# Reduzir a cache e reajustar os valores

`corbul kvfit MODEL --text FILE --question Q` mede o que custa manter só uma fração das linhas da cache de um texto (escolhidas pelo SnapKV), com e sem **reajuste dos valores** que ficam.

Os últimos `--window` tokens do texto são registados como consultas de sonda. Os valores das linhas mantidas são depois movidos o mínimo para que a atenção sobre elas dê, nas sondas, o que dava a atenção sobre todas as linhas (mínimos quadrados regularizados, forma dual, `crates/engine/src/kvfit.rs`). O ajuste nunca pode ser pior do que os valores simples *nas sondas*; se ajuda numa pergunta não sondada depende do modelo e do texto, e só este comando, executado no teu modelo, to diz.

Opções: `--budget 0.25` (fração de linhas mantidas), `--window 16`, `--ridge 0.01`, `--max-new 32`. Mostra o erro do ajuste nas sondas, a maior diferença das primeiras pontuações face à cache completa, se o primeiro token coincide e quantos dos primeiros tokens da resposta coincidem.

Limites: só valores (as chaves levam a posição e ficam; o peso das linhas removidas não é modelado), as camadas de janela deslizante ficam como estão, o ajuste custa tempo proporcional a linhas × sondas por cabeça. Ainda não ligado aos pacotes de contexto (um pacote reduzido não se pode emendar). Nada medido num modelo real.

Os pacotes de contexto também podem ficar guardados a metade do tamanho: `corbul packs --pack-kv f16` e `corbul serve --packs-kv f16` (arredondamento F16 em cada linha), e comprimidos sem perdas: `--pack-compress`, `--packs-compress`.

No Colibri Companion: *Documentos → Pacotes de contexto (corbul)* tem as opções F16 e compressão (aplicam-se na próxima vez que o corbul arrancar) e *Medir a redução da cache…*, que executa este comando no modelo que o corbul está a executar (carrega o modelo uma segunda vez).
