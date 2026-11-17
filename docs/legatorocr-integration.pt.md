# Integração do LegatorOCR

Estado: biblioteca em `crates/core/src/legatorocr.rs`, ligada à janela Documentos por `crates/app/src/ocr_ui.rs` (secção «Ficheiros digitalizados (LegatorOCR)»). Testada com um binário real `legatorocr 0.2.0-pre.65` numa única página gerada de texto sintético limpo; nunca com a digitalização real de um livro.

## O que faz

O LegatorOCR (github.com/chiuta/LegatorOCR, **MIT**) transforma digitalizações em PDF ou texto pesquisável. O Colibri Companion (Apache-2.0) nunca liga nem copia o seu código; só executa o programa como comando externo, o que ambas as licenças permitem.

1. `locate` / `locate_in`: procura um executável `legatorocr` (uma pasta indicada pelo utilizador, a pasta da aplicação, depois `PATH`). Só o sistema de ficheiros, sem rede.
2. `version`: executa `legatorocr --version`; a resposta tem de começar por `legatorocr`, para que um ficheiro alheio com o mesmo nome não seja tomado pelo programa.
3. `collect_inputs`: uma imagem ou um PDF, ou as imagens e PDF diretamente numa pasta, por ordem de nome (no máximo 2000).
4. `read_scan`: executa `legatorocr <páginas> --ocr --lang <código> --txt <ficheiro> --format none --out-dir <pasta>` (mais `--clean-text` para romeno) através de `proc::run_capture` com limite de tempo, e depois lê o texto de volta (no máximo 32 MB).
5. `ingest_scan`: junta esse texto a uma coleção de documentos (`rag::Library::add_document`) com o nome `<ficheiro ou pasta> (OCR)`. O modelo local responde então a partir da digitalização e as passagens são citadas com esse nome como qualquer outro documento.

Segurança: os argumentos são passados um a um (sem shell); as entradas tornam-se absolutas para que um ficheiro chamado `-x.png` não seja lido como opção; o código de língua é verificado (`clean_lang`); a saída tem limite; uma execução que não termina é morta; as falhas voltam com as últimas linhas da saída de erro do programa; nada arranca se não for chamado.

## Limites, ditos com franqueza

- O texto é OCR. Não é uma transcrição verificada, e a qualidade do OCR (e que línguas funcionam) depende do LegatorOCR e dos pacotes de língua do Tesseract instalados: `legatorocr --doctor` lista-os (na máquina de teste faltavam os modelos de romeno e húngaro).
- Os números de página perdem-se: toda a digitalização é um só documento, por isso uma citação nomeia a digitalização, não a página.
- A pesquisa na biblioteca é por palavras-chave (BM25), como em qualquer documento.
- A janela só mostra a secção quando o programa é encontrado (procurado uma vez, em segundo plano, ao abrir a janela; nada é descarregado nem iniciado sozinho). Sem ele aparecem apenas uma dica com `github.com/chiuta/LegatorOCR` e um botão «Procurar de novo».
- A execução é um fio em segundo plano com um botão Parar; o LegatorOCR não dá nenhum valor de progresso, por isso a janela mostra os segundos decorridos, não uma percentagem. Fechar a aplicação inteira durante uma execução não é tratado de forma especial.
- A lista de línguas são as 8 línguas da aplicação mapeadas para códigos do Tesseract; se uma funciona depende dos dados instalados para o LegatorOCR (`legatorocr --doctor`). A aplicação não o verifica antes: uma língua em falta volta como a mensagem de erro do próprio programa.
- As imagens de página largadas na janela (não os PDF) vão para o campo de digitalização quando o programa é encontrado.

## Experimenta

```
cargo test -p companion-core legatorocr
LEGATOROCR_BIN=/path/to/legatorocr LEGATOROCR_TEST_PAGE=/path/to/page.png \
  cargo test -p companion-core real_legatorocr -- --nocapture
```
