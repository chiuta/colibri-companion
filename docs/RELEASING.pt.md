# Publicar uma versão

Como se faz uma versão (para o mantenedor)

1. **Versão.** Muda `version` no `Cargo.toml` da raiz e `TAG` em `index.html` para o novo número (por exemplo `0.1.0-alpha.12` e `v0.1.0-alpha.12`), corre `cargo update -w --offline`, acrescenta `docs/releases/<tag>.md`, e faz commit e push para `main`.
2. **Compilação.** Espera que as execuções `build` e `engine-ci` desse commit estejam verdes. Depois inicia à mão o workflow `build` (Actions, Run workflow) e escreve a etiqueta, por exemplo `v0.1.0-alpha.12`. Compila todas as plataformas e cria uma versão **rascunho**.
3. **Publicação.** Abre o rascunho, cola `docs/releases/<tag>.md` na descrição, assinala pré-versão e carrega em Publish. **Não mudes o nome da etiqueta no rascunho**: os ficheiros foram compilados com a versão do `Cargo.toml`, e outra etiqueta faz com que o programa indique uma versão e a publicação outra (e a verificação de atualizações oferece uma sem fim). O workflow recusa uma etiqueta que já aponta para outro commit.

## Se um número foi publicado por engano

Não o reutilizes. Apaga à mão essa versão e a sua etiqueta e publica o número seguinte.
