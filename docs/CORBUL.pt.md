# Executar ficheiros GGUF com o corbul

O colibri só importa as famílias de modelos que implementa (Qwen3.x, DeepSeek, GLM, Mistral-Large, MiMo e mais algumas). Um GGUF de outra família, por exemplo Gemma, não se importa. O Colibri Companion tenta então o motor próprio, o **corbul**, que lê o `.gguf` diretamente.

- Em *Os meus modelos → Carregar modelos GGUF de uma pasta* cada ficheiro tem o botão *Executar com o corbul* (aparece quando o `corbul` está junto à aplicação ou no PATH; os arquivos de lançamento incluem-no).
- Quando uma importação falha, a mensagem nomeia o ficheiro, a arquitetura declarada e as últimas linhas do registo, e o corbul é tentado automaticamente.
- O corbul escuta só no loopback, numa porta livre, com a prateleira de pacotes de contexto na pasta de dados da aplicação (ver CONTEXT-PACKS). Pára quando paras o modelo, inicias outro ou fechas a aplicação. O registo vê-se em *Mostrar o registo*.
- Limites: o corbul é mais lento do que o colibri em modelos muito grandes, sem streaming de especialistas a partir do disco, e suporta as famílias do seu próprio motor; se não conseguir carregar o ficheiro, a mensagem diz porquê.
