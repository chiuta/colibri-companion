# Rularea fișierelor GGUF cu corbul

Colibri importă doar familiile de modele pe care le implementează (Qwen3.x, DeepSeek, GLM, Mistral-Large, MiMo și câteva altele). Un GGUF din altă familie, de exemplu Gemma, nu se importă. Colibri Companion încearcă atunci motorul propriu, **corbul**, care citește direct fișierul `.gguf`.

- În *Modelele mele → Încarcă modele GGUF dintr-un dosar* fiecare fișier are butonul *Rulează cu corbul* (apare când `corbul` e lângă aplicație sau în PATH; arhivele de lansare îl conțin).
- Când un import eșuează, mesajul numește fișierul, arhitectura declarată și ultimele linii din jurnal, iar corbul e încercat automat.
- corbul ascultă doar pe loopback, pe un port liber, cu raftul de pachete de context în dosarul de date al aplicației (vezi CONTEXT-PACKS). Se oprește când oprești modelul, pornești altul sau închizi aplicația. Jurnalul apare la *Arată jurnalul*.
- Limite: corbul e mai lent decât colibri pe modelele foarte mari, nu transmite experți de pe disc și suportă familiile implementate de motorul propriu; dacă nu poate încărca fișierul, mesajul spune de ce.
