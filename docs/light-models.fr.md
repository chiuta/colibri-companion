# Modèles locaux légers (vérifié le 2026-11-17)

« Léger » veut seulement dire que le téléchargement tient confortablement dans 4/8/16/32 Go de RAM. Cela ne dit rien de la qualité ; aucun classement ici n'est un benchmark. Source de vérité : `crates/core/src/light_models.rs`. Toutes les entrées tournent via n'importe quel serveur compatible OpenAI (Ollama, llama.cpp, LM Studio) réglé dans `server_url` ; aucune n'est dans le catalogue colibri.

Paliers de RAM : minimum = taille + 2 Go, confortable = 1,5 × taille + 2 Go, arrondis au palier 4/8/16/32 supérieur.

| id | taille Go (bas–haut imprimés) | contexte | licence | RAM min / confortable |
|---|---|---|---|---|
| lfm3-1.7b | ~1.0 (4 bits) | 256K | LFM Open License v1.0 (**non standard**) | 4 / 4 |
| qwen3.5-0.8b | 1.2-1.3 | 256K | apache-2.0 | 4 / 4 |
| lfm2.5-2.6b | 1.67 (Q4_K_M) | 131,072 | lfm1.0 (**non standard**) | 4 / 8 |
| qwen3.5-4b | 3.3-4.0 | 256K | apache-2.0 | 8 / 8 |
| qwen3.5-9b | 6.6-7.6 | 256K | apache-2.0 | 16 / 16 |
| olmoe-1b-7b (colibri, ancien 2024) | 7 (int8 converti) | n/s | Apache-2.0 | 8 (README) / 16 |
| gemma4-e2b | 4.6-7.5 | 128K | Apache 2.0 | 16 / 16 |
| gemma4-12b | 7.7-8.0 | 256K | Apache 2.0 | 16 / 16 |
| gemma4-e4b | 6.6-9.5 | 128K | Apache 2.0 | 16 / 16 |
| gpt-oss-20b | 14 (MXFP4) | n/s | Apache 2.0 | 16 / 32 |
| gemma4-26b-a4b | 16-19 | 256K | Apache 2.0 | 32 / 32 |

(n/s = non spécifié.) Les pages Ollama impriment une fourchette de taille par étiquette et ne nomment jamais la quantification par défaut ; les paliers utilisent donc le haut de la fourchette. Seuls LFM2.5 (Q4_K_M), gpt-oss (MXFP4) et OLMoE (README) ont une taille liée à un format nommé. Les 7,5 Go de Gemma 4 E2B égalent l'étiquette séparée `e2b-mlx` ; la page ne dit pas quelle variante est laquelle. Nemotron 3.5 Bolt E4B fonctionne uniquement sur GPU avec vLLM et n'est jamais proposé par `for_ram`.

## Non vérifié (pas dans la liste)
- Qwen3.5 2B : taille lue sur Ollama, licence HF illisible (0.8B, 4B, 9B sont Apache-2.0).
- Granite 4.2 3B/8B/30B : la doc IBM indique Apache 2.0, 128K, appel d'outils ; date du 25 août 2026 seulement par des pages secondaires ; pas de taille en Go.
- MiMo-V2.6-Distill-Qwen-9B : fiche HF d'origine illisible ; miroir seulement.
- Mellum2 12B-A2.5B : Apache 2.0, contexte 131,072, mais ni taille GGUF ni année ; aucune page ne mentionne « Mellum2.1 ».
- Nemotron 3.5 Bolt E4B : GPU avec vLLM seulement, exclu.
- Qwen3.8 : Qwen ne publie aucun modèle sous 27B ; le Qwen3.8-4B d'empero-ai est une distillation tierce.
- Flint 40B-A2.5B et LFM3-VL-2B (lignes 252–253 du README de colibri), PLaMo 3 Translate, Phi-4-mini, Granite 4.1, Bonsai 2 ; les étiquettes Gemma 4 QAT n'apparaissent pas sur la page Ollama.
- Une recherche des sorties de sept.–nov. 2026 n'a rendu que des pages de classement tierces, aucune lue comme source primaire.
- Pas légers : Ripple-65B, Kimi K3.1 Pro, ERNIE-5.5-Flash-Base, Qwen3.5 122B, PLaMo 3 31B.

## Sources
Ollama : pages de bibliothèque gemma4, qwen3.5, gpt-oss:20b. Fiche du modèle Google Gemma 4 (ai.google.dev) et huggingface.co/google/gemma-4-12B. huggingface.co/Qwen/Qwen3.5-0.8B. huggingface.co/LiquidAI/LFM3-1.7B (+ -GGUF). Introuvables sur toute page : le mois de sortie de Gemma 4, la longueur de contexte de gpt-oss, les commandes llama.cpp (aucune émise).

## Agents (appel d'outils, codage agentique, long contexte)

Ce dont Companion a besoin d'un modèle (lu dans `chat.rs`, `mcp.rs`) : les outils MCP sont proposés en texte brut, un bloc `tool` délimité que le modèle peut écrire et que `mcp::parse_tool_request` analyse, donc **aucun modèle n'a besoin du champ `tools` d'OpenAI**. Le modèle doit suivre ce format, diffuser en SSE avec `include_usage`, peut émettre du `reasoning_content` ou du texte `<think>` (séparé par `chat.rs`), et a besoin de contexte pour jusqu'à 5 tours de résultats d'outils. Les parties `image_url` ne comptent que pour la vision. Un badge « tools » est donc un bon signe, pas une exigence.

`for_agents(ram_gb)` ne renvoie que les modèles dont l'appel d'outils est documenté sur une page primaire, avec la même règle d'adéquation que `for_ram`. Sources documentées :

| modèle | appel d'outils (source) | codage agentique | sortie structurée |
|---|---|---|---|
| gpt-oss-20b | badge `tools` d'Ollama ; « designed for agentic tasks » | non vérifié | non vérifié |
| gemma4 e2b / e4b / 12b / 26b-a4b | badge `tools` d'Ollama, « native function-calling support » | non vérifié | non vérifié |
| qwen3.5 0.8b / 4b | badge `tools` d'Ollama (famille) | non vérifié | non vérifié |
| lfm3-1.7b | fiche HF : « designed for local agents, tool calling » (appels XML) | non vérifié | fiche HF : « structured outputs » (non vérifié en test) |
| qwen3.5 9b | badge `tools` d'Ollama | non vérifié | non vérifié |
| lfm2.5-2.6b, olmoe-1b-7b | non vérifié | non vérifié | non vérifié |

Pas dans la liste des agents (ni taille ni licence sur une page primaire, ou pas local) :
- Devstral Small 2 24B : la doc Mistral indique Apache 2.0, 256k, appel de fonctions, sorties structurées, 2025-12-09, mais l'état de l'API est Deprecated (retiré le 2026-03-31) et aucune taille locale n'est imprimée.
- Qwen3-Coder-30B-A3B : la fiche Qwen indique appel d'outils et codage agentique, contexte 262,144, mais n'imprime ni licence ni taille ; le README de colibri le liste à la ligne 209.
- Granite 4.2 : la doc IBM documente l'appel d'outils et le contexte 128K, sans taille.
- MiMo distill, Mellum2 (codage agentique affirmé seulement sur des pages non lues), Nemotron 3.5 Bolt (vLLM seulement).

Limites : les petits modèles appellent les outils de façon peu fiable, et un badge dit que le modèle de chat prend en charge les outils, pas que le modèle choisit bien. Rien ici n'a été exécuté avec les automatisations de Companion.

## Revérification du 2026-11-17 (catalogue complet)
Relu aujourd'hui : Ollama gemma4 (toutes les tailles/contextes concordent), qwen3.5 (tailles, badges ; la page ne nomme aucune licence), gpt-oss:20b (14 Go, outils, Apache 2.0, pas de contexte) ; HF gemma-4-12B, Qwen3.5-0.8B/4B, LFM3-1.7B, LFM2.5-2.6B ; docs IBM Granite 4.2.
Corrigé : contexte LFM2.5 (131,072, était « non indiqué ») ; appel d'outils LFM3 désormais documenté ; Qwen3.5 9B ajouté.
Non relu : OLMoE (chiffres du README), licence Gemma 4 E2B/E4B/26B (fiche de famille lue plus tôt ; fiche 12B lue aujourd'hui), taille GGUF de LFM3, taille Q4_K_M de LFM2.5 (la page HF ne donne que des liens), mois de sortie (aucune page primaire ne les imprime ; le 27 oct. de LFM3 vient d'une synthèse).
Les pages Ollama gemma4, qwen3.5 et gpt-oss n'impriment aucune licence ; les licences viennent des fiches HF/Google.
