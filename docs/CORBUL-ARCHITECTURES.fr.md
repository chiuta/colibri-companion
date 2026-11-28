# Modèles que corbul exécute et que colibri n'exécute pas

Companion lance **corbul**, son propre moteur, pour tout GGUF que colibri ne peut pas importer. Depuis alpha.18 il y va **directement** pour les familles ci-dessous, sans import qui échoue d'abord.

## Ce que corbul exécute
- **Gemma 4** (E2B, E4B, 26B-A4B à experts, 31B), depuis un `.gguf` : attention mixte glissante et globale, tailles de tête par couche, clés utilisées comme valeurs sur les couches globales, RoPE proportionnel, K/V partagé dans les dernières couches, embeddings par couche, MLP dense à côté des experts routés, plafonnement des logits finaux.
- **GPT-2, GPT-J, GPT-NeoX/Pythia, Falcon, BLOOM, MPT**, depuis un `.gguf` : LayerNorm, QKV fusionné ou séparé, positions apprises / rotatives / ALiBi, résidu parallèle. Testés seulement sur de petits modèles synthétiques. OPT n'a pas de conversion GGUF.
- **Gemma 1–3, Phi, OLMo, Granite, Cohere, InternLM2, StableLM, StarCoder2, EXAONE, Baichuan, MiniCPM, SmolLM3**, Llama et Mistral, Qwen et les autres familles à experts que le moteur avait déjà.
- **Mamba, Mamba-2, RWKV-6/7, Jamba, Bamba** (récurrents et hybrides).

`corbul serve MODÈLE` répond à `/health`, `/v1/models`, `/v1/chat/completions` et `/v1/completions` (aussi en flux) avec le modèle de chat du modèle, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k` et `seed`. Une conversation qui prolonge la précédente ne relit pas son historique.

## Ce qu'il ne fait pas pour ces modèles
Appels d'outils, sortie structurée, plusieurs réponses par requête, embeddings, paquets de contexte (Companion utilise alors le chat ordinaire), images en entrée, plusieurs modèles à la fois. Gemma 4 est lu uniquement depuis un `.gguf`, pas depuis des dossiers safetensors. Un GGUF sans vocabulaire propre peut utiliser un `tokenizer.json` placé à côté.

## Ce qui n'est pas vérifié
Tout est testé sur de **petits modèles synthétiques** (invariants propres : cache égal au recalcul, fenêtres glissantes, K/V partagé, routage, experts quantifiés, réponses HTTP). **Rien n'a été comparé à un vrai fichier Gemma 4.** Si les réponses d'un vrai fichier semblent fausses, lance toi-même `corbul serve FICHIER.gguf` et envoie la sortie ; on vérifie d'abord le motif glissant/global et les tables RoPE.

## Pas encore pris en charge
OPT, gpt-oss, PhiMoE, DBRX, Llama 4, GLM-4-MoE, Hunyuan, Ernie-MoE, Grok, le modèle DeepSeek complet, MiniMax, Zamba, modèles encodeur-décodeur. corbul les nomme et ne devine pas.
