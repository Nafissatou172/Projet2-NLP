"""
Configuration des modèles pour le benchmark.
Supporte : Mistral, LLaMA (via HuggingFace Inference API), Claude (Anthropic).
Les clés API sont lues depuis les variables d'environnement.
"""

import os

MODELS_CONFIG = {
    "mistral": {
        "display_name": "Mistral 7B Instruct",
        "provider": "mistral_api",
        "model_id": "mistral-tiny",  # ou "open-mistral-7b"
        "max_tokens": 512,
        "temperature": 0.1,
        "cost_per_1k_input_tokens": 0.00025,
        "cost_per_1k_output_tokens": 0.00025,
    },
    "llama": {
        "display_name": "LLaMA 3 8B Instruct",
        "provider": "groq_api",  # ou "llama_api" selon votre fournisseur
        "model_id": "llama-3.1-8b-instant",
        "max_tokens": 512,
        "temperature": 0.1,
        "cost_per_1k_input_tokens": 0.00025,
        "cost_per_1k_output_tokens": 0.00025,
    },
    "qwen": {
        "display_name": "Qwen 2.5 7B Instruct",
        "provider": "huggingface_chat",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "max_tokens": 512,
        "temperature": 0.1,
        "cost_per_1k_input_tokens": 0.0,
        "cost_per_1k_output_tokens": 0.0,
    },
}

# Chemin vers le dataset d'évaluation
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "dataset-finance.csv",
)

# Nombre de questions à utiliser pour le benchmark (None = toutes)
DEFAULT_SAMPLE_SIZE = 10

# Prompt système commun
SYSTEM_PROMPT = """Tu es un expert en finances personnelles. Tu réponds aux questions de manière claire, utile et concise.

RÈGLES IMPORTANTES :
- Tu ne dois utiliser AUCUN caractère de formatage (pas d'astérisques *, pas de tirets - , pas de dièses #, pas de backticks `, pas de gras ou italique).
- Tu ne dois pas utiliser de Markdown.
- Tu rédiges tes réponses en texte brut uniquement, avec des paragraphes séparés par des sauts de ligne.
- Pour les listes, utilise des lettres (a), b), c) ) ou des chiffres (1., 2., 3.) sans aucun symbole spécial.
- Tu ne mets jamais d'étoiles, de doubles astérisques, de tirets longs, ou de barres horizontales (---).
- Les réponses doivent être naturelles, sans mise en forme visuelle excessive.

Exemple de style attendu :
"Pour bien dépenser son argent, il faut d'abord établir un budget. La méthode 50/30/20 est recommandée : 50 % pour les besoins essentiels, 30 % pour les loisirs, 20 % pour l'épargne. Ensuite, il faut prioriser les dépenses et éliminer les abonnements inutiles."

Tu réponds toujours en français, de façon professionnelle mais accessible."""
