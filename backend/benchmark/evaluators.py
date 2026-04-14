"""
Évaluateurs pour les 4 piliers du benchmark :
  1. Qualité / Exactitude   → similarité cosinus sémantique (sentence-transformers)
  2. Fidélité au contexte   → similarité sémantique + keyword recall combinés
  3. Latence               → temps de réponse mesuré directement
  4. Coût                  → calculé depuis les tokens consommés
"""

import re
import math
from typing import Optional

from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
#  Modèle d'embedding (chargé une seule fois)
# ─────────────────────────────────────────────

_model = None

def _get_model():
    """Charge le modèle multilingue une seule fois."""
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


# ─────────────────────────────────────────────
#  Utilitaires texte (pour keyword recall)
# ─────────────────────────────────────────────

_STOPWORDS_FR = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "est", "il", "elle", "que", "qui", "ce", "se", "sa", "son", "ses",
    "au", "aux", "par", "sur", "dans", "avec", "pour", "ou", "ne",
    "pas", "plus", "mais", "donc", "or", "ni", "car", "à", "l", "d",
    "y", "on", "je", "tu", "nous", "vous", "ils", "elles",
}


def _tokenize(text: str) -> list[str]:
    """Tokenise un texte en mots en minuscule, sans ponctuation."""
    tokens = re.findall(r'\b[a-zéèêëàâùûüîïôçœæ]+\b', text.lower())
    return [t for t in tokens if t not in _STOPWORDS_FR and len(t) > 2]


# ─────────────────────────────────────────────
#  1. Qualité / Exactitude (similarité sémantique)
# ─────────────────────────────────────────────

def evaluate_quality(generated: str, reference: str) -> float:
    """
    Similarité cosinus sémantique entre la réponse générée
    et la réponse de référence, via un modèle multilingue.
    Retourne un score entre 0 et 1.
    """
    if not generated or not reference:
        return 0.0
    model = _get_model()
    emb_gen = model.encode(generated, convert_to_tensor=True)
    emb_ref = model.encode(reference, convert_to_tensor=True)
    sim = util.cos_sim(emb_gen, emb_ref).item()
    return round(max(0.0, sim), 4)


# ─────────────────────────────────────────────
#  2. Fidélité au contexte (sémantique + keywords)
# ─────────────────────────────────────────────

def evaluate_faithfulness(generated: str, reference: str) -> float:
    """
    Score combiné :
      - 70% similarité sémantique (le sens est-il préservé ?)
      - 30% keyword recall (les termes-clés de la référence sont-ils présents ?)
    Retourne un score entre 0 et 1.
    """
    if not generated or not reference:
        return 0.0

    # Partie sémantique (70%)
    model = _get_model()
    emb_gen = model.encode(generated, convert_to_tensor=True)
    emb_ref = model.encode(reference, convert_to_tensor=True)
    semantic_sim = max(0.0, util.cos_sim(emb_gen, emb_ref).item())

    # Partie keyword recall (30%)
    ref_tokens = set(_tokenize(reference))
    gen_tokens = set(_tokenize(generated))
    if ref_tokens:
        keyword_recall = len(ref_tokens & gen_tokens) / len(ref_tokens)
    else:
        keyword_recall = 0.0

    score = 0.70 * semantic_sim + 0.30 * keyword_recall
    return round(score, 4)


# ─────────────────────────────────────────────
#  3. Latence
# ─────────────────────────────────────────────

def evaluate_latency(latency_seconds: float) -> dict:
    """
    Classifie la latence en catégorie de performance.
    Retourne le temps brut + une note sur 1.
    """
    if latency_seconds <= 1.0:
        score = 1.0
        label = "Excellent"
    elif latency_seconds <= 3.0:
        score = 0.8
        label = "Bon"
    elif latency_seconds <= 6.0:
        score = 0.6
        label = "Acceptable"
    elif latency_seconds <= 10.0:
        score = 0.3
        label = "Lent"
    else:
        score = 0.0
        label = "Très lent"

    return {
        "seconds": round(latency_seconds, 3),
        "score": score,
        "label": label,
    }


# ─────────────────────────────────────────────
#  4. Coût
# ─────────────────────────────────────────────

def evaluate_cost(
    input_tokens: int,
    output_tokens: int,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
) -> dict:
    """
    Calcule le coût estimé en USD à partir des tokens consommés.
    """
    cost_input = (input_tokens / 1000) * cost_per_1k_input
    cost_output = (output_tokens / 1000) * cost_per_1k_output
    total = cost_input + cost_output

    # Score inversé : moins c'est cher, mieux c'est (normalisé sur $0.01 max)
    score = max(0.0, 1.0 - (total / 0.01))

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(total, 6),
        "score": round(score, 4),
    }


# ─────────────────────────────────────────────
#  Score global
# ─────────────────────────────────────────────

WEIGHTS = {
    "quality": 0.40,
    "faithfulness": 0.30,
    "latency": 0.20,
    "cost": 0.10,
}


def compute_global_score(
    quality: float,
    faithfulness: float,
    latency_score: float,
    cost_score: float,
) -> float:
    """Score agrégé pondéré sur 1."""
    score = (
        quality * WEIGHTS["quality"]
        + faithfulness * WEIGHTS["faithfulness"]
        + latency_score * WEIGHTS["latency"]
        + cost_score * WEIGHTS["cost"]
    )
    return round(score, 4)
