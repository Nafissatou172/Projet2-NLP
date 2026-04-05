"""
Évaluateurs pour les 4 piliers du benchmark :
  1. Qualité / Exactitude   → similarité cosinus avec sentence-transformers
  2. Fidélité au contexte   → overlap de mots-clés (keywords recall)
  3. Latence               → temps de réponse mesuré directement
  4. Coût                  → calculé depuis les tokens consommés
"""

import re
import math
from typing import Optional


# ─────────────────────────────────────────────
#  Utilitaires texte
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


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity entre deux vecteurs TF (bag-of-words)."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency (non normalisé)."""
    freq: dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq


# ─────────────────────────────────────────────
#  1. Qualité / Exactitude
# ─────────────────────────────────────────────

def evaluate_quality(generated: str, reference: str) -> float:
    """
    Similarité cosinus (bag-of-words TF) entre la réponse générée
    et la réponse de référence du dataset.
    Retourne un score entre 0 et 1.
    """
    if not generated or not reference:
        return 0.0
    tokens_gen = _tokenize(generated)
    tokens_ref = _tokenize(reference)
    if not tokens_gen or not tokens_ref:
        return 0.0
    return round(_cosine_similarity(_tf(tokens_gen), _tf(tokens_ref)), 4)


# ─────────────────────────────────────────────
#  2. Fidélité au contexte (keyword recall)
# ─────────────────────────────────────────────

def evaluate_faithfulness(generated: str, reference: str) -> float:
    """
    Proportion des mots-clés de la référence présents dans la réponse générée.
    Mesure si le modèle n'hallucine pas et reste fidèle au contexte.
    Retourne un score entre 0 et 1.
    """
    if not generated or not reference:
        return 0.0
    ref_tokens = set(_tokenize(reference))
    gen_tokens = set(_tokenize(generated))
    if not ref_tokens:
        return 0.0
    overlap = ref_tokens & gen_tokens
    return round(len(overlap) / len(ref_tokens), 4)


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
