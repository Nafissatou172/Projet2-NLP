"""
raft.py — Simulation du pipeline RAFT (Retrieval-Augmented Fine-Tuning)

RAFT est une technique qui combine :
  1. La génération d'un dataset de fine-tuning synthétique (question + documents sources + CoT)
  2. Un modèle "fine-tuné" capable d'ignorer les documents non-pertinents (distracteurs)

Ce module simule les deux phases :
  Phase A — Génération du dataset RAFT (construction des exemples d'entraînement)
  Phase B — Inférence RAFT (le modèle répond en identifiant les docs oracle vs distracteurs)
"""

import re
import time
import random
from typing import List, Dict, Any, Tuple, Optional

from benchmark.models_config import MODELS_CONFIG, SYSTEM_PROMPT
from benchmark.model_clients import call_model

# ─────────────────────────────────────────────
#  Constantes de configuration RAFT
# ─────────────────────────────────────────────

# Ratio documents oracle / distracteurs (papier RAFT : p = 1/k, k = 4 docs)
RAFT_NUM_DOCS = 4           # Nombre total de documents dans le contexte RAFT
RAFT_ORACLE_RATIO = 0.5     # Proportion de documents oracle (pertinents)
RAFT_DISTRACTOR_RATIO = 0.5 # Proportion de distracteurs (non-pertinents)

# Modèle utilisé pour générer les réponses RAFT
RAFT_PRIMARY_MODEL = "mistral"
RAFT_FALLBACK_MODEL = "qwen"

# Nombre max de tokens pour la réponse RAFT (plus long car CoT)
RAFT_MAX_TOKENS = 2000

# ─────────────────────────────────────────────
#  Prompt spécifique RAFT (Chain-of-Thought)
# ─────────────────────────────────────────────

RAFT_SYSTEM_PROMPT = """Tu es un assistant financier expert formé avec la technique RAFT (Retrieval-Augmented Fine-Tuning).

Ta mission est de répondre aux questions financières en suivant un raisonnement structuré (Chain-of-Thought) :

1. IDENTIFIER les documents oracle (pertinents) parmi les documents fournis
2. IGNORER les distracteurs (documents non-pertinents ou trompeurs)
3. EXTRAIRE les informations clés des documents oracle
4. SYNTHÉTISER une réponse précise et fidèle

Format de réponse attendu :
<raisonnement>
Analyse des documents : [identifier lesquels sont pertinents et pourquoi]
Informations clés extraites : [lister les faits importants]
</raisonnement>
<réponse>
[Réponse finale claire et concise basée uniquement sur les documents oracle]
</réponse>

Si aucun document oracle n'est trouvé, utilise tes connaissances générales en finance."""


RAFT_INFERENCE_PROMPT_TEMPLATE = """Voici {num_docs} documents financiers. Certains sont pertinents (oracle), d'autres sont des distracteurs.

{docs_section}

Question : {question}

Réponds en identifiant d'abord les documents oracle, puis fournis une réponse basée sur eux."""


# ─────────────────────────────────────────────
#  Phase A : Génération du dataset RAFT
# ─────────────────────────────────────────────

def generate_raft_dataset_entry(
    question: str,
    oracle_docs: List[str],
    distractor_docs: List[str],
    reference_answer: str,
) -> Dict[str, Any]:
    """
    Crée un exemple d'entraînement RAFT au format :
    {
        "question": ...,
        "oracle_docs": [...],
        "distractor_docs": [...],
        "all_docs": [...],  # mélangés
        "cot_answer": ...,  # réponse avec chaîne de pensée
        "final_answer": ...
    }
    """
    # Mélanger les documents oracle et distracteurs
    all_docs = oracle_docs + distractor_docs
    random.shuffle(all_docs)

    # Construire un CoT simulé
    cot_parts = []
    cot_parts.append("## Analyse des documents :")
    for i, doc in enumerate(all_docs):
        is_oracle = doc in oracle_docs
        label = "✓ ORACLE (pertinent)" if is_oracle else "✗ DISTRACTEUR (ignorer)"
        cot_parts.append(f"  Document {i+1} [{label}]: {doc[:120]}...")

    cot_parts.append("\n## Extraction des informations clés :")
    if oracle_docs:
        for doc in oracle_docs:
            cot_parts.append(f"  → {doc[:200]}...")

    cot_parts.append("\n## Réponse finale :")
    cot_parts.append(f"  {reference_answer}")

    cot_answer = "\n".join(cot_parts)

    return {
        "question": question,
        "oracle_docs": oracle_docs,
        "distractor_docs": distractor_docs,
        "all_docs": all_docs,
        "num_oracle": len(oracle_docs),
        "num_distractors": len(distractor_docs),
        "cot_answer": cot_answer,
        "final_answer": reference_answer,
    }


def build_raft_dataset(
    questions: List[Dict[str, Any]],
    collection,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Construit un dataset RAFT complet à partir des questions et de la collection vectorielle.
    
    Pour chaque question :
      - Récupère les top_k documents pertinents (oracle candidates)
      - Sélectionne des distracteurs aléatoirement depuis d'autres questions
      - Génère l'entrée RAFT
    """
    dataset = []
    all_chunks = []

    # Collecter tous les chunks disponibles pour les distracteurs
    try:
        all_data = collection.get(include=["documents"])
        all_chunks = all_data.get("documents", [])
    except Exception:
        all_chunks = []

    for item in questions:
        question = item.get("question", "")
        reference = item.get("reponse", "")

        if not question or not reference:
            continue

        # Récupérer les documents oracle via recherche vectorielle
        oracle_docs = []
        try:
            results = collection.query(
                query_texts=[question],
                n_results=min(top_k, max(1, int(top_k * RAFT_ORACLE_RATIO))),
            )
            oracle_docs = results.get("documents", [[]])[0]
        except Exception:
            oracle_docs = []

        # Sélectionner des distracteurs aléatoires
        num_distractors = top_k - len(oracle_docs)
        distractor_pool = [c for c in all_chunks if c not in oracle_docs]
        distractors = random.sample(
            distractor_pool,
            min(num_distractors, len(distractor_pool))
        ) if distractor_pool else []

        entry = generate_raft_dataset_entry(
            question=question,
            oracle_docs=oracle_docs,
            distractor_docs=distractors,
            reference_answer=reference,
        )
        dataset.append(entry)

    return dataset


# ─────────────────────────────────────────────
#  Phase B : Inférence RAFT (modèle simulé fine-tuné)
# ─────────────────────────────────────────────

def _build_raft_context(oracle_docs: List[str], distractor_docs: List[str]) -> str:
    """
    Construit le contexte RAFT en mélangeant oracles et distracteurs,
    puis en les numérotant (sans révéler lesquels sont oracles).
    """
    all_docs = oracle_docs + distractor_docs
    random.shuffle(all_docs)

    doc_section = ""
    for i, doc in enumerate(all_docs):
        doc_preview = doc[:500] + "..." if len(doc) > 500 else doc
        doc_section += f"\n--- Document {i+1} ---\n{doc_preview}\n"

    return doc_section, all_docs


def _parse_raft_response(raw_text: str) -> Tuple[str, str]:
    """
    Parse la réponse RAFT en extrayant le raisonnement et la réponse finale.
    Retourne (raisonnement, réponse_finale).
    """
    # Essayons d'extraire les balises structurées
    reasoning_match = re.search(
        r'<raisonnement>(.*?)</raisonnement>',
        raw_text, re.DOTALL | re.IGNORECASE
    )
    answer_match = re.search(
        r'<r[ée]ponse>(.*?)</r[ée]ponse>',
        raw_text, re.DOTALL | re.IGNORECASE
    )

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        # Fallback : prendre le début du texte comme raisonnement
        lines = raw_text.strip().split('\n')
        reasoning = '\n'.join(lines[:-1]) if len(lines) > 1 else ""

    if answer_match:
        final_answer = answer_match.group(1).strip()
    else:
        # Fallback : prendre la dernière partie significative
        final_answer = raw_text.strip()
        # Retirer les balises résiduelles
        final_answer = re.sub(r'<[^>]+>', '', final_answer).strip()

    return reasoning, final_answer


def clean_raft_response(text: str) -> str:
    """Nettoie le texte markdown pour une sortie propre."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '  • ', text, flags=re.MULTILINE)
    text = text.replace('*', '')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def run_raft_inference(
    question: str,
    collection,
    num_oracle: int = 2,
    num_distractors: int = 2,
) -> Dict[str, Any]:
    """
    Effectue une inférence RAFT complète :
      1. Récupère les documents oracle depuis la collection
      2. Sélectionne des distracteurs aléatoires
      3. Construit le prompt RAFT avec documents mélangés
      4. Appelle le LLM avec le système RAFT
      5. Parse et retourne la réponse structurée

    Returns:
        dict avec les clés :
          - question, response, reasoning, final_answer
          - oracle_docs, distractor_docs, all_docs
          - model_used, latency_seconds, tokens
          - raft_stats (nb docs oracle/distracteurs identifiés)
    """
    start_time = time.perf_counter()

    # 1. Récupérer les documents oracle
    oracle_docs = []
    oracle_metas = []
    try:
        results = collection.query(
            query_texts=[question],
            n_results=num_oracle,
        )
        oracle_docs = results.get("documents", [[]])[0]
        oracle_metas = results.get("metadatas", [[]])[0]
    except Exception as e:
        pass

    # 2. Récupérer des distracteurs (documents peu pertinents)
    distractor_docs = []
    distractor_metas = []
    try:
        # On cherche des docs avec une requête inversée/aléatoire pour trouver des distracteurs
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs_pool = all_data.get("documents", [])
        all_metas_pool = all_data.get("metadatas", [])

        # Filtrer les documents déjà sélectionnés comme oracle
        non_oracle_indices = [
            i for i, doc in enumerate(all_docs_pool)
            if doc not in oracle_docs
        ]

        if non_oracle_indices:
            selected_indices = random.sample(
                non_oracle_indices,
                min(num_distractors, len(non_oracle_indices))
            )
            distractor_docs = [all_docs_pool[i] for i in selected_indices]
            distractor_metas = [all_metas_pool[i] for i in selected_indices]
    except Exception:
        pass

    # 3. Construire le contexte RAFT (documents mélangés, sans labels)
    doc_section, all_docs_mixed = _build_raft_context(oracle_docs, distractor_docs)

    # 4. Construire le prompt final
    raft_prompt = RAFT_INFERENCE_PROMPT_TEMPLATE.format(
        num_docs=len(all_docs_mixed),
        docs_section=doc_section,
        question=question,
    )

    # 5. Appel LLM
    config = MODELS_CONFIG.get(RAFT_PRIMARY_MODEL)
    if not config:
        config = MODELS_CONFIG.get(RAFT_FALLBACK_MODEL)

    if not config:
        return {
            "error": "Aucun modèle RAFT disponible",
            "question": question,
        }

    llm_response = call_model(
        provider=config["provider"],
        model_id=config["model_id"],
        system_prompt=RAFT_SYSTEM_PROMPT,
        question=raft_prompt,
        max_tokens=RAFT_MAX_TOKENS,
        temperature=0.3,  # Plus déterministe pour le RAFT
    )

    elapsed = time.perf_counter() - start_time

    if llm_response.get("error"):
        return {
            "error": llm_response["error"],
            "question": question,
        }

    raw_text = llm_response.get("text", "")

    # 6. Parser la réponse RAFT
    reasoning, final_answer = _parse_raft_response(raw_text)
    final_answer_clean = clean_raft_response(final_answer)
    reasoning_clean = clean_raft_response(reasoning)

    # 7. Statistiques RAFT
    raft_stats = {
        "num_oracle_docs": len(oracle_docs),
        "num_distractor_docs": len(distractor_docs),
        "total_docs_presented": len(all_docs_mixed),
        "oracle_ratio": round(len(oracle_docs) / max(len(all_docs_mixed), 1), 2),
        "has_reasoning": bool(reasoning),
        "has_structured_response": bool(
            re.search(r'<r[ée]ponse>', raw_text, re.IGNORECASE)
        ),
    }

    return {
        "question": question,
        "response": final_answer_clean,
        "reasoning": reasoning_clean,
        "raw_response": raw_text,
        "model_used": config["display_name"],
        "provider": config["provider"],
        "oracle_docs": [
            {
                "text": doc[:300] + "..." if len(doc) > 300 else doc,
                "source": meta.get("source", "unknown"),
            }
            for doc, meta in zip(oracle_docs, oracle_metas)
        ],
        "distractor_docs": [
            {
                "text": doc[:200] + "..." if len(doc) > 200 else doc,
                "source": meta.get("source", "unknown"),
            }
            for doc, meta in zip(distractor_docs, distractor_metas)
        ],
        "raft_stats": raft_stats,
        "latency_seconds": round(elapsed, 3),
        "input_tokens": llm_response.get("input_tokens", 0),
        "output_tokens": llm_response.get("output_tokens", 0),
    }


# ─────────────────────────────────────────────
#  Utilitaire : Génération du rapport RAFT
# ─────────────────────────────────────────────

def generate_raft_report(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Génère un rapport statistique sur le dataset RAFT construit.
    """
    if not dataset:
        return {"error": "Dataset vide"}

    total = len(dataset)
    avg_oracle = sum(e["num_oracle"] for e in dataset) / total
    avg_distractors = sum(e["num_distractors"] for e in dataset) / total
    avg_total_docs = avg_oracle + avg_distractors

    return {
        "dataset_size": total,
        "avg_oracle_docs_per_entry": round(avg_oracle, 2),
        "avg_distractor_docs_per_entry": round(avg_distractors, 2),
        "avg_total_docs_per_entry": round(avg_total_docs, 2),
        "oracle_ratio": round(avg_oracle / avg_total_docs, 2) if avg_total_docs > 0 else 0,
        "sample_entry": dataset[0] if dataset else None,
        "description": (
            "Dataset RAFT généré pour le fine-tuning. Chaque entrée contient une question, "
            "des documents oracle (pertinents), des distracteurs, et une réponse avec "
            "Chain-of-Thought pour apprendre au modèle à filtrer les documents non-pertinents."
        ),
    }
