"""
multi_agent.py — Pipeline Multi-Agent avec enrichissement RAFT

Pipeline :
  1. Searcher Agent  → extrait les mots-clés
  2. Critic Agent    → valide si le contexte est pertinent (boucle)
  3. [RAFT]          → inject distracteurs dans le contexte validé
  4. Generator Agent → raisonnement CoT RAFT → réponse finale structurée

Retourne : response, reasoning, oracle_docs, distractor_docs, raft_stats
"""

import re
import random
import time
from typing import Dict, Any, Tuple, List

from benchmark.model_clients import call_model
from benchmark.models_config import MODELS_CONFIG, SYSTEM_PROMPT
from utils.rag_optimized_functions import vector_rerank_search, rerank, FINAL_COUNT, DEFAULT_RETRIEVAL_COUNT

# ─────────────────────────────────────────────
#  Constantes RAFT
# ─────────────────────────────────────────────
MULTI_RAFT_NUM_DISTRACTORS = 2   # distracteurs injectés pour le Generator
MULTI_RAFT_MAX_ORACLE = 3        # docs oracle max envoyés au Generator

# ─────────────────────────────────────────────
#  Prompts des agents
# ─────────────────────────────────────────────
SEARCHER_PROMPT = """Tu es l'agent 'Rechercheur' (Searcher).
Ton rôle est de lire la question de l'utilisateur et d'en extraire 1 à 3 mots-clés essentiels pour trouver la réponse dans une base de documents financiers.
Tu dois répondre UNIQUEMENT par les mots-clés séparés par des espaces. 
Exemple pour la question "Quels sont les objectifs du Fonsis ?" : Fonsis objectifs
Exemple pour la question "Comment ouvrir un compte ?" : ouvrir compte
Ne fais aucune phrase. Ne réponds rien d'autre que les mots-clés.
"""

CRITIC_PROMPT = """Tu es l'agent 'Critique' (Critic).
Ton rôle est d'évaluer si le CONTEXTE fourni contient suffisamment d'informations claires pour répondre à la QUESTION de l'utilisateur.
Si le contexte suffit pour répondre, tu DOIS répondre exactement par le mot: PASS
Si le contexte est hors sujet ou ne contient pas de réponse utile, tu DOIS répondre par le mot: FAIL suivi d'une courte suggestion de nouveaux mots-clés à chercher.

Exemple de réponse 1:
PASS

Exemple de réponse 2:
FAIL: stratégie investissement actions
"""

# Prompt CoT RAFT pour le Generator
GENERATOR_RAFT_PROMPT = """Tu es l'agent 'Generator' entraîné avec RAFT (Retrieval-Augmented Fine-Tuning).
Tu reçois plusieurs documents financiers récupérés par ton équipe. Certains sont pertinents (oracle), d'autres sont des distracteurs.

ÉTAPE 1 — IDENTIFIER LES ORACLES :
Pour chaque document, note s'il est pertinent (oracle ✓) ou non (distracteur ✗) pour la question.

ÉTAPE 2 — EXTRAIRE LES FAITS CLÉS :
Depuis les documents oracle uniquement, extrais les informations essentielles.

ÉTAPE 3 — GÉNÉRER LA RÉPONSE :
Produis une réponse claire, précise et en français basée EXCLUSIVEMENT sur les oracles.

Format OBLIGATOIRE :
<raisonnement>
[Analyse doc par doc + extraction des faits clés]
</raisonnement>
<réponse>
[Réponse finale concise destinée à l'utilisateur]
</réponse>
"""


# ─────────────────────────────────────────────
#  Helpers internes
# ─────────────────────────────────────────────

def execute_search(collection, query: str) -> Tuple[List[str], List[Dict]]:
    if collection.count() == 0:
        return [], []
    retrieved_docs, retrieved_metas = vector_rerank_search(collection, query, DEFAULT_RETRIEVAL_COUNT)
    if not retrieved_docs:
        return [], []
    reranked = rerank(query, retrieved_docs, top_k=FINAL_COUNT)
    best_docs = [retrieved_docs[i] for i, _ in reranked]
    best_metas = [retrieved_metas[i] for i, _ in reranked]
    return best_docs, best_metas


def _pick_distractors(collection, oracle_docs: List[str], n: int) -> Tuple[List[str], List[str]]:
    """Sélectionne n documents non-oracle comme distracteurs."""
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])
        pool = [(d, m) for d, m in zip(all_docs, all_metas) if d not in oracle_docs]
        selected = random.sample(pool, min(n, len(pool))) if pool else []
        return [d for d, _ in selected], [m.get("source", "unknown") for _, m in selected]
    except Exception:
        return [], []


def _build_mixed_context(oracle_docs: List[str], distractor_docs: List[str]) -> Tuple[str, List[str]]:
    """Mélange oracles et distracteurs, retourne la section texte + liste mélangée."""
    all_docs = oracle_docs[:MULTI_RAFT_MAX_ORACLE] + distractor_docs
    random.shuffle(all_docs)
    section = ""
    for i, doc in enumerate(all_docs):
        preview = doc[:500] + "..." if len(doc) > 500 else doc
        section += f"\n--- Document {i+1} ---\n{preview}\n"
    return section, all_docs


def _parse_cot_response(raw: str) -> Tuple[str, str]:
    """Parse le raisonnement et la réponse finale du format CoT RAFT."""
    reasoning_m = re.search(r'<raisonnement>(.*?)</raisonnement>', raw, re.DOTALL | re.IGNORECASE)
    answer_m    = re.search(r'<r[ée]ponse>(.*?)</r[ée]ponse>',   raw, re.DOTALL | re.IGNORECASE)

    reasoning = reasoning_m.group(1).strip() if reasoning_m else raw.strip()
    answer    = answer_m.group(1).strip()    if answer_m    else raw.strip()
    answer    = re.sub(r'<[^>]+>', '', answer).strip()
    return reasoning, answer


# ─────────────────────────────────────────────
#  Fonction principale
# ─────────────────────────────────────────────

def run_multi_agent_rag(collection, question: str, max_search_loops: int = 3) -> Dict[str, Any]:
    """
    Pipeline Multi-Agent RAFT :
      Searcher → Critic (boucle) → [injection distracteurs] → Generator CoT RAFT
    """
    config = MODELS_CONFIG.get("mistral")
    if not config:
        config = MODELS_CONFIG.get("qwen")

    provider = config["provider"]
    model_id  = config["model_id"]

    t_start = time.perf_counter()
    total_in_tokens  = 0
    total_out_tokens = 0
    oracle_chunks: List[str] = []
    oracle_metas_list: List[Dict] = []
    all_sources = set()
    agent_trace = []

    current_search_query = question
    loop_count     = 0
    context_passed = False
    best_docs: List[str] = []

    # ── Boucle Searcher → Critic ────────────────────────────────
    while loop_count < max_search_loops:
        loop_count += 1

        # 1. Searcher
        agent_trace.append({"role": "Searcher", "content": f"Formulation de la requête : '{current_search_query}'"})
        searcher_res = call_model(
            provider=provider, model_id=model_id,
            system_prompt=SEARCHER_PROMPT,
            question=f"Question : {current_search_query}",
            max_tokens=50, temperature=0.1,
        )
        if searcher_res["error"]:
            return {"error": f"Erreur Searcher IA: {searcher_res['error']}"}

        total_in_tokens  += searcher_res["input_tokens"]
        total_out_tokens += searcher_res["output_tokens"]

        search_query = searcher_res["text"].strip().replace('"', '')
        agent_trace.append({"role": "Searcher", "content": f"Mots-clés : '{search_query}'"})

        # Exécution de la recherche
        best_docs, best_metas = execute_search(collection, search_query)
        for d in best_docs:
            if d not in oracle_chunks:
                oracle_chunks.append(d)
        for m in best_metas:
            all_sources.add(m.get("source", "unknown"))
            oracle_metas_list.append(m)

        if not best_docs:
            agent_trace.append({"role": "Searcher", "content": "Aucun document trouvé."})
            context_text = "Base de données vide ou aucun résultat."
        else:
            context_text = "\n---\n".join(best_docs)
            agent_trace.append({"role": "Searcher", "content": f"{len(best_docs)} extraits récupérés."})

        # 2. Critic
        critic_input = f"QUESTION: {question}\n\nCONTEXTE RECUPERE:\n{context_text}"
        critic_res = call_model(
            provider=provider, model_id=model_id,
            system_prompt=CRITIC_PROMPT,
            question=critic_input,
            max_tokens=100, temperature=0.1,
        )
        if critic_res["error"]:
            return {"error": f"Erreur Critic IA: {critic_res['error']}"}

        total_in_tokens  += critic_res["input_tokens"]
        total_out_tokens += critic_res["output_tokens"]

        critic_response = critic_res["text"].strip()

        if critic_response.upper().startswith("PASS"):
            agent_trace.append({"role": "Critic", "content": "PASS — Le contexte est pertinent."})
            context_passed = True
            break
        else:
            agent_trace.append({"role": "Critic", "content": f"FAIL — {critic_response}"})
            if "FAIL" in critic_response.upper() and ":" in critic_response:
                current_search_query = critic_response.split(":", 1)[1].strip()
            else:
                current_search_query = question

    # ── Phase RAFT : injection distracteurs ─────────────────────
    oracle_for_raft  = oracle_chunks[:MULTI_RAFT_MAX_ORACLE]
    distractor_texts, distractor_sources = _pick_distractors(
        collection, oracle_chunks, MULTI_RAFT_NUM_DISTRACTORS
    )

    doc_section, all_mixed = _build_mixed_context(oracle_for_raft, distractor_texts)

    agent_trace.append({
        "role": "RAFT",
        "content": (
            f"Injection de {len(distractor_texts)} distracteurs. "
            f"Total présenté au Generator : {len(all_mixed)} docs."
        )
    })

    # ── Generator RAFT CoT ──────────────────────────────────────
    generator_prompt = (
        f"Question de l'utilisateur : {question}\n\n"
        f"Voici {len(all_mixed)} documents financiers "
        f"(validés par le Critic + distracteurs mélangés) :\n"
        f"{doc_section}"
    )

    agent_trace.append({"role": "Generator", "content": "Raisonnement CoT RAFT en cours..."})

    gen_res = call_model(
        provider=provider, model_id=model_id,
        system_prompt=GENERATOR_RAFT_PROMPT,
        question=generator_prompt,
        max_tokens=2000, temperature=0.2,
    )

    total_latency    = round(time.perf_counter() - t_start, 3)
    total_in_tokens  += gen_res.get("input_tokens",  0)
    total_out_tokens += gen_res.get("output_tokens", 0)

    if gen_res.get("error"):
        return {"error": f"Erreur Generator IA: {gen_res['error']}", "trace": agent_trace}

    raw_cot = gen_res.get("text", "")
    reasoning, final_answer = _parse_cot_response(raw_cot)

    agent_trace.append({"role": "Generator", "content": "Réponse finale structurée générée."})

    # ── Stats RAFT ──────────────────────────────────────────────
    raft_stats = {
        "num_oracle_docs":         len(oracle_for_raft),
        "num_distractor_docs":     len(distractor_texts),
        "total_docs_presented":    len(all_mixed),
        "oracle_ratio":            round(len(oracle_for_raft) / max(len(all_mixed), 1), 2),
        "has_reasoning":           bool(reasoning),
        "has_structured_response": bool(re.search(r'<r[ée]ponse>', raw_cot, re.IGNORECASE)),
        "searcher_loops":          loop_count,
        "context_validated":       context_passed,
    }

    # Enrichir les oracle_docs avec source
    oracle_sources_list = [m.get("source", "unknown") for m in oracle_metas_list[:MULTI_RAFT_MAX_ORACLE]]

    return {
        "question":    question,
        "response":    final_answer,
        "reasoning":   reasoning,
        "raw_cot":     raw_cot,
        "model_used":  config["display_name"],
        "loops_count": loop_count,
        "context_passed": context_passed,
        "sources":     list(all_sources),
        "oracle_docs": [
            {"text": d[:300] + "..." if len(d) > 300 else d, "source": s}
            for d, s in zip(oracle_for_raft, oracle_sources_list)
        ],
        "distractor_docs": [
            {"text": d[:200] + "..." if len(d) > 200 else d, "source": s}
            for d, s in zip(distractor_texts, distractor_sources)
        ],
        "raft_stats":    raft_stats,
        "traces":        agent_trace,
        "latency_seconds": total_latency,
        "input_tokens":  total_in_tokens,
        "output_tokens": total_out_tokens,
    }
