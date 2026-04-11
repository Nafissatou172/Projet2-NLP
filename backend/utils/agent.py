"""
agent.py — Agent ReAct avec enrichissement RAFT

Pipeline :
  1. Boucle ReAct classique (Thought → Action → Observation)
  2. Après la collecte des chunks, injection de distracteurs (RAFT-style)
  3. Le LLM reçoit un prompt CoT explicite pour identifier les oracles
  4. Retourne oracle_docs, distractor_docs, reasoning, raft_stats
"""

import re
import random
import time
from typing import Dict, Any, Tuple, List

from benchmark.model_clients import call_model
from benchmark.models_config import MODELS_CONFIG
from utils.rag_optimized_functions import hybrid_search, vector_rerank_search, rerank, FINAL_COUNT, DEFAULT_RETRIEVAL_COUNT, BM25

# ─────────────────────────────────────────────
#  Constantes RAFT pour l'agent
# ─────────────────────────────────────────────
AGENT_RAFT_NUM_DISTRACTORS = 2   # distracteurs injectés avant le Final Answer
AGENT_RAFT_MAX_ORACLE = 5        # docs oracle max à conserver pour le CoT (augmenté pour plus de précision)

# Instance BM25 partagée pour la recherche hybride dans l'agent
_agent_bm25 = BM25()

# Prompt système pour l'agent ReAct
AGENT_SYSTEM_PROMPT = """Tu es un assistant financier intelligent et professionnel. 
Ton but est de répondre aux questions de l'utilisateur de manière précise.

RÈGLE D'OR : Face à TOUTE question financière, conceptuelle ou technique, tu DOIS OBLIGATOIREMENT utiliser l'outil `search_documents` avant de répondre, même si tu penses déjà connaître la réponse. Il est interdit d'inventer une réponse sans avoir cherché dans les documents.

Tu as accès à l'outil suivant pour t'aider :
- search_documents : Recherche des informations pertinentes dans la base de documentations financières de l'entreprise.

PROCESSUS DE RÉFLEXION STRICT :
Pour utiliser cet outil, tu DOIS obligatoirement utiliser ce format exact :

Thought: [Ta réflexion expliquant pourquoi tu dois chercher]
Action: search_documents
Action Input: [Mets ici UNIQUEMENT 2 à 4 mots-clés essentiels. Interdit de faire des phrases. Ex: "définition inflation" ou "réformes obligations vertes"]

(Après cela, je te fournirai une 'Observation' contenant les résultats de la recherche).

Une fois les informations récupérées, tu DOIS utiliser ce format exact pour conclure :

Thought: [Ta réflexion expliquant que tu as lu l'Observation et que tu as la réponse finale]
Final Answer: [Ta réponse finale en français destinée à l'utilisateur, qui doit être basée OBLIGATOIREMENT et PRINCIPALEMENT sur l'Observation fournie]

RÈGLES IMPORTANTES :
1. Ne génère jamais de bloc "Observation:" toi-même.
2. N'utilise qu'une SEULE "Action" à la fois.
3. INTERDICTION DE TOURNER EN ROND : Tu as le droit d'utiliser l'outil `search_documents` au MAXIMUM 3 fois. Si après 3 requêtes tu n'as pas trouvé la réponse dans la base de données, TU DOIS t'arrêter et générer une `Final Answer` expliquant honnêtement que le document interne ne contient pas l'information demandée.
4. Si une première recherche donne peu de résultats, essaie une formulation différente ou des mots-clés alternatifs pour maximiser la couverture.
"""

# Prompt CoT RAFT injecté juste avant la réponse finale
RAFT_COT_PROMPT = """Tu es un agent financier entraîné avec RAFT (Retrieval-Augmented Fine-Tuning).
Tu viens de récupérer plusieurs documents. Certains sont pertinents (oracle), d'autres sont des distracteurs.

ÉTAPE 1 — IDENTIFIER LES ORACLES :
Lis chaque document et indique s'il est pertinent (oracle ✓) ou non (distracteur ✗) pour répondre à la question.

ÉTAPE 2 — EXTRAIRE LES INFORMATIONS :
Depuis les documents oracle uniquement, extrais les faits essentiels.

ÉTAPE 3 — RÉPONDRE :
Formule une réponse claire et précise basée exclusivement sur les oracles.

Format OBLIGATOIRE :
<raisonnement>
[Analyse doc par doc, extraction des faits clés]
</raisonnement>
<réponse>
[Réponse finale concise et précise]
</réponse>
"""


def execute_search_tool(collection, query: str) -> Tuple[str, List[str], List[str]]:
    """Exécute la recherche hybride (BM25 + vecteur) et retourne le contexte, les sources et les chunks."""
    try:
        if collection.count() == 0:
            return "Observation: La base de données est vide.", [], []

        # 1. Recherche hybride (BM25 + vectorielle via RRF)
        retrieved_docs, retrieved_metas = hybrid_search(
            collection, query, _agent_bm25, DEFAULT_RETRIEVAL_COUNT
        )

        if not retrieved_docs:
            # Fallback sur recherche vectorielle seule
            retrieved_docs, retrieved_metas = vector_rerank_search(collection, query, DEFAULT_RETRIEVAL_COUNT)

        if not retrieved_docs:
            return "Observation: Aucun document trouvé pour cette recherche.", [], []

        # 2. Reranking cross-encoder
        reranked = rerank(query, retrieved_docs, top_k=FINAL_COUNT)
        # Filtrer uniquement les chunks avec un score positif (pertinence réelle)
        best_pairs = [(retrieved_docs[i], retrieved_metas[i], score) for i, score in reranked if score > -5.0]
        if not best_pairs:
            best_pairs = [(retrieved_docs[i], retrieved_metas[i], score) for i, score in reranked]

        best_docs  = [doc  for doc, _,   _     in best_pairs]
        best_metas = [meta for _,   meta, _     in best_pairs]

        context = "\n---\n".join(best_docs)
        sources = [m.get("source", "unknown") for m in best_metas]

        return f"Observation: J'ai trouvé les extraits suivants dans la documentation:\n{context}", sources, best_docs

    except Exception as e:
        return f"Observation: Erreur lors de la recherche ({str(e)}). Essaie de répondre sans cet outil.", [], []


def _pick_distractors(collection, oracle_docs: List[str], n: int) -> Tuple[List[str], List[str]]:
    """Sélectionne n distracteurs depuis la collection (docs non oracle)."""
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])
        pool = [(d, m) for d, m in zip(all_docs, all_metas) if d not in oracle_docs]
        selected = random.sample(pool, min(n, len(pool))) if pool else []
        dist_texts = [d for d, _ in selected]
        dist_sources = [m.get("source", "unknown") for _, m in selected]
        return dist_texts, dist_sources
    except Exception:
        return [], []


def _build_raft_context(oracle_docs: List[str], distractor_docs: List[str]) -> str:
    """Mélange oracles et distracteurs sans étiquettes, numérotés."""
    all_docs = oracle_docs[:AGENT_RAFT_MAX_ORACLE] + distractor_docs
    random.shuffle(all_docs)
    section = ""
    for i, doc in enumerate(all_docs):
        preview = doc[:500] + "..." if len(doc) > 500 else doc
        section += f"\n--- Document {i+1} ---\n{preview}\n"
    return section, all_docs


def _parse_cot_response(raw: str) -> Tuple[str, str]:
    """Extrait le raisonnement et la réponse finale du format CoT RAFT."""
    reasoning_m = re.search(r'<raisonnement>(.*?)</raisonnement>', raw, re.DOTALL | re.IGNORECASE)
    answer_m = re.search(r'<r[ée]ponse>(.*?)</r[ée]ponse>', raw, re.DOTALL | re.IGNORECASE)

    reasoning = reasoning_m.group(1).strip() if reasoning_m else raw.strip()
    answer = answer_m.group(1).strip() if answer_m else raw.strip()

    # Nettoyer les balises résiduelles
    answer = re.sub(r'<[^>]+>', '', answer).strip()
    return reasoning, answer


def run_react_agent(collection, question: str, max_iterations: int = 6) -> Dict[str, Any]:
    """
    Exécute la boucle de raisonnement ReAct enrichie avec RAFT.

    Phase 1 : boucle ReAct classique (collect oracle_docs via search_documents)
    Phase 2 : injection de distracteurs + prompt CoT RAFT → réponse structurée
    """
    config = MODELS_CONFIG.get("mistral")
    if not config:
        config = MODELS_CONFIG.get("qwen")

    provider = config["provider"]
    model_id = config["model_id"]

    prompt_contexte = AGENT_SYSTEM_PROMPT + f"\n\nQuestion (User): {question}\n"

    t_start = time.perf_counter()
    total_in_tokens = 0
    total_out_tokens = 0
    tools_used = 0
    all_sources: List[str] = []
    oracle_docs: List[str] = []
    agent_trace = []

    # ── Phase 1 : boucle ReAct ──────────────────────────────────
    for idx in range(max_iterations):
        response = call_model(
            provider=provider,
            model_id=model_id,
            system_prompt="Suis strictement les instructions du format Thought/Action ou Thought/Final Answer.",
            question=prompt_contexte,
            max_tokens=1000,
            temperature=0.2,
        )

        if response["error"]:
            return {"error": f"Erreur du modèle IA: {response['error']}", "trace": agent_trace}

        total_in_tokens += response["input_tokens"]
        total_out_tokens += response["output_tokens"]

        generation = response["text"].strip()
        agent_trace.append({"role": "agent", "content": generation})
        prompt_contexte += f"\n{generation}\n"

        # Le modèle veut conclure → on intercepte et on fait RAFT à la place
        final_answer_match = re.search(
            r"(?:Final Answer|Réponse finale|Réponse Finale|Réponse|Final answer)\s*:\s*(.*)",
            generation, re.DOTALL | re.IGNORECASE
        )
        if final_answer_match:
            # On a collecté des oracles → on passe en Phase 2
            agent_trace.append({"role": "system", "content": "[RAFT] Phase ReAct terminée → injection des distracteurs"})
            break

        # Le modèle veut chercher
        action_match = re.search(r"Action:\s*(.*)", generation, re.IGNORECASE)
        action_input_match = re.search(r"Action Input:\s*(.*)", generation, re.IGNORECASE)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()

            if "search_documents" in action.lower():
                tools_used += 1
                observation, sources, chunks = execute_search_tool(collection, action_input)
                all_sources.extend(sources)
                oracle_docs.extend(chunks)

                agent_trace.append({"role": "tool (search_documents)", "content": observation})
                prompt_contexte += f"\n{observation}\n"
            else:
                observation = f"Observation: Outil inconnu '{action}'. Seul 'search_documents' est disponible."
                agent_trace.append({"role": "tool error", "content": observation})
                prompt_contexte += f"\n{observation}\n"
        else:
            observation = "Observation: Format invalide. Utilise 'Action:'/'Action Input:' ou 'Final Answer:'."
            agent_trace.append({"role": "system", "content": observation})
            prompt_contexte += f"\n{observation}\n"

    # ── Phase 2 : RAFT CoT ──────────────────────────────────────
    # Sélectionner des distracteurs
    distractor_texts, distractor_sources = _pick_distractors(
        collection, oracle_docs, AGENT_RAFT_NUM_DISTRACTORS
    )

    # Construire le contexte mixte
    doc_section, all_mixed = _build_raft_context(oracle_docs, distractor_texts)

    raft_prompt = (
        f"Question : {question}\n\n"
        f"Voici {len(all_mixed)} documents récupérés par l'agent "
        f"(certains sont pertinents, d'autres sont des distracteurs) :\n"
        f"{doc_section}"
    )

    agent_trace.append({"role": "RAFT CoT", "content": f"Envoi de {len(all_mixed)} docs au LLM pour raisonnement CoT"})

    raft_response = call_model(
        provider=provider,
        model_id=model_id,
        system_prompt=RAFT_COT_PROMPT,
        question=raft_prompt,
        max_tokens=2000,
        temperature=0.2,
    )

    total_latency = round(time.perf_counter() - t_start, 3)
    total_in_tokens += raft_response.get("input_tokens", 0)
    total_out_tokens += raft_response.get("output_tokens", 0)

    if raft_response.get("error"):
        return {"error": f"Erreur RAFT CoT: {raft_response['error']}", "trace": agent_trace}

    raw_cot = raft_response.get("text", "")
    reasoning, final_answer = _parse_cot_response(raw_cot)

    agent_trace.append({"role": "RAFT CoT", "content": f"Raisonnement extrait ({len(reasoning)} chars)"})

    # ── Stats RAFT ──────────────────────────────────────────────
    raft_stats = {
        "num_oracle_docs":      len(oracle_docs[:AGENT_RAFT_MAX_ORACLE]),
        "num_distractor_docs":  len(distractor_texts),
        "total_docs_presented": len(all_mixed),
        "oracle_ratio":         round(len(oracle_docs[:AGENT_RAFT_MAX_ORACLE]) / max(len(all_mixed), 1), 2),
        "has_reasoning":        bool(reasoning),
        "has_structured_response": bool(re.search(r'<r[ée]ponse>', raw_cot, re.IGNORECASE)),
        "react_iterations":     idx + 1,
        "tools_used":           tools_used,
    }

    return {
        "question":       question,
        "response":       final_answer,
        "reasoning":      reasoning,
        "raw_cot":        raw_cot,
        "model_used":     config["display_name"],
        "tools_used_count": tools_used,
        "sources":        list(set(all_sources)),
        "oracle_docs": [
            {"text": d[:300] + "..." if len(d) > 300 else d, "source": s}
            for d, s in zip(oracle_docs[:AGENT_RAFT_MAX_ORACLE], all_sources[:AGENT_RAFT_MAX_ORACLE])
        ],
        "distractor_docs": [
            {"text": d[:200] + "..." if len(d) > 200 else d, "source": s}
            for d, s in zip(distractor_texts, distractor_sources)
        ],
        "raft_stats":     raft_stats,
        "traces":         agent_trace,
        "latency_seconds": total_latency,
        "input_tokens":   total_in_tokens,
        "output_tokens":  total_out_tokens,
    }
