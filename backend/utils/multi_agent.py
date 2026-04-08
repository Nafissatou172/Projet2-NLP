import json
import re
from typing import Dict, Any, Tuple, List
from benchmark.model_clients import call_model
from benchmark.models_config import MODELS_CONFIG, SYSTEM_PROMPT
from utils.rag_optimized_functions import vector_rerank_search, rerank, FINAL_COUNT, DEFAULT_RETRIEVAL_COUNT

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

def run_multi_agent_rag(collection, question: str, max_search_loops: int = 3) -> Dict[str, Any]:
    # Récupérer la config du modèle
    config = MODELS_CONFIG.get("mistral")
    if not config:
        config = MODELS_CONFIG.get("qwen") # Fallback
        
    provider = config["provider"]
    model_id = config["model_id"]
    temperature = 0.2

    total_latency = 0.0
    total_in_tokens = 0
    total_out_tokens = 0
    all_chunks = []
    all_sources = set()
    agent_trace = []
    
    current_search_query = question

    # Boucle Searcher -> Critic
    loop_count = 0
    context_passed = False
    best_docs = []

    while loop_count < max_search_loops:
        loop_count += 1
        
        # 1. Searcher Agent
        agent_trace.append({"role": "Searcher", "content": f"Formulation de la requête à partir de : '{current_search_query}'"})
        searcher_res = call_model(
            provider=provider, model_id=model_id,
            system_prompt=SEARCHER_PROMPT,
            question=f"Question : {current_search_query}",
            max_tokens=50, temperature=0.1
        )
        if searcher_res["error"]:
            return {"error": f"Erreur Searcher IA: {searcher_res['error']}"}
        
        total_latency += searcher_res["latency_seconds"]
        total_in_tokens += searcher_res["input_tokens"]
        total_out_tokens += searcher_res["output_tokens"]
        
        search_query = searcher_res["text"].strip().replace('"', '')
        agent_trace.append({"role": "Searcher", "content": f"Mots-clés extraits : '{search_query}'"})

        # Exécution de la recherche
        best_docs, best_metas = execute_search(collection, search_query)
        for d in best_docs:
            all_chunks.append(d)
        for m in best_metas:
            all_sources.add(m.get("source", "unknown"))

        if not best_docs:
            agent_trace.append({"role": "Searcher", "content": "Aucun document trouvé pour ces mots-clés."})
            context_text = "Base de données vide ou aucun résultat."
        else:
            context_text = "\n---\n".join(best_docs)
            agent_trace.append({"role": "Searcher", "content": f"J'ai récupéré {len(best_docs)} extraits de documents."})

        # 2. Critic Agent
        critic_input = f"QUESTION: {question}\n\nCONTEXTE RECUPERE:\n{context_text}"
        critic_res = call_model(
            provider=provider, model_id=model_id,
            system_prompt=CRITIC_PROMPT,
            question=critic_input,
            max_tokens=100, temperature=0.1
        )
        if critic_res["error"]:
            return {"error": f"Erreur Critic IA: {critic_res['error']}"}

        total_latency += critic_res["latency_seconds"]
        total_in_tokens += critic_res["input_tokens"]
        total_out_tokens += critic_res["output_tokens"]
        
        critic_response = critic_res["text"].strip()
        
        if critic_response.upper().startswith("PASS"):
            agent_trace.append({"role": "Critic", "content": "Évaluation : PASS. Le contexte est pertinent !"})
            context_passed = True
            break
        else:
            agent_trace.append({"role": "Critic", "content": f"Évaluation : {critic_response}"})
            # Extraire les nouveaux mots-clés suggérés
            if "FAIL" in critic_response.upper() and ":" in critic_response:
                current_search_query = critic_response.split(":", 1)[1].strip()
            else:
                current_search_query = question # Retour aux requêtes de base
    
    # 3. Generator Agent
    if best_docs:
        final_context = "\n\n---\n\n".join(best_docs)
        generator_context = f"Voici des extraits validés par le Critic pour répondre :\n{final_context}\n\n---\n\n"
    else:
        generator_context = "Aucun document pertinent trouvé même après plusieurs recherches. Veuillez indiquer que l'information n'est pas disponible.\n\n"

    system_prompt_with_context = SYSTEM_PROMPT + f"\n\n{generator_context}"
    
    agent_trace.append({"role": "Generator", "content": "Je rédige la réponse finale basée sur le contexte validé."})
    
    generator_res = call_model(
        provider=provider, model_id=model_id,
        system_prompt=system_prompt_with_context,
        question=question,
        max_tokens=1000, temperature=0.2
    )
    
    if generator_res["error"]:
        return {"error": f"Erreur Generator IA: {generator_res['error']}"}

    total_latency += generator_res["latency_seconds"]
    total_in_tokens += generator_res["input_tokens"]
    total_out_tokens += generator_res["output_tokens"]
    
    # Nettoyage de la réponse
    cleaned_text = generator_res["text"]
    
    return {
        "question": question,
        "response": cleaned_text,
        "model_used": config["display_name"],
        "loops_count": loop_count,
        "context_passed": context_passed,
        "sources": list(all_sources),
        "retrieved_chunks": all_chunks,
        "traces": agent_trace,
        "latency_seconds": total_latency,
        "input_tokens": total_in_tokens,
        "output_tokens": total_out_tokens
    }
