import json
import re
from typing import Dict, Any, Tuple, List
from benchmark.model_clients import call_model
from benchmark.models_config import MODELS_CONFIG
from utils.rag_optimized_functions import vector_rerank_search, rerank, FINAL_COUNT, DEFAULT_RETRIEVAL_COUNT

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
3. INTERDICTION DE TOURNER EN ROND : Tu as le droit d'utiliser l'outil `search_documents` au MAXIMUM 2 fois. Si après 2 requêtes tu n'as pas trouvé la réponse dans la base de données, TU DOIS t'arrêter et générer une `Final Answer` expliquant honnêtement que le document interne ne contient pas l'information demandée.
"""

def execute_search_tool(collection, query: str) -> Tuple[str, List[str]]:
    """Exécute l'outil de RAG optimisé et retourne le contexte en texte et les sources."""
    try:
        if collection.count() == 0:
            return "Observation: La base de données est vide.", []

        # 1. Recherche vectorielle
        retrieved_docs, retrieved_metas = vector_rerank_search(collection, query, DEFAULT_RETRIEVAL_COUNT)
        
        if not retrieved_docs:
            return "Observation: Aucun document trouvé pour cette recherche.", [], []

        # 2. Reranking
        reranked = rerank(query, retrieved_docs, top_k=FINAL_COUNT)
        best_docs = [retrieved_docs[i] for i, _ in reranked]
        best_metas = [retrieved_metas[i] for i, _ in reranked]

        context = "\n---\n".join(best_docs)
        sources = [m.get("source", "unknown") for m in best_metas]
        
        return f"Observation: J'ai trouvé les extraits suivants dans la documentation:\n{context}", sources, best_docs
        
    except Exception as e:
        return f"Observation: Erreur lors de la recherche ({str(e)}). Essaie de répondre sans cet outil.", [], []


def run_react_agent(collection, question: str, max_iterations: int = 6) -> Dict[str, Any]:
    """
    Exécute la boucle de raisonnement ReAct.
    """
    config = MODELS_CONFIG.get("mistral")
    if not config:
        config = MODELS_CONFIG.get("qwen") # Fallback
        
    provider = config["provider"]
    model_id = config["model_id"]
    temperature = 0.2

    # L'historique complet de la "conversation" de raisonnement
    prompt_contexte = AGENT_SYSTEM_PROMPT + f"\n\nQuestion (User): {question}\n"
    
    total_latency = 0.0
    total_in_tokens = 0
    total_out_tokens = 0
    tools_used = 0
    all_sources = []
    all_chunks = []
    agent_trace = []

    for idx in range(max_iterations):
        # On demande au modèle la prochaine étape
        response = call_model(
            provider=provider,
            model_id=model_id,
            system_prompt="Suis strictement les instructions du format Thought/Action ou Thought/Final Answer.",
            question=prompt_contexte,
            max_tokens=1000,
            temperature=0.2,
        )

        if response["error"]:
            return {
                "error": f"Erreur du modèle IA: {response['error']}",
                "trace": agent_trace
            }
            
        total_latency += response["latency_seconds"]
        total_in_tokens += response["input_tokens"]
        total_out_tokens += response["output_tokens"]
        
        generation = response["text"].strip()
        agent_trace.append({"role": "agent", "content": generation})
        prompt_contexte += f"\n{generation}\n"

        # Tenter d'extraire Final Answer (ou Réponse finale si le modèle a traduit)
        final_answer_match = re.search(r"(?:Final Answer|Réponse finale|Réponse Finale|Réponse|Final answer)\s*:\s*(.*)", generation, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            return {
                "response": final_answer_match.group(1).strip(),
                "model_used": config["display_name"],
                "tools_used_count": tools_used,
                "sources": list(set(all_sources)),
                "retrieved_chunks": all_chunks,
                "iterations": idx + 1,
                "traces": agent_trace,
                "latency_seconds": total_latency,
                "input_tokens": total_in_tokens,
                "output_tokens": total_out_tokens
            }

        # Tenter d'extraire une Action
        action_match = re.search(r"Action:\s*(.*)", generation, re.IGNORECASE)
        action_input_match = re.search(r"Action Input:\s*(.*)", generation, re.IGNORECASE)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            if "search_documents" in action.lower():
                tools_used += 1
                observation, sources, chunks = execute_search_tool(collection, action_input)
                all_sources.extend(sources)
                all_chunks.extend(chunks)
                
                agent_trace.append({"role": "tool (search_documents)", "content": observation})
                prompt_contexte += f"\n{observation}\n"
            else:
                observation = f"Observation: Outil inconnu '{action}'. Seul l'outil 'search_documents' est disponible."
                agent_trace.append({"role": "tool error", "content": observation})
                prompt_contexte += f"\n{observation}\n"
        else:
            # Le modèle a mal respecté le format, on l'oblige à corriger
            observation = "Observation: Format invalide. Tu dois utiliser soit 'Action:' et 'Action Input:', soit 'Final Answer:'."
            agent_trace.append({"role": "system", "content": observation})
            prompt_contexte += f"\n{observation}\n"

    # Si on dépasse max_iterations sans Final Answer
    return {
        "response": "Je suis désolé, je n'ai pas pu conclure ma réflexion à temps pour répondre à cette question.",
        "model_used": config["display_name"],
        "tools_used_count": tools_used,
        "sources": list(set(all_sources)),
        "retrieved_chunks": all_chunks,
        "iterations": max_iterations,
        "traces": agent_trace,
        "latency_seconds": total_latency,
        "input_tokens": total_in_tokens,
        "output_tokens": total_out_tokens
    }
