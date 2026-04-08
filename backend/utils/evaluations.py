import numpy as np
from sentence_transformers import SentenceTransformer, util  

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def cosine_similarity_between(text1, text2):
    emb = get_embedder()
    emb1 = emb.encode(text1, convert_to_tensor=True)
    emb2 = emb.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def retrieval_precision(retrieved_chunks, reference_answer, threshold=0.5):
    """Proportion de chunks dont la similarité avec la réponse de référence dépasse le seuil."""
    if not retrieved_chunks:
        return 0.0
    ref_emb = get_embedder().encode(reference_answer)
    scores = []
    for chunk in retrieved_chunks:
        chunk_text = chunk if isinstance(chunk, str) else chunk.get("text", "")
        if not chunk_text:
            continue
        chunk_emb = get_embedder().encode(chunk_text)
        sim = util.cos_sim(ref_emb, chunk_emb).item()
        scores.append(sim)
    relevant = sum(1 for s in scores if s > threshold)
    return relevant / len(scores)

def retrieval_recall_at_k(retrieved_chunks, reference_answer, k=5, threshold=0.5):
    """Recall@K : proportion de chunks pertinents parmi les top K."""
    top_chunks = retrieved_chunks[:k]
    return retrieval_precision(top_chunks, reference_answer, threshold)

import requests
import time
from typing import List, Dict, Any

def run_evaluations(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Exécute l'évaluation sur l'échantillon de questions pour l'ensemble des modèles API."""
    processes = [
        {"name": "LLM simple", "endpoint": "/llm-simple", "has_retrieval": False},
        {"name": "RAG", "endpoint": "/rag-simple", "has_retrieval": True},
        {"name": "RAG optimisé", "endpoint": "/rag-optimized", "has_retrieval": True},
        {"name": "RAG Agent", "endpoint": "/rag-agent", "has_retrieval": True},
        {"name": "RAG + Multi-agents", "endpoint": "/rag-multi-agent", "has_retrieval": True},
    ]

    base_url = "http://localhost:5001"
    results = {}

    for proc in processes:
        eval_details = []
        for item in questions:
            q = item.get("question", "")
            ref_answer = item.get("reponse", "")

            try:
                start = time.perf_counter()
                resp = requests.post(f"{base_url}/api{proc['endpoint']}", json={"question": q}, timeout=90)
                latency = time.perf_counter() - start
                if resp.status_code != 200:
                    raise Exception(resp.json().get("error", "Erreur API"))
                data = resp.json()

                generated = data.get("response", "")
                # Qualité : similarité entre réponse générée et référence
                quality = cosine_similarity_between(generated, ref_answer)

                # Fidélité : similarité entre génération et contexte (si retrieval)
                faithfulness = 0.0
                precision = recall = 0.0
                if proc["has_retrieval"]:
                    # Récupérer les chunks (selon le format renvoyé)
                    if proc["name"] == "RAG":
                        chunks = data.get("context_chunks", [])
                        retrieved_texts = [c.get("text", "") for c in chunks if c.get("text")]
                    elif proc["name"] == "RAG optimisé":
                        retrieved_texts = data.get("retrieved_chunks", [])
                        if retrieved_texts and isinstance(retrieved_texts[0], dict):
                            retrieved_texts = [c.get("text", "") for c in retrieved_texts]
                    elif proc["name"] == "RAG Agent":
                        retrieved_texts = data.get("retrieved_chunks", [])
                        if retrieved_texts and isinstance(retrieved_texts[0], dict):
                            retrieved_texts = [c.get("text", "") for c in retrieved_texts]
                    elif proc["name"] == "RAG + Multi-agents":
                        retrieved_texts = data.get("retrieved_chunks", [])
                        if retrieved_texts and isinstance(retrieved_texts[0], dict):
                            retrieved_texts = [c.get("text", "") for c in retrieved_texts]
                    else:
                        retrieved_texts = []

                    if retrieved_texts:
                        # Fidélité : similarité max entre génération et chaque chunk
                        gen_emb = get_embedder().encode(generated)
                        chunk_embs = get_embedder().encode(retrieved_texts)
                        sims = util.cos_sim(gen_emb, chunk_embs)[0].tolist()
                        faithfulness = max(sims) 

                        # Métriques de retrieval
                        precision = retrieval_precision(retrieved_texts, ref_answer)
                        recall = retrieval_recall_at_k(retrieved_texts, ref_answer, k=5)
                else:
                    # Pour LLM simple, fidélité = qualité (par défaut)
                    faithfulness = quality

                eval_details.append({
                    "question": q[:100],
                    "quality": round(quality, 4),
                    "faithfulness": round(faithfulness, 4),
                    "retrieval_precision": round(precision, 4),
                    "retrieval_recall": round(recall, 4),
                    "latency_seconds": round(latency, 3)
                })
            except Exception as e:
                eval_details.append({
                    "question": q[:100],
                    "error": str(e)
                })

        # Calcul des moyennes
        valid = [d for d in eval_details if "quality" in d]
        if valid:
            avg_quality = np.mean([d["quality"] for d in valid]) * 100
            avg_faith = np.mean([d["faithfulness"] for d in valid]) * 100
            avg_prec = np.mean([d["retrieval_precision"] for d in valid]) * 100
            avg_recall = np.mean([d["retrieval_recall"] for d in valid]) * 100
            avg_lat = np.mean([d["latency_seconds"] for d in valid])    
        else:
            avg_quality = avg_faith = avg_prec = avg_recall = avg_lat = 0.0

        results[proc["name"]] = {
            "average": {
                "quality": round(avg_quality, 2), 
                "faithfulness": round(avg_faith, 2), 
                "retrieval_precision": round(avg_prec, 2), 
                "retrieval_recall_at_5": round(avg_recall, 2), 
                "latency_seconds": round(avg_lat, 3) 
            },
            "details": eval_details
        }

    return results