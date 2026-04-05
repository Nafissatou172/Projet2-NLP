import os
import csv
import io
import random
import time
from datetime import datetime, timezone
from typing import Optional

from flask import Blueprint, jsonify, request

from pathlib import Path
import pdfplumber
from docx import Document
import chromadb
from chromadb.utils import embedding_functions
import hashlib

# ─────────────────────────────────────────────
#  import pour le RAG
# ─────────────────────────────────────────────
import chromadb
from utils.rag_simple_functions import init_vectorstore, TOP_K

# ─────────────────────────────────────────────
#  import pour l'évaluation
# ─────────────────────────────────────────────
import requests
import time
import numpy as np

# Chemins
DOCUMENTS_DIR = Path(__file__).parent / "documents" # chemin vers les documents du RAG
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db" # chemin vers la base de données vectorielle du RAG

from benchmark.models_config import MODELS_CONFIG, DATASET_PATH, DEFAULT_SAMPLE_SIZE, SYSTEM_PROMPT
from benchmark.model_clients import call_model
from benchmark.evaluators import (
    evaluate_quality,
    evaluate_faithfulness,
    evaluate_latency,
    evaluate_cost,
    compute_global_score,
)

# ─────────────────────────────────────────────
#  Blueprint
# ─────────────────────────────────────────────
# RAG configuration
DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
CHUNK_SIZE = 1000  # caractères
CHUNK_OVERLAP = 200
TOP_K = 5

# ─────────────────────────────────────────────
#  Blueprint
# ─────────────────────────────────────────────

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Cache en mémoire du dernier run du benchmark
_last_result: Optional[dict] = None


# ─────────────────────────────────────────────
#  Routes de base (Index & Health)
# ─────────────────────────────────────────────

@api_bp.route('/index')
def index():
    return jsonify({
        'message': 'Bienvenue sur l\'API FinChat-SN',
        'status': 'ok'
    })


@api_bp.route('/health')
def health():
    return jsonify({'status': 'healthy'})


# ─────────────────────────────────────────────
#  Logique Benchmark : Chargement du dataset
# ─────────────────────────────────────────────

def _load_dataset(
    sample_size: Optional[int] = None,
    categories: Optional[list] = None,
    difficulty: Optional[str] = None,
) -> list[dict]:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")

    import pandas as pd

    # Lecture robuste
    df = pd.read_csv(DATASET_PATH, encoding="utf-8", on_bad_lines='skip')
    
    # Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Garder seulement les colonnes utiles
    cols = ['question', 'reponse', 'categorie', 'niveau_difficulte']
    df = df[[c for c in cols if c in df.columns]]
    
    # Supprimer les lignes avec question ou reponse vide
    df = df.dropna(subset=['question', 'reponse'])
    
    # Nettoyer le texte
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip('"')
    
    # Filtres
    if categories:
        df = df[df['categorie'].isin(categories)]
    if difficulty:
        df = df[df['niveau_difficulte'].str.lower() == difficulty.lower()]
    
    # Échantillonnage
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    return df.to_dict(orient='records')


# ─────────────────────────────────────────────
#  Logique Benchmark : Exécution
# ─────────────────────────────────────────────

def _run_model_benchmark(model_key: str, questions: list[dict]) -> dict:
    """Évalue un modèle sur toutes les questions et agrège les métriques."""
    config = MODELS_CONFIG[model_key]

    results_per_question = []
    total_quality = 0.0
    total_faithfulness = 0.0
    total_latency = 0.0
    total_cost_usd = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    errors = 0

    for item in questions:
        question = item["question"]
        reference = item["reponse"]

        # Appel au modèle
        response = call_model(
            provider=config["provider"],
            model_id=config["model_id"],
            system_prompt=SYSTEM_PROMPT,
            question=question,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )

        if response["error"]:
            errors += 1
            results_per_question.append({
                "question": question,
                "reference": reference,
                "generated": "",
                "error": response["error"],
                "metrics": None,
            })
            continue

        generated = response["text"]
        latency_s = response["latency_seconds"]
        in_tok = response["input_tokens"]
        out_tok = response["output_tokens"]

        # Évaluation des 4 piliers
        quality = evaluate_quality(generated, reference)
        faithfulness = evaluate_faithfulness(generated, reference)
        latency_info = evaluate_latency(latency_s)
        cost_info = evaluate_cost(
            in_tok, out_tok,
            config["cost_per_1k_input_tokens"],
            config["cost_per_1k_output_tokens"],
        )
        global_score = compute_global_score(
            quality, faithfulness, latency_info["score"], cost_info["score"]
        )

        total_quality += quality
        total_faithfulness += faithfulness
        total_latency += latency_s
        total_cost_usd += cost_info["cost_usd"]
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        results_per_question.append({
            "question": question,
            "reference": reference,
            "generated": generated,
            "category": item.get("categorie", ""),
            "difficulty": item.get("niveau_difficulte", ""),
            "error": None,
            "metrics": {
                "quality": quality,
                "faithfulness": faithfulness,
                "latency": latency_info,
                "cost": cost_info,
                "global_score": global_score,
            },
        })

    answered = len(questions) - errors
    safe_div = lambda x: round(x / answered, 4) if answered > 0 else 0.0

    avg_latency = safe_div(total_latency)
    latency_info_avg = evaluate_latency(avg_latency)
    cost_info_avg = evaluate_cost(
        total_input_tokens, total_output_tokens,
        config["cost_per_1k_input_tokens"],
        config["cost_per_1k_output_tokens"],
    )

    avg_quality = safe_div(total_quality)
    avg_faithfulness = safe_div(total_faithfulness)

    return {
        "model_key": model_key,
        "display_name": config["display_name"],
        "provider": config["provider"],
        "model_id": config["model_id"],
        "questions_total": len(questions),
        "questions_answered": answered,
        "errors": errors,
        "aggregated": {
            "quality": avg_quality,
            "faithfulness": avg_faithfulness,
            "latency_avg_seconds": avg_latency,
            "latency_label": latency_info_avg["label"],
            "latency_score": latency_info_avg["score"],
            "total_cost_usd": round(total_cost_usd, 6),
            "cost_score": cost_info_avg["score"],
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "global_score": compute_global_score(
                avg_quality,
                avg_faithfulness,
                latency_info_avg["score"],
                cost_info_avg["score"],
            ),
        },
        "details": results_per_question,
    }


# ─────────────────────────────────────────────
#  Routes Benchmark
# ─────────────────────────────────────────────

@api_bp.route("/benchmark", methods=["POST"])
def run_benchmark():
    """Lance le benchmark sur les modèles demandés."""
    global _last_result

    body = request.get_json(silent=True) or {}
    model_keys = body.get("models", list(MODELS_CONFIG.keys()))
    sample_size = body.get("sample_size", DEFAULT_SAMPLE_SIZE)
    categories = body.get("categories", None)
    difficulty = body.get("difficulty", None)

    # Validation des modèles demandés
    invalid = [m for m in model_keys if m not in MODELS_CONFIG]
    if invalid:
        return jsonify({
            "error": f"Modèles inconnus : {invalid}. Disponibles : {list(MODELS_CONFIG.keys())}"
        }), 400

    # Chargement du dataset
    try:
        questions = _load_dataset(sample_size, categories, difficulty)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    if not questions:
        return jsonify({"error": "Aucune question trouvée avec les filtres appliqués."}), 400

    # Run benchmark pour chaque modèle
    started_at = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    model_results = []
    for key in model_keys:
        result = _run_model_benchmark(key, questions)
        model_results.append(result)

    elapsed = round(time.perf_counter() - t_start, 2)

    # Classement par score global
    ranked = sorted(
        model_results,
        key=lambda r: r["aggregated"]["global_score"],
        reverse=True,
    )
    for rank, r in enumerate(ranked, 1):
        r["rank"] = rank

    _last_result = {
        "started_at": started_at,
        "duration_seconds": elapsed,
        "sample_size": len(questions),
        "filters": {
            "categories": categories,
            "difficulty": difficulty,
        },
        "models_evaluated": model_keys,
        "results": ranked,
    }

    return jsonify(_last_result), 200


@api_bp.route("/get-benchmark", methods=["GET"])
def get_last_benchmark():
    """Retourne le dernier résultat de benchmark en cache."""
    if _last_result is None:
        return jsonify({
            "message": "Aucun benchmark exécuté. Lance un POST /api/benchmark d'abord."
        }), 404
    return jsonify(_last_result), 200


@api_bp.route("/benchmark/models", methods=["GET"])
def list_models():
    """Liste les modèles disponibles avec leur configuration."""
    models_info = {}
    for key, cfg in MODELS_CONFIG.items():
        models_info[key] = {
            "display_name": cfg["display_name"],
            "provider": cfg["provider"],
            "model_id": cfg["model_id"],
            "cost_per_1k_input_tokens": cfg["cost_per_1k_input_tokens"],
            "cost_per_1k_output_tokens": cfg["cost_per_1k_output_tokens"],
        }
    return jsonify({"models": models_info}), 200


@api_bp.route("/benchmark/dataset-info", methods=["GET"])
def dataset_info():
    try:
        rows = _load_dataset()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    categories = {}
    difficulties = {}
    for r in rows:
        cat = r.get("categorie", "Inconnue")
        if cat and isinstance(cat, str):
            cat = cat.strip()
        else:
            cat = "Inconnue"
        diff = r.get("niveau_difficulte", "Inconnu")
        if diff and isinstance(diff, str):
            diff = diff.strip()
        else:
            diff = "Inconnu"
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    # Supprimer les clés vides ou NaN (mais normalement déjà géré)
    categories = {k: v for k, v in categories.items() if k and k != "nan"}
    difficulties = {k: v for k, v in difficulties.items() if k and k != "nan"}

    return jsonify({
        "total_questions": len(rows),
        "categories": categories,
        "difficulties": difficulties,
        "dataset_path": DATASET_PATH,
    }), 200


# ─────────────────────────────────────────────
#  Endpoint LLM simple (meilleur modèle du dernier benchmark)
# ─────────────────────────────────────────────

import re

def clean_response(text: str) -> str:
    """
    Supprime tous les caractères de formatage Markdown des réponses.
    """
    # Supprimer les astérisques doubles et simples (gras/italique)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **gras** -> gras
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italique* -> italique
    
    # Supprimer les backticks (code)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Supprimer les titres Markdown (#, ##, etc.)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Supprimer les lignes de séparation (---, ***, ___)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Supprimer les listes avec astérisques ou tirets en début de ligne
    # Remplacer par un simple tiret ou bullet point propre
    text = re.sub(r'^[\s]*[-*+]\s+', '  • ', text, flags=re.MULTILINE)
    
    # Supprimer les astérisques isolés restants
    text = text.replace('*', '')
    
    # Supprimer les doubles espaces et sauts de ligne multiples
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

@api_bp.route("/llm-simple", methods=["POST"])
def llm_simple():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "La requête doit contenir un champ 'question'"}), 400

    question = body["question"].strip()
    if not question:
        return jsonify({"error": "La question ne peut pas être vide"}), 400

    # Liste ordonnée des modèles à essayer (priorité : meilleur benchmark, puis fallback)
    candidates = []
    if _last_result and _last_result.get("results"):
        best_key = _last_result["results"][0]["model_key"]
        candidates.append(best_key)
    # Ajouter des fallbacks (modèles qui fonctionnent)
    for fallback in ["qwen", "mistral", "llama"]:
        if fallback not in candidates:
            candidates.append(fallback)

    last_error = None
    for model_key in candidates:
        config = MODELS_CONFIG.get(model_key)
        if not config:
            continue
        try:
            response = call_model(
                provider=config["provider"],
                model_id=config["model_id"],
                system_prompt=SYSTEM_PROMPT,
                question=question,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
            if not response["error"]:
                # Nettoyer la réponse avant de l'envoyer
                cleaned_text = clean_response(response["text"])
                return jsonify({
                    "question": question,
                    "response": cleaned_text,
                    "model_used": config["display_name"],
                    "model_key": model_key,
                    "provider": config["provider"],
                    "latency_seconds": response["latency_seconds"],
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                }), 200
            else:
                last_error = response["error"]
        except Exception as e:
            last_error = str(e)
            continue

    return jsonify({"error": f"Tous les modèles ont échoué. Dernière erreur : {last_error}"}), 500



# ─────────────────────────────────────────────
#  Endpoint du RAG simple (meilleur modèle du dernier benchmark)
# ─────────────────────────────────────────────

# Variable globale pour la collection
_rag_collection = None

def get_rag_collection():
    global _rag_collection
    if _rag_collection is None:
        _rag_collection = init_vectorstore(DOCUMENTS_DIR, CHROMA_DB_DIR)
    return _rag_collection


@api_bp.route("/rag-simple", methods=["POST"])
def rag_simple():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "La requête doit contenir un champ 'question'"}), 400
    
    question = body["question"].strip()
    if not question:
        return jsonify({"error": "La question ne peut pas être vide"}), 400
    
    collection = get_rag_collection()
    
    try:
        results = collection.query(
            query_texts=[question],
            n_results=TOP_K,
        )
    except Exception as e:
        return jsonify({"error": f"Erreur de recherche vectorielle : {str(e)}"}), 500
    
    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    if chunks:
        context = "\n\n---\n\n".join(chunks)
        context = f"Documents pertinents :\n{context}\n\n---\n\n"
    else:
        context = "Aucun document pertinent trouvé. Répondez de votre mieux avec vos connaissances générales.\n\n"
    
    system_prompt_with_context = SYSTEM_PROMPT + f"\n\nVoici des extraits de documents financiers à utiliser pour répondre :\n{context}"
    
    # Sélection du meilleur modèle (ou Qwen par défaut)
    # Utiliser Mistral directement (ou un modèle par défaut fiable)
    config = MODELS_CONFIG.get("mistral")
    if not config:
        config = MODELS_CONFIG.get("qwen")  # fallback  
    
    try:
        response = call_model(
            provider=config["provider"],
            model_id=config["model_id"],
            system_prompt=system_prompt_with_context,
            question=question,
            max_tokens=1500,
            temperature=config["temperature"],
        )
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'appel au modèle : {str(e)}"}), 500
    
    if response["error"]:
        return jsonify({"error": response["error"]}), 500
    
    cleaned_text = clean_response(response["text"])
    return jsonify({
        "question": question,
        "response": cleaned_text,
        "model_used": config["display_name"],
        "provider": config["provider"],
        "context_chunks": [{"text": c[:200] + "...", "source": m.get("source", "unknown")} for c, m in zip(chunks, metadatas)] if chunks else [],
        "latency_seconds": response["latency_seconds"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
    }), 200


# ─────────────────────────────────────────────
#  Endpoint du RAG optimise avec Re-Ranking (meilleur modèle du dernier benchmark)
# ─────────────────────────────────────────────

from utils.rag_optimized_functions import hybrid_search, rerank, BM25, DEFAULT_RETRIEVAL_COUNT, FINAL_COUNT, vector_rerank_search

# Variables globales pour l'optimisé
_rag_collection = None          # déjà défini pour le simple
_bm25 = None
_hybrid_search_initialized = False

def get_hybrid_components():
    global _rag_collection, _bm25, _hybrid_search_initialized
    if _rag_collection is None:
        _rag_collection = get_rag_collection()  # la même collection
    if _bm25 is None:
        _bm25 = BM25()
    return _rag_collection, _bm25


@api_bp.route("/rag-optimized", methods=["POST"])
def rag_optimized():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "Question manquante"}), 400

    question = body["question"].strip()
    if not question:
        return jsonify({"error": "Question vide"}), 400

    # Récupérer les composants
    collection, bm25 = get_hybrid_components()
    # Vérifier que la collection contient des documents
    if collection.count() == 0:
        return jsonify({"error": "Aucun document indexé dans la base de connaissances. Veuillez ajouter des PDF/DOCX dans le dossier 'documents'."}), 400

    try:
        # 1. Hybrid search
        # 1. Récupération vectorielle + reranking (sans BM25)
        retrieved_docs, retrieved_metas = vector_rerank_search(collection, question, DEFAULT_RETRIEVAL_COUNT)
        # 2. Reranking
        if retrieved_docs:
            reranked = rerank(question, retrieved_docs, top_k=FINAL_COUNT)
            best_docs = [retrieved_docs[i] for i, _ in reranked]
            best_metas = [retrieved_metas[i] for i, _ in reranked]
        else:
            best_docs = []
            best_metas = []

        # Construire le contexte
        if best_docs:
            context = "\n\n---\n\n".join(best_docs)
            context = f"Voici des extraits très pertinents pour répondre à la question :\n{context}\n\n---\n\n"
        else:
            context = "Aucun document pertinent trouvé.\n\n"

        # Prompt système avec contexte
        system_prompt_with_context = SYSTEM_PROMPT + f"\n\n{context}"

        # Utiliser Mistral explicitement
        config = MODELS_CONFIG.get("mistral")
        if not config:
            return jsonify({"error": "Modèle Mistral non configuré"}), 500

        # Appel LLM
        response = call_model(
            provider=config["provider"],
            model_id=config["model_id"],
            system_prompt=system_prompt_with_context,
            question=question,
            max_tokens=1500,
            temperature=config["temperature"],
        )

        if response["error"]:
            return jsonify({"error": response["error"]}), 500

        # Nettoyage des caractères markdown
        cleaned = clean_response(response["text"])

        return jsonify({
            "question": question,
            "response": cleaned,
            "model_used": config["display_name"],
            "retrieved_count": len(retrieved_docs),
            "retrieved_chunks": best_docs,
            "reranked_count": len(best_docs),
            "sources": [m.get("source", "unknown") for m in best_metas],
            "latency_seconds": response["latency_seconds"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
        }), 200

    except Exception as e:
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500

# enpoints qui vérifie le nombre de chunks dans la base de données
@api_bp.route("/rag-index-info", methods=["GET"])
def rag_index_info():
    collection = get_rag_collection()
    count = collection.count()
    # Optionnel : lister les sources
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    sources = set(m.get("source", "unknown") for m in all_metas)
    return jsonify({"chunks_count": count, "sources": list(sources)})


# ─────────────────────────────────────────────
#  Evaluation des modèles LLM simple ; RAG; RAG optimisé ; RAFT; RAG Agent ; RAG multi-agent
# ─────────────────────────────────────────────

from utils.evaluations import retrieval_precision, retrieval_recall_at_k, cosine_similarity_between, get_embedder
from sentence_transformers import SentenceTransformer, util

# Les processus sont évalués sur 30 questions du dataset à défaut de préciser le sample_size 
# Évaluer sur 30 questions (par défaut)
#curl http://localhost:5001/api/evaluations
# Évaluer sur 100 questions
#curl "http://localhost:5001/api/evaluations?sample_size=100"
# Toutes les questions (attention, long)
#curl "http://localhost:5001/api/evaluations?sample_size=1000"

@api_bp.route("/evaluations", methods=["GET"])
def evaluations():
    sample_size = request.args.get("sample_size", default=30, type=int)
    # Charger un échantillon du dataset (ou tout)
    questions = _load_dataset(sample_size=sample_size)

    processes = [
        {"name": "LLM simple", "endpoint": "/llm-simple", "has_retrieval": False},
        {"name": "RAG", "endpoint": "/rag-simple", "has_retrieval": True},
        {"name": "RAG optimisé", "endpoint": "/rag-optimized", "has_retrieval": True},
    ]

    base_url = "http://localhost:5001"
    results = {}

    for proc in processes:
        eval_details = []
        for item in questions:
            q = item["question"]
            ref_answer = item["reponse"]

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
                        # context_chunks est une liste de dict avec "text"
                        retrieved_texts = [c.get("text", "") for c in chunks if c.get("text")]
                    else:  # RAG optimisé
                        retrieved_texts = data.get("retrieved_chunks", [])
                        if retrieved_texts and isinstance(retrieved_texts[0], dict):
                            retrieved_texts = [c.get("text", "") for c in retrieved_texts]

                    if retrieved_texts:
                        # Fidélité : similarité max entre génération et chaque chunk (ou moyenne)
                        gen_emb = get_embedder().encode(generated)
                        chunk_embs = get_embedder().encode(retrieved_texts)
                        sims = util.cos_sim(gen_emb, chunk_embs)[0].tolist()
                        faithfulness = max(sims)  # la meilleure correspondance 

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
                "quality": round(avg_quality, 2), # en %
                "faithfulness": round(avg_faith, 2), # en %
                "retrieval_precision": round(avg_prec, 2), # en %
                "retrieval_recall_at_5": round(avg_recall, 2), # en %
                "latency_seconds": round(avg_lat, 3) # en secondes
            },
            "details": eval_details
        }

    return jsonify(results), 200