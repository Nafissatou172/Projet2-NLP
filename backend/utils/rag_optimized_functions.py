import math
import re
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_RETRIEVAL_COUNT = 50   # nombre de chunks avant reranking (augmenté pour améliorer le rappel)
FINAL_COUNT = 5     # nombre final après reranking (augmenté pour améliorer la précision et le rappel)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60                     # paramètre pour Reciprocal Rank Fusion
CHUNK_SIZE = 600        # plus petit qu'avant
CHUNK_OVERLAP = 100     # chevauchement réduit

# ---------------------------
# BM25 (keyword search)
# ---------------------------
class BM25:
    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.doc_len = []
        self.avg_len = 0
        self.idf = {}
        self.tokenized_corpus = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents: List[str]):
        self.corpus = documents
        self.tokenized_corpus = [self.tokenize(doc) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.tokenized_corpus]
        self.avg_len = sum(self.doc_len) / len(self.doc_len)

        self.doc_freqs = []
        for tokens in self.tokenized_corpus:
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)

        N = len(documents)
        term_doc_count = {}
        for freq in self.doc_freqs:
            for term in freq:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1

        self.idf = {}
        for term, df in term_doc_count.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str) -> List[float]:
        query_tokens = self.tokenize(query)
        scores = []
        for i, freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[i]
            for token in query_tokens:
                if token in freq:
                    tf = freq[token]
                    idf = self.idf.get(token, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
                    score += idf * (numerator / denominator)
            scores.append(score)
        return scores


# ---------------------------
# Reciprocal Rank Fusion
# ---------------------------
def reciprocal_rank_fusion(scores_a: List[float], scores_b: List[float], k=RRF_K) -> List[float]:
    # convertir scores en rangs (1 = meilleur score)
    rank_a = sorted(range(len(scores_a)), key=lambda i: scores_a[i], reverse=True)
    rank_b = sorted(range(len(scores_b)), key=lambda i: scores_b[i], reverse=True)

    rrf_scores = [0.0] * len(scores_a)
    for i, idx in enumerate(rank_a):
        rrf_scores[idx] += 1 / (k + i + 1)
    for i, idx in enumerate(rank_b):
        rrf_scores[idx] += 1 / (k + i + 1)
    return rrf_scores


# ---------------------------
# Hybrid Search
# ---------------------------
def hybrid_search(collection, query: str, bm25: BM25, n_results: int = DEFAULT_RETRIEVAL_COUNT):
    all_chunks = collection.get(include=["documents"])["documents"]
    if not all_chunks:
        return [], []
    total_chunks = len(all_chunks)
    retrieval_limit = min(total_chunks, 200)
    
    # Vector search sans include "ids" (ids sont toujours retournés)
    vector_res = collection.query(
        query_texts=[query],
        n_results=retrieval_limit,
        include=["documents", "metadatas", "distances"]
    )
    if not vector_res.get("documents") or not vector_res["documents"]:
        return [], []

    # Récupérer les IDs des documents vectoriels
    vector_ids = vector_res["ids"][0] if "ids" in vector_res else []
    
    # Construire mapping id -> score vectoriel
    all_ids = collection.get(include=[])["ids"]
    id_to_index = {id_: idx for idx, id_ in enumerate(all_ids)}
    vector_scores_full = [0.0] * total_chunks
    for i, doc_id in enumerate(vector_ids):
        if doc_id in id_to_index:
            vector_scores_full[id_to_index[doc_id]] = 1 - vector_res["distances"][0][i]

    # BM25
    if not bm25.corpus:
        bm25.fit(all_chunks)
    keyword_scores = bm25.score(query)

    rrf_scores = reciprocal_rank_fusion(vector_scores_full, keyword_scores)
    top_indices = sorted(range(total_chunks), key=lambda i: rrf_scores[i], reverse=True)[:n_results]

    all_metas = collection.get(include=["metadatas"])["metadatas"]
    selected_docs = [all_chunks[i] for i in top_indices]
    selected_metas = [all_metas[i] for i in top_indices]
    return selected_docs, selected_metas


# ---------------------------
# Reranking (Cross-Encoder)
# ---------------------------
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def rerank(query: str, documents: List[str], top_k: int = FINAL_COUNT) -> List[Tuple[int, float]]:
    if not documents:
        return []
    reranker = get_reranker()
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]   # ici les indices sont par rapport à `documents`

def vector_rerank_search(collection, query: str, n_results: int = DEFAULT_RETRIEVAL_COUNT):
    """
    Recherche vectorielle seule + reranking.
    """
    # 1. Récupération vectorielle des n_results chunks
    vector_res = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    if not vector_res.get("documents") or not vector_res["documents"]:
        return [], []
    
    docs = vector_res["documents"][0]
    metas = vector_res["metadatas"][0]
    # Pas de fusion, on garde ces chunks pour le reranking
    return docs, metas


# ---------------------------
# Query Expansion (optionnelle)
# ---------------------------
# Si vous voulez générer des variantes de la question via un petit LLM
# Exemple simple : reformulation avec un modèle local (ou via l'API Mistral)
# Nous l'ajoutons en option pour ne pas surcharger.

def expand_query_with_llm(question: str, call_llm_func) -> List[str]:
    """
    Utilise un LLM pour générer plusieurs reformulations de la question.
    call_llm_func est une fonction qui prend (prompt, question) et retourne une réponse.
    """
    prompt = f"""Génère 3 reformulations de la question suivante, en conservant le sens et le vocabulaire financier.
Question originale : {question}
Réponds uniquement avec les 3 reformulations, une par ligne, sans numéros ni commentaires.
"""
    # Appel à un LLM (par ex. Mistral Tiny ou Qwen)
    # Exemple avec Mistral via votre fonction existante
    from model_clients import call_model
    from models_config import MODELS_CONFIG
    config = MODELS_CONFIG.get("qwen")  # ou "mistral"
    resp = call_model(config["provider"], config["model_id"], "Tu es un assistant utile.", prompt, 150, 0.2)
    if resp["error"]:
        return [question]
    lines = resp["text"].strip().split("\n")
    variations = [line.strip() for line in lines if line.strip()]
    return [question] + variations[:3]  # inclut l'original

