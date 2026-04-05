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