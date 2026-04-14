# utils/rag_simple_functions.py

import os
import re
import hashlib
from pathlib import Path
import pdfplumber
from docx import Document
import chromadb
from chromadb.utils import embedding_functions

# Configuration
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
TOP_K = 5

def extract_text_from_pdf(file_path: Path) -> str:
    """Extrait le texte d'un fichier PDF."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_path: Path) -> str:
    """Extrait le texte d'un fichier DOCX."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_all_documents(documents_dir: Path) -> list[dict]:
    """Charge tous les documents du dossier documents_dir."""
    docs = []
    for file_path in documents_dir.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == ".docx":
            text = extract_text_from_docx(file_path)
        else:
            continue
        if text.strip():
            docs.append({
                "source": file_path.name,
                "text": text
            })
    return docs

def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Découpe le texte en chunks avec chevauchement."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]
        chunk_id = hashlib.md5(f"{source}_{start}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source": source,
            "start": start,
            "end": end
        })
        start += chunk_size - overlap
    return chunks

def init_vectorstore(documents_dir: Path, chroma_db_dir: Path) -> chromadb.Collection:
    """Initialise ChromaDB et indexe les documents si nécessaire."""
    chroma_db_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chroma_db_dir))
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    collection_name = "finance_docs"
    existing_collections = [col.name for col in client.list_collections()]
    if collection_name in existing_collections:
        print(f"Collection {collection_name} déjà existante, réutilisation.")
        return client.get_collection(name=collection_name, embedding_function=embedding_fn)
    
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    
    documents = load_all_documents(documents_dir)
    if not documents:
        print("Aucun document trouvé dans le dossier documents/")
        return collection
    
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], doc["source"])
        all_chunks.extend(chunks)
    
    if not all_chunks:
        print("Aucun chunk généré")
        return collection
    
    ids = [chunk["id"] for chunk in all_chunks]
    texts = [chunk["text"] for chunk in all_chunks]
    metadatas = [{"source": chunk["source"], "start": chunk["start"], "end": chunk["end"]} for chunk in all_chunks]
    
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"Collection {collection_name} créée avec {len(all_chunks)} chunks")
    return collection