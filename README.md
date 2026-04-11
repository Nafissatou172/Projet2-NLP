# 🇸🇳 FinChat Senegal — Assistance Financière Intelligente 🇸🇳

**FinChat-SN** est une plateforme applicative complète (Backend Flask + Frontend React/TypeScript) conçue pour comparer et illustrer différentes architectures de traitement du langage naturel (NLP) appliquées à l'assistance financière.

L'objectif est de démontrer l'évolution des réponses d'un LLM depuis un appel simple jusqu'aux architectures avancées (RAG, RAFT, ReAct Agent, Multi-Agents) avec un système complet d'évaluation des performances en temps réel.

---

## 📁 Structure du Projet

```text
Projet2-NLP/
├── backend/
│   ├── app.py                          # Point d'entrée Flask
│   ├── routes.py                       # Tous les endpoints /api/*
│   ├── requirements.txt                # Dépendances Python
│   ├── env.example                     # Modèle de configuration des variables d'environnement
│   ├── data_quality.py                 # Script d'analyse de la qualité des données
│   ├── benchmark/
│   │   ├── model_clients.py            # Clients Mistral, Groq, HuggingFace
│   │   └── models_config.py            # Configuration des LLMs disponibles
│   ├── utils/
│   │   ├── agent.py                    # Agent ReAct (boucle Thought → Action → Observation)
│   │   ├── evaluations.py              # Métriques : Cosine Similarity, Précision, Recall@K, Latence
│   │   ├── multi_agent.py              # Architecture 3-agents (Searcher → Critic → Generator)
│   │   ├── raft.py                     # Simulation RAFT : oracle docs, distracteurs, CoT
│   │   ├── rag_optimized_functions.py  # RAG hybride (BM25 + Vector + Re-Ranking)
│   │   └── rag_simple_functions.py     # RAG basique (similarité cosinus)
│   ├── chroma_db/                      # Base vectorielle ChromaDB (embeddings)
│   ├── dataset/                        # Jeux de tests CSV (Q&A de référence)
│   └── documents/                      # Documents PDF source (finance personnelle)
│
└── frontend/
    ├── src/
    │   ├── App.tsx                     # Orchestrateur principal (état, appels API)
    │   ├── components/
    │   │   ├── ChatInterface.tsx        # Interface de chat + panneau d'analyse adaptatif
    │   │   ├── RaftPanel.tsx            # Visualisation RAFT (oracle, distracteurs, CoT, stats)
    │   │   ├── EvaluationDashboard.tsx  # Dashboard de scores (graphiques Recharts)
    │   │   ├── BenchmarkSection.tsx     # Lancement et résultats de benchmarks LLM
    │   │   ├── ProcessSelector.tsx      # Sélecteur de pipeline IA
    │   │   └── Sidebar.tsx             # Navigation latérale
    │   ├── types/                       # Interfaces TypeScript (Message, ProcessType, Metrics…)
    │   └── data/                        # Données mock / fallback UI
    ├── package.json
    └── index.html
```

---

## 🧠 Architectures Comparées (6 Pipelines)

Sélectionnables en temps réel depuis l'interface de chat :

| # | Pipeline | Description |
|---|----------|-------------|
| 1 | **LLM simple** | Appel direct au modèle sans contexte externe |
| 2 | **RAG** | Retrieval-Augmented Generation — recherche vectorielle classique |
| 3 | **RAG Optimisé** | RAG hybride BM25 + vecteurs avec re-ranking cross-encoder |
| 4 | **RAG fine-tuné (RAFT)** | Simulation RAFT : oracle docs réels + distracteurs + raisonnement CoT |
| 5 | **RAG + Agent IA** | Agent autonome ReAct qui décide quand lancer des recherches supplémentaires |
| 6 | **RAG + Multi-agents** | Pipeline 3-agents : Searcher → Critic → Generator avec validation du contexte |

### 🔬 Simulation RAFT (Retrieval-Augmented Fine-Tuning)

Le pipeline RAFT simule l'entraînement d'un modèle affiné :
- **Documents Oracle** : chunks réellement pertinents à la question
- **Distracteurs** : chunks non pertinents volontairement injectés
- **Raisonnement Chain-of-Thought** : le modèle justifie sa sélection de documents
- **Statistiques** : ratio oracle, tokens, latence, score de structure

### 🤖 Panneau d'Analyse Adaptatif

Après chaque réponse des pipelines Agent, Multi-agents ou RAFT, un **panneau dépliable** s'affiche automatiquement avec les statistiques du pipeline utilisé. Le titre et la couleur s'adaptent selon le processus :

| Pipeline | Titre du panneau | Couleur |
|----------|-----------------|---------|
| RAG fine-tuné (RAFT) | Analyse RAFT | 🟣 Violet |
| RAG + Agent IA | Analyse Agent IA (ReAct) | 🔵 Bleu |
| RAG + Multi-agents | Analyse Multi-agents | 🟠 Ambre |

---

## 📊 Évaluation Automatisée

Accessible depuis l'onglet **Évaluation** du frontend, le module teste simultanément tous les pipelines sur un échantillon de questions de référence et génère un tableau de bord comparatif :

| Métrique | Description |
|----------|-------------|
| **Cosine Similarity** | Proximité sémantique avec la réponse de référence |
| **Fidélité** | Détection d'hallucinations hors contexte |
| **Précision Retrieval** | Qualité des documents récupérés |
| **Recall@5** | Couverture des documents pertinents dans le top 5 |
| **Latence** | Temps de réponse de bout en bout |

---

## ⚙️ Configuration

### Variables d'environnement

Copiez `env.example` vers `.env` dans le dossier `backend/` et renseignez vos clés :

```bash
cp backend/env.example backend/.env
```

```env
FLASK_DEBUG=True
SECRET_KEY=change-me-in-production
PORT=5001

HUGGINGFACE_API_KEY=**clé_huggingface**
ANTHROPIC_API_KEY=**clé_anthropic**
MISTRAL_API_KEY=**clé_mistral**
GROQ_API_KEY=**clé_groq**

MOCK_MODE=false
```

> **Note :** Seul `HUGGINGFACE_API_KEY` est strictement requis pour le pipeline RAG (embeddings). Les autres clés activent les modèles LLM correspondants dans le benchmark.

---

## 🚀 Lancement

### Backend (Flask — Python 3.10+)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

> Le serveur démarre sur **http://localhost:5001**

### Frontend (React + Vite — Node 18+)

```bash
cd frontend
npm install
npm run dev
```

> L'interface est accessible sur **http://localhost:5173**

---

## 🛠️ Stack Technique

| Couche | Technologies |
|--------|-------------|
| **LLM** | OpenAI-compatible API, Groq, Mistral, HuggingFace Inference |
| **Embeddings** | `sentence-transformers` via HuggingFace |
| **Vector Store** | ChromaDB |
| **Re-ranking** | BM25 (`rank_bm25`) + Cross-encoder |
| **Backend** | Flask, Python |
| **Frontend** | React 18, TypeScript, Vite, Lucide Icons, Recharts |
| **Styles** | TailwindCSS |

---

## 📝 Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/api/llm-simple` | Réponse LLM directe |
| `POST` | `/api/rag-simple` | RAG basique |
| `POST` | `/api/rag-optimized` | RAG hybride + re-ranking |
| `POST` | `/api/raft` | Simulation RAFT (oracle + distracteurs + CoT) |
| `POST` | `/api/rag-agent` | Agent ReAct |
| `POST` | `/api/rag-multi-agent` | Pipeline multi-agents |
| `GET`  | `/api/evaluations` | Évaluation comparative (`?sample_size=N`) |
| `POST` | `/api/benchmark` | Benchmark multi-modèles |
