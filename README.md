<![CDATA[<div align="center">

# 🇸🇳 FinChat Senegal — Assistance Financière Intelligente

**Chatbot IA spécialisé en finance personnelle, comparant 6 architectures NLP avancées**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge)](https://www.trychroma.com)

</div>

---

## 📌 Contexte & Objectif

Ce projet est une **plateforme full-stack** conçue pour explorer et comparer différentes architectures de traitement du langage naturel (NLP) appliquées à la **finance personnelle**.

### Le problème
Les LLMs utilisés seuls ("vanilla") génèrent souvent des réponses génériques, parfois obsolètes ou incohérentes (hallucinations) sur des sujets financiers spécifiques. Comment améliorer la pertinence et la fiabilité des réponses ?

### La solution
Nous implémentons et comparons **6 pipelines progressifs** — du LLM simple jusqu'aux architectures multi-agents — permettant de mesurer concrètement l'impact de chaque technique d'augmentation sur la qualité des réponses.

### Données sources
- **3 documents PDF** de référence sur la finance personnelle (budgets, épargne, investissement, dettes, fiscalité…)
- **35 questions de test** couvrant 10 catégories (budget, épargne, investissement, dette, retraite, fiscalité, assurance, immobilier, concepts, protection sociale)

---

## 🧠 Architectures Comparées

Sélectionnables en temps réel depuis l'interface de chat :

| # | Pipeline | Principe | Avantage clé |
|:-:|----------|----------|-------------|
| 1 | **LLM Simple** | Appel direct au modèle sans contexte externe | Baseline de référence |
| 2 | **RAG** | Retrieval-Augmented Generation — recherche vectorielle | Réponses contextualisées |
| 3 | **RAG Optimisé** | RAG hybride BM25 + vecteurs + re-ranking cross-encoder | Meilleure précision de retrieval |
| 4 | **RAFT** | Simulation fine-tuning : oracle docs + distracteurs + Chain-of-Thought | Raisonnement structuré |
| 5 | **RAG + Agent (ReAct)** | Agent autonome : boucle Thought → Action → Observation | Recherches itératives adaptatives |
| 6 | **RAG + Multi-Agents** | Pipeline 3 agents : Searcher → Critic → Generator | Validation croisée du contexte |

### 🔬 Focus : Simulation RAFT

Le pipeline RAFT simule le comportement d'un modèle affiné sur des tâches de retrieval :
- **Documents Oracle** : chunks réellement pertinents à la question
- **Distracteurs** : chunks non pertinents volontairement injectés
- **Chain-of-Thought (CoT)** : le modèle justifie explicitement sa sélection de documents
- **Statistiques** : ratio oracle/distracteurs, tokens consommés, latence, score de structure

### 🤖 Focus : Architecture Multi-Agents

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Searcher   │────▶│   Critic     │────▶│  Generator   │
│ (Retrieval) │     │ (Validation) │     │  (Réponse)   │
└─────────────┘     └──────────────┘     └──────────────┘
```

---

## ⚙️ Stack Technique

| Couche | Technologies |
|--------|-------------|
| **LLM** | Claude (Anthropic), Mistral, Groq, HuggingFace Inference |
| **Embeddings** | `sentence-transformers` via HuggingFace |
| **Vector Store** | ChromaDB (stockage local des embeddings) |
| **Re-ranking** | BM25 (`rank_bm25`) + Cross-encoder |
| **Backend** | Python 3.10+, Flask 3.x, flask-cors |
| **Frontend** | React 18, TypeScript 5, Vite 5 |
| **UI** | TailwindCSS, Lucide Icons, Recharts (graphiques) |
| **Évaluation** | Cosine Similarity, Fidélité, Précision, Recall\@5, Latence |

---

## 📁 Structure du Projet

```
RAG-IA-FINANCE/
├── backend/
│   ├── app.py                          # Point d'entrée Flask
│   ├── routes.py                       # Endpoints API REST
│   ├── data_quality.py                 # Script d'analyse qualité du dataset
│   ├── requirements.txt               # Dépendances Python
│   ├── env.example                     # Template variables d'environnement
│   ├── benchmark/
│   │   ├── model_clients.py            # Clients LLM (Mistral, Groq, HuggingFace)
│   │   ├── evaluators.py              # Évaluateurs de benchmark
│   │   └── models_config.py            # Configuration des modèles disponibles
│   ├── utils/
│   │   ├── rag_simple_functions.py     # RAG basique (similarité cosinus)
│   │   ├── rag_optimized_functions.py  # RAG hybride (BM25 + Vector + Re-Ranking)
│   │   ├── raft.py                     # Simulation RAFT (oracle + distracteurs + CoT)
│   │   ├── agent.py                    # Agent ReAct (Thought → Action → Observation)
│   │   ├── multi_agent.py              # Pipeline 3 agents (Searcher → Critic → Generator)
│   │   └── evaluations.py              # Métriques d'évaluation
│   ├── dataset/                        # Dataset CSV de questions-réponses (35 Q&A)
│   └── documents/                      # 3 documents PDF sources (finance personnelle)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Orchestrateur principal
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx       # Interface de chat + panneau d'analyse
│   │   │   ├── RaftPanel.tsx           # Visualisation RAFT (oracle, distracteurs, CoT)
│   │   │   ├── EvaluationDashboard.tsx # Dashboard de scores (graphiques Recharts)
│   │   │   ├── BenchmarkSection.tsx    # Benchmarks multi-modèles
│   │   │   ├── ProcessSelector.tsx     # Sélecteur de pipeline IA
│   │   │   └── Sidebar.tsx            # Navigation latérale
│   │   └── types/                      # Interfaces TypeScript
│   ├── package.json
│   └── index.html
│
├── all_questions_evaluation.csv        # Résultats d'évaluation comparative
└── README.md
```

---

## 🚀 Installation & Lancement

### Prérequis

- **Python** 3.10+
- **Node.js** 18+
- **npm** 9+
- Clés API : HuggingFace (requis), + optionnel : Anthropic, Mistral, Groq

### 1. Cloner le repository

```bash
git clone https://github.com/Nafissatou172/Projet2-NLP.git
cd Projet2-NLP
```

### 2. Configuration des variables d'environnement

```bash
cp backend/env.example backend/.env
```

Éditez `backend/.env` avec vos clés API :

```env
FLASK_DEBUG=True
SECRET_KEY=change-me-in-production
PORT=5001

HUGGINGFACE_API_KEY=votre_clé_huggingface    # Requis (embeddings)
ANTHROPIC_API_KEY=votre_clé_anthropic         # Optionnel
MISTRAL_API_KEY=votre_clé_mistral             # Optionnel
GROQ_API_KEY=votre_clé_groq                   # Optionnel

MOCK_MODE=false
```

> **Note :** Seule la clé `HUGGINGFACE_API_KEY` est strictement nécessaire pour le pipeline RAG (embeddings). Les autres clés activent les modèles LLM correspondants.

### 3. Backend (Flask)

```bash
cd backend
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
python app.py
```

> 🟢 Le serveur API démarre sur **http://localhost:5001**

### 4. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

> 🟢 L'interface est accessible sur **http://localhost:5173**

---

## 📊 Évaluation & Résultats

### Métriques utilisées

| Métrique | Description |
|----------|-------------|
| **Cosine Similarity** | Proximité sémantique entre la réponse générée et la réponse de référence |
| **Fidélité** | Taux de détection d'hallucinations (réponses hors contexte) |
| **Précision Retrieval** | Qualité et pertinence des documents récupérés |
| **Recall\@5** | Couverture des documents pertinents dans le top 5 des résultats |
| **Latence** | Temps de réponse de bout en bout (en secondes) |

### Dashboard d'évaluation

L'onglet **Évaluation** du frontend permet de :
1. Lancer une évaluation simultanée de tous les pipelines sur un échantillon de questions
2. Visualiser un tableau de bord comparatif avec graphiques (Recharts)
3. Exporter les résultats au format CSV

### Panneau d'analyse adaptatif

Après chaque réponse des pipelines avancés, un panneau dépliable affiche les statistiques :

| Pipeline | Panneau | Couleur |
|----------|---------|---------|
| RAFT | Analyse RAFT (oracle, distracteurs, CoT) | 🟣 Violet |
| Agent ReAct | Analyse Agent IA (étapes de raisonnement) | 🔵 Bleu |
| Multi-agents | Analyse Multi-agents (flux entre agents) | 🟠 Ambre |

---

## 🔌 API Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/api/llm-simple` | Réponse LLM directe (sans contexte) |
| `POST` | `/api/rag-simple` | RAG basique |
| `POST` | `/api/rag-optimized` | RAG hybride + re-ranking |
| `POST` | `/api/raft` | Simulation RAFT |
| `POST` | `/api/rag-agent` | Agent ReAct |
| `POST` | `/api/rag-multi-agent` | Pipeline multi-agents |
| `GET`  | `/api/evaluations` | Évaluation comparative (`?sample_size=N`) |
| `POST` | `/api/benchmark` | Benchmark multi-modèles |

---

## 🧪 Dataset de test

Le fichier `all_questions_evaluation.csv` contient **35 questions** couvrant 10 catégories de finance personnelle :

| Catégorie | Exemples de questions |
|-----------|----------------------|
| Budget | Comment créer un budget mensuel ? Règle 50/30/20 ? |
| Épargne | Fonds d'urgence, Livret A, règle des 6 mois |
| Investissement | ETF, PEA, diversification, dividendes |
| Dette | Méthode avalanche vs boule de neige, TAEG |
| Retraite | PER, intérêts composés long terme |
| Fiscalité | Impôt progressif, flat tax / PFU |
| Assurance | Assurances indispensables, assurance vie |
| Immobilier | Acheter vs louer, capacité d'emprunt |
| Concepts | Intérêts composés, inflation, bilan financier |
| Protection sociale | Assurance chômage, aides au logement |

---

## 👥 Équipe

Projet réalisé dans le cadre du cursus NLP/IA.

---

## 📄 Licence

Ce projet est à usage académique et de démonstration.
]]>
