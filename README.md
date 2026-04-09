# 🇸🇳 FinChat Senegal - Assistance Financière Intelligente 🇸🇳

FinChat-SN est une plateforme applicative novatrice (Backend Flask + Frontend React) conçue pour comparer et illustrer différentes architectures de Modèles d'Intelligence Artificielle de Traitement du Langage Naturel (NLP). Elle se spécialise dans l'assistance financière, utilisant un ensemble de documents financiers complexes.

L'objectif de ce projet est de démontrer l'évolution et l'optimisation des réponses d'un LLM depuis un appel standard vers des architectures complexes (RAG, ReAct Agent, Workflow Multi-Agents) et d'y adosser un système complet d'évaluation temps réel des performances.

## 📁 Structure globale du Projet

```text
Projet2-NLP/
├── backend/                   # Dossier contenant toute la logique serveur & IA
│   ├── app.py                 # Point d'entrée de l'API Flask
│   ├── routes.py              # Définition de toutes nos routes /api et des endpoints de traitement
│   ├── requirements.txt       # Dépendances Python nécessaires (Flask, ChromaDB, HuggingFace, etc.)
│   ├── benchmark/             # Module dédié au benchmarking de différents modèles
│   │   ├── model_clients.py   # Gestion de la communication avec mistral-api, groq, huggingface_chat
│   │   └── models_config.py   # Configuration et listing statique des LLMs disponibles
│   ├── utils/                 # Cœur algorithmique - Les différents pipelines RAG
│   │   ├── agent.py               # Implémentation de l'Agent IA ReAct (Boucle d'action/observation)
│   │   ├── evaluations.py         # Métriques d'évaluations (Cosine Similarity, Précision, Recall@K, Latence)
│   │   ├── multi_agent.py         # Implémentation de l'architecture "Linguist Chain" à 3 Agents (Searcher, Critic, Generator)
│   │   ├── rag_optimized_functions.py # Logique RAG (BM25, Re-Ranking et Vector Store)
│   │   └── rag_simple_functions.py    # Logique RAG de base (Recherche par cosinus standard)
│   ├── chroma_db/             # Base de données vectorielle (Chroma) pour le stockage des embeddings
│   ├── dataset/               # Ensembles de tests (CSV de Q&A de références) utilisés pour évaluer le RAG
│   └── documents/             # Base documentaire d'origine (les PDF de finance pour populer la DB vectorielle)
│
└── frontend/                  # Dossier contenant l'application Client
    ├── src/
    │   ├── App.tsx            # Composant principal d'orchestration (IHM principale, Appels API)
    │   ├── components/        # Dossier contenant l'interface modulaire
    │   │   ├── ChatInterface.tsx         # Le module de Chat (discussion avec l'assistant)
    │   │   ├── EvaluationDashboard.tsx   # Dashboard graphique traçant nos scores (Recharts)
    │   │   ├── BenchmarkSection.tsx      # Panneau latéral pour configurer ou lancer un test benchmark
    │   │   └── Sidebar.tsx               # Panneau de navigation latéral entre les différents modes RAG
    │   ├── types/             # Interface Typescript (Message, ProcessType, Metrics, etc.)
    │   └── data/              # Mock Data ou données locales pour les fallbackUI
    ├── package.json           # Dépendances Node.js (React, Vite, Recharts, TailwindCSS)
    └── index.html             # Template HTML applicatif
```

## 🧠 Fonctionnalités Principales (Architectures Comparées)

Le projet intègre 5 pipelines de fonctionnement, sélectionnables depuis le Front-end :

1. **LLM simple** : Interrogation directe du modèle IA, sans aucune connaissance ajoutée.
2. **RAG** : Retrieval-Augmented Generation basique avec recherche vectorielle classique.
3. **RAG Optimisé** : Amélioration du processus via un algorithme de Re-Ranking (Tri intelligent).
4. **RAG + Agent IA (ReAct)** : Un Agent qui décide de manière autonome quand il doit ou non faire des recherches additionnelles.
5. **RAG + Multi-agents** : Un système robuste en 3 temps (Le "Chercheur", L'"Évaluateur" (Critic), et Le "Rédacteur"). Ils collaborent et valident la véracité des informations trouvées avant de répondre à l'utilisateur final.

## 📊 Évaluation Automatisée

La plateforme est livrée avec un évaluateur embarqué via l'onglet `Évaluation` du Frontend.
Il analyse toutes les approches ci-dessus simultanément et trace en direct un comparatif basé sur plusieurs KPIs :
* **Qualité** (Cosine Similarity VS référence)
* **Fidélité** (S'assure que le modèle n'hallucine pas en dehors du document trouvé)
* **Précision Retrieval & Recall@5** (Qualité des chunks sélectionnés)
* **Latence (Vitesse)**

## 🚀 Lancement

**Backend :**
```bash
cd backend
pip install -r requirements.txt
python app.py
```
*Le backend sera servi par défaut sur http://localhost:5001.* *(Veuillez posséder votre `.env` à la racine backend)*

**Frontend :**
```bash
cd frontend
npm install
npm run dev
```
*Le frontend sera disponible pour ouvrir dans votre navigateur depuis http://localhost:5173.*
