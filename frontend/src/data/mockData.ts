import { ProcessType, BenchmarkResults, StepScores } from '../types';

export const processResponses: Record<ProcessType, string> = {
  'LLM simple':
    "Pour économiser 500€ par mois, je vous recommande de : 1) Créer un budget détaillé, 2) Réduire les dépenses non essentielles, 3) Automatiser vos épargnes. C'est une approche générale basée sur les principes financiers courants.",
  'RAG':
    "D'après votre historique financier et les documents d'analyse budgétaire, vous dépensez actuellement 380€/mois en restauration. En réduisant cette catégorie de 50% et en optimisant vos abonnements (120€ identifiés), vous atteindrez facilement 500€ d'économies mensuelles.",
  'RAG optimisé':
    "Analyse détaillée de vos 6 derniers mois : Restaurants (380€), Abonnements sous-utilisés (120€), Transport (possibilité de covoiturage : 85€). En appliquant ces 3 optimisations ciblées et en suivant le plan d'épargne automatique que je vous propose, vous économiserez 585€/mois dès le premier mois.",
  'RAG fine-tuné (RAFT)':
    "Basé sur votre profil similaire aux utilisateurs du cluster #247 (revenus, dépenses, objectifs), voici votre plan personnalisé : 1) Réduction restaurants : 380€ → 180€ (200€ économisés), 2) Renégociation assurances : 45€/mois économisés selon votre profil, 3) Optimisation énergétique adaptée à votre logement : 110€/mois. Total : 355€ garantis + 200€ avec effort modéré = 555€.",
  'RAG + Agent IA':
    "J'ai analysé vos transactions des 12 derniers mois et détecté des opportunités : 1) Vos dépenses en restaurants ont augmenté de 35% (280€ → 380€). Je vous propose un plan graduel sur 3 mois. 2) J'ai identifié 4 abonnements redondants (120€). 3) Votre banque propose une offre groupée économisant 25€/mois. J'ai préparé les documents de souscription. Voulez-vous que je vous aide à les compléter ?",
  'RAG + Multi-agents':
    "📊 Agent Analyse : Revenu stable 3200€, dépenses 2850€, marge 350€. 📈 Agent Optimisation : 15 leviers identifiés, priorisés selon impact/effort. 🤖 Agent Négociation : J'ai contacté vos fournisseurs (assurance, internet) et obtenu des réductions préliminaires de 68€/mois. 💰 Agent Épargne : Placement automatique configuré vers PEL (3%) pour 500€/mois. 🎯 Résultat : Objectif dépassé (568€ économisés) avec plan d'action détaillé par agent.",
};

export const benchmarkData: BenchmarkResults = {
  metrics: [
    {
      processus: 'LLM simple',
      précision: 62,
      pertinence: 58,
      tempsDeRéponse: 1.2,
      fidélité: 55,
    },
    {
      processus: 'RAG',
      précision: 78,
      pertinence: 82,
      tempsDeRéponse: 2.1,
      fidélité: 75,
    },
    {
      processus: 'RAG optimisé',
      précision: 85,
      pertinence: 88,
      tempsDeRéponse: 2.8,
      fidélité: 83,
    },
    {
      processus: 'RAG fine-tuné',
      précision: 91,
      pertinence: 89,
      tempsDeRéponse: 3.2,
      fidélité: 90,
    },
    {
      processus: 'RAG + Agent',
      précision: 93,
      pertinence: 94,
      tempsDeRéponse: 4.5,
      fidélité: 92,
    },
    {
      processus: 'RAG + Multi',
      précision: 96,
      pertinence: 97,
      tempsDeRéponse: 5.8,
      fidélité: 95,
    },
  ],
};

export const stepScoresData: StepScores = {
  'LLM simple': [
    { étape: 'Compréhension', score: 75 },
    { étape: 'Génération', score: 65 },
    { étape: 'Cohérence', score: 60 },
  ],
  'RAG': [
    { étape: 'Récupération docs', score: 82 },
    { étape: 'Pertinence', score: 78 },
    { étape: 'Génération', score: 80 },
    { étape: 'Vérification', score: 75 },
  ],
  'RAG optimisé': [
    { étape: 'Récupération docs', score: 88 },
    { étape: 'Reranking', score: 85 },
    { étape: 'Génération', score: 87 },
    { étape: 'Vérification', score: 83 },
  ],
  'RAG fine-tuné (RAFT)': [
    { étape: 'Récupération docs', score: 92 },
    { étape: 'Reranking', score: 90 },
    { étape: 'Adaptation contexte', score: 91 },
    { étape: 'Génération', score: 90 },
    { étape: 'Vérification', score: 89 },
  ],
  'RAG + Agent IA': [
    { étape: 'Analyse situation', score: 94 },
    { étape: 'Récupération docs', score: 93 },
    { étape: 'Planification', score: 92 },
    { étape: 'Génération', score: 93 },
    { étape: 'Action proposée', score: 90 },
  ],
  'RAG + Multi-agents': [
    { étape: 'Agent Analyse', score: 96 },
    { étape: 'Agent Recherche', score: 95 },
    { étape: 'Agent Optimisation', score: 97 },
    { étape: 'Agent Négociation', score: 94 },
    { étape: 'Coordination', score: 95 },
    { étape: 'Génération finale', score: 96 },
  ],
};
