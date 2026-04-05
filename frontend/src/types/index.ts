export type ProcessType =
  | 'LLM simple'
  | 'RAG'
  | 'RAG optimisé'
  | 'RAG fine-tuné (RAFT)'
  | 'RAG + Agent IA'
  | 'RAG + Multi-agents';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

export interface BenchmarkMetric {
  processus: string;
  précision: number;
  pertinence: number;
  tempsDeRéponse: number;
  fidélité: number;
}

export interface BenchmarkResults {
  metrics: BenchmarkMetric[];
}

export interface StepScore {
  étape: string;
  score: number;
}

export interface StepScores {
  [key: string]: StepScore[];
}
