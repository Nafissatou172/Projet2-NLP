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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  raftResult?: any;   // payload brut de /api/raft, /api/rag-agent, /api/rag-multi-agent
  processName?: string; // nom du processus IA ayant généré la réponse
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
