// types.ts
export interface BenchmarkResults {
  duration_seconds: number;
  sample_size: number;
  models_evaluated: string[];
  results: ModelResult[];
}

export interface ModelResult {
  model_key: string;
  display_name: string;
  provider: string;
  aggregated: AggregatedMetrics;
  details: QuestionDetail[];
  rank: number;
  errors: number;
  questions_answered: number;
  questions_total: number;
}

export interface AggregatedMetrics {
  quality: number;
  faithfulness: number;
  latency_avg_seconds: number;
  latency_label: string;
  latency_score: number;
  cost_score: number;
  total_cost_usd: number;
  global_score: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

export interface QuestionDetail {
  question: string;
  reference: string;
  generated: string;
  category: string;
  difficulty: string;
  error: string | null;
  metrics: {
    quality: number;
    faithfulness: number;
    latency: { seconds: number; label: string; score: number };
    cost: { cost_usd: number; score: number };
    global_score: number;
  } | null;
}

// Format pour les graphiques
export interface ChartData {
  processus: string;
  précision: number;
  pertinence: number;
  fidélité: number;
  tempsDeRéponse: number;
  scoreGlobal: number;
}


export interface EvaluationAverage {
  quality: number;
  faithfulness: number;
  retrieval_precision: number;
  retrieval_recall_at_5: number;
  latency_seconds: number;
}

export interface EvaluationResults {
  [processName: string]: {
    average: EvaluationAverage;
    details?: any[];
  };
}