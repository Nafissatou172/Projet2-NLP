import React from 'react';
import { ProcessType } from '../types';
import { Zap } from 'lucide-react';

interface ProcessSelectorProps {
  selectedProcess: ProcessType;
  onProcessChange: (process: ProcessType) => void;
}

const processes: ProcessType[] = [
  'LLM simple',
  'RAG',
  'RAG optimisé',
  'RAG fine-tuné (RAFT)',
  'RAG + Agent IA',
  'RAG + Multi-agents',
];

const processDescriptions: Record<ProcessType, string> = {
  'LLM simple': 'Réponse directe du modèle',
  'RAG': 'Récupération avec documents',
  'RAG optimisé': 'RAG avec reranking',
  'RAG fine-tuné (RAFT)': 'RAG avec fine-tuning',
  'RAG + Agent IA': 'RAG avec agent autonome',
  'RAG + Multi-agents': 'Orchestration multi-agents',
};

export const ProcessSelector: React.FC<ProcessSelectorProps> = ({ selectedProcess, onProcessChange }) => {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>): void => {
    onProcessChange(event.target.value as ProcessType);
  };

  return (
    <div className="bg-white rounded-xl shadow-md p-6 border border-green-100">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-green-600" />
        <label htmlFor="process-select" className="text-sm font-semibold text-gray-800">
          Sélectionner le processus IA
        </label>
      </div>

      <select
        id="process-select"
        value={selectedProcess}
        onChange={handleChange}
        className="w-full px-4 py-3 border-2 border-green-200 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white text-gray-900 shadow-sm transition-all"
      >
        {processes.map((process) => (
          <option key={process} value={process}>
            {process}
          </option>
        ))}
      </select>

      <p className="mt-3 text-sm text-gray-600">
        <span className="font-semibold text-green-700">Description:</span> {processDescriptions[selectedProcess]}
      </p>
    </div>
  );
};
