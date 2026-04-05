import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { EvaluationAverage } from '../types/types';
import { ProcessType } from '../types';
import { CheckCircle } from 'lucide-react';

interface StepEvaluationSectionProps {
  currentProcess: ProcessType;
  evalScores?: EvaluationAverage | null;
}

const colors = ['#10b981', '#059669', '#047857', '#065f46', '#064e3b'];

export const StepEvaluationSection: React.FC<StepEvaluationSectionProps> = ({ currentProcess, evalScores }) => {
  if (!evalScores) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 border border-green-100">
        <div className="flex items-center gap-2 mb-6 pb-4 border-b border-green-100">
          <CheckCircle className="w-6 h-6 text-green-600" />
          <h2 className="text-xl font-bold text-gray-900">Évaluation par Étape</h2>
        </div>
        <div className="h-96 flex flex-col items-center justify-center bg-gradient-to-b from-gray-50 to-white rounded-lg border-2 border-dashed border-green-200">
          <CheckCircle className="w-12 h-12 text-green-200 mb-3" />
          <p className="text-gray-500 font-medium">Aucune donnée d'évaluation pour {currentProcess}</p>
          <p className="text-xs text-gray-400 mt-2">Lancez un benchmark ou une évaluation d'abord</p>
        </div>
      </div>
    );
  }

  // Transformer les moyennes en format pour le graphique
  const stepScores = [
    { étape: "Qualité (génération)", score: Math.round(evalScores.quality * 100) },
    { étape: "Fidélité (aux sources)", score: Math.round(evalScores.faithfulness * 100) },
    { étape: "Précision (retrieval)", score: Math.round(evalScores.retrieval_precision * 100) },
    { étape: "Rappel@5", score: Math.round(evalScores.retrieval_recall_at_5 * 100) },
    { étape: "Latence (secondes)", score: evalScores.latency_seconds > 5 ? 0 : Math.max(0, 100 - evalScores.latency_seconds * 10) }, // métrique inversée
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-green-100">
      <div className="flex items-center gap-2 mb-6 pb-4 border-b border-green-100">
        <CheckCircle className="w-6 h-6 text-green-600" />
        <h2 className="text-xl font-bold text-gray-900">Évaluation par Métrique</h2>
        <span className="ml-auto px-3 py-1 bg-gradient-to-r from-green-100 to-green-50 text-green-700 text-xs font-semibold rounded-full border border-green-200">
          {currentProcess}
        </span>
      </div>

      <div className="space-y-6">
        <div className="h-80 bg-gradient-to-b from-gray-50 to-white rounded-lg p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stepScores} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" domain={[0, 100]} stroke="#9ca3af" label={{ value: 'Score (%)', position: 'insideBottom', offset: -5 }} />
              <YAxis dataKey="étape" type="category" width={150} stroke="#9ca3af" />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '2px solid #10b981', borderRadius: '8px' }} />
              <Bar dataKey="score" fill="#10b981" radius={[0, 8, 8, 0]}>
                {stepScores.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {stepScores.map((step, index) => (
            <div key={index} className="p-4 bg-gradient-to-br from-gray-50 to-white rounded-lg border-2 border-green-100 hover:border-green-300 transition-colors">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-4 h-4 rounded-full shadow-md" style={{ backgroundColor: colors[index % colors.length] }}></div>
                <p className="text-sm font-semibold text-gray-800">{step.étape}</p>
              </div>
              <div className="flex items-end gap-2">
                <p className="text-3xl font-bold text-green-600">{step.score}</p>
                <p className="text-gray-500 font-medium mb-1">%</p>
              </div>
              <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
                <div className="h-2 rounded-full transition-all" style={{ width: `${step.score}%`, backgroundColor: colors[index % colors.length] }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};