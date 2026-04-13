import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { EvaluationResults } from '../types/types';
import { CheckCircle, Play } from 'lucide-react';

interface EvaluationDashboardProps {
  results: EvaluationResults | null;
  onRunEvaluation: () => void;
  isEvaluating: boolean;
}

const processOrder = ['LLM simple', 'RAG', 'RAG optimisé', 'RAG Agent', 'RAG + Multi-agents'];
const metricNames = {
  quality: 'Qualité',
  faithfulness: 'Fidélité',
  retrieval_precision: 'Précision retrieval',
  retrieval_recall_at_5: 'Rappel@5',
};

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({ results, onRunEvaluation, isEvaluating }) => {
  // Préparer les données pour le graphique (tous processus)
  const chartData = results
    ? processOrder.map(proc => ({
        processus: proc,
        Qualité: results[proc]?.average?.quality ? Math.round(results[proc].average.quality) : 0,
        Fidélité: results[proc]?.average?.faithfulness ? Math.round(results[proc].average.faithfulness ) : 0,
        Précision: results[proc]?.average?.retrieval_precision ? Math.round(results[proc].average.retrieval_precision ) : 0,
        Rappel: results[proc]?.average?.retrieval_recall_at_5 ? Math.round(results[proc].average.retrieval_recall_at_5 ) : 0,
        Latence: results[proc]?.average?.latency_seconds ? results[proc].average.latency_seconds.toFixed(1) : 0,
      }))
    : [];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-green-100">
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-green-100">
        <div className="flex items-center gap-2">
          <CheckCircle className="w-6 h-6 text-green-600" />
          <h2 className="text-xl font-bold text-gray-900">Évaluation des Processus</h2>
        </div>
        <button
          onClick={onRunEvaluation}
          disabled={isEvaluating}
          className="px-4 py-2 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-lg hover:from-green-700 hover:to-green-600 disabled:from-gray-300 disabled:to-gray-300 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          {isEvaluating ? 'Évaluation en cours...' : 'Lancer l\'évaluation'}
        </button>
      </div>

      {results ? (
        <div className="space-y-8">
          {/* Graphique comparatif */}
          <div className="h-96 bg-gradient-to-b from-gray-50 to-white rounded-lg p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="processus" angle={-15} textAnchor="end" height={80} />
                <YAxis domain={[0, 100]} label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Qualité" fill="#10b981" />
                <Bar dataKey="Fidélité" fill="#059669" />
                <Bar dataKey="Précision" fill="#047857" />
                <Bar dataKey="Rappel" fill="#065f46" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Tableau des métriques détaillées */}
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white rounded-lg overflow-hidden">
              <thead className="bg-green-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Processus</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">Qualité (%)</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">Fidélité (%)</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">Précision retrieval (%)</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">Reccall (%)</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">Latence (s)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {processOrder.map(proc => {
                  const avg = results[proc]?.average;
                  if (!avg) return null;
                  return (
                    <tr key={proc} className="hover:bg-gray-50">
                      <td className="px-4 py-3 font-medium text-gray-900">{proc}</td>
                      <td className="px-4 py-3 text-center">{Math.round(avg.quality)}%</td>
                      <td className="px-4 py-3 text-center">{Math.round(avg.faithfulness )}%</td>
                      <td className="px-4 py-3 text-center">{Math.round(avg.retrieval_precision )}%</td>
                      <td className="px-4 py-3 text-center">{Math.round(avg.retrieval_recall_at_5 )}%</td>
                      <td className="px-4 py-3 text-center">{avg.latency_seconds.toFixed(2)} s</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="h-96 flex flex-col items-center justify-center bg-gradient-to-b from-gray-50 to-white rounded-lg border-2 border-dashed border-green-200">
          <CheckCircle className="w-12 h-12 text-green-200 mb-3" />
          <p className="text-gray-500 font-medium">Aucune évaluation disponible</p>
          <p className="text-sm text-gray-400 mt-1">Cliquez sur "Lancer l'évaluation" pour comparer les processus</p>
        </div>
      )}
    </div>
  );
};