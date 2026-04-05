import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { BenchmarkResults, ChartData } from '../types/types';
import { Zap } from 'lucide-react';

interface BenchmarkSectionProps {
  results: BenchmarkResults | null;
  setResults: (results: BenchmarkResults | null) => void;
  isRunning: boolean;
  setIsRunning: (running: boolean) => void;
}

export const BenchmarkSection: React.FC<BenchmarkSectionProps> = ({
  results,
  setResults,
  isRunning,
  setIsRunning,
}) => {
  const transformToChartData = (data: BenchmarkResults): ChartData[] => {
    return data.results.map(model => ({
      processus: model.display_name,
      précision: Math.round(model.aggregated.quality * 100),
      pertinence: Math.round(model.aggregated.quality * 100),
      fidélité: Math.round(model.aggregated.faithfulness * 100),
      tempsDeRéponse: parseFloat(model.aggregated.latency_avg_seconds.toFixed(2)),
      scoreGlobal: Math.round(model.aggregated.global_score * 100),
    }));
  };

  const runBenchmark = async () => {
    setIsRunning(true);
    try {
      const response = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ models: ['mistral', 'llama', 'qwen'], sample_size: 10 }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: BenchmarkResults = await response.json();
      setResults(data);
    } catch (err: any) {
      console.error(err);
      alert(`Erreur : ${err.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  const chartData = results ? transformToChartData(results) : [];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-green-100">
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-green-100">
        <div className="flex items-center gap-2">
          <Zap className="w-6 h-6 text-green-600" />
          <h2 className="text-xl font-bold text-gray-900">Benchmark - Comparaison des Processus</h2>
        </div>
        <button
          onClick={runBenchmark}
          disabled={isRunning}
          className="px-6 py-2.5 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-lg hover:from-green-700 hover:to-green-600 disabled:from-gray-300 disabled:to-gray-300 disabled:cursor-not-allowed transition-all font-semibold shadow-md hover:shadow-lg"
        >
          {isRunning ? 'Benchmark en cours...' : 'Lancer le benchmark'}
        </button>
      </div>

      {results ? (
        <div className="space-y-8">
          {/* Graphique qualité */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Métriques de qualité</h3>
            <div className="h-80 bg-gradient-to-b from-gray-50 to-white rounded-lg p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="processus" angle={-15} textAnchor="end" height={80} fontSize={12} />
                  <YAxis domain={[0, 100]} stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '2px solid #10b981', borderRadius: '8px' }} />
                  <Legend />
                  <Bar dataKey="précision" fill="#10b981" name="Précision (%)" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="pertinence" fill="#059669" name="Pertinence (%)" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="fidélité" fill="#047857" name="Fidélité (%)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          {/* Graphique temps de réponse */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Temps de réponse (secondes)</h3>
            <div className="h-64 bg-gradient-to-b from-gray-50 to-white rounded-lg p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="processus" angle={-15} textAnchor="end" height={80} fontSize={12} />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '2px solid #ef4444', borderRadius: '8px' }} />
                  <Legend />
                  <Bar dataKey="tempsDeRéponse" fill="#ef4444" name="Temps de réponse (s)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          {/* Score global */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Score global (0-100)</h3>
            <div className="h-64 bg-gradient-to-b from-gray-50 to-white rounded-lg p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="processus" angle={-15} textAnchor="end" height={80} fontSize={12} />
                  <YAxis domain={[0, 100]} stroke="#9ca3af" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="scoreGlobal" fill="#3b82f6" name="Score Global" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="text-sm text-gray-500 text-center mt-4">
            Benchmark exécuté en {results.duration_seconds} secondes sur {results.sample_size} questions.
          </div>
        </div>
      ) : (
        <div className="h-96 flex flex-col items-center justify-center bg-gradient-to-b from-gray-50 to-white rounded-lg border-2 border-dashed border-green-200">
          <Zap className="w-12 h-12 text-green-200 mb-3" />
          <p className="text-gray-500 font-medium">Cliquez sur "Lancer le benchmark" pour comparer les processus</p>
        </div>
      )}
    </div>
  );
};