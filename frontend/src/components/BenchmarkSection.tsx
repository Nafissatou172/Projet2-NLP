import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { BenchmarkResults } from '../types';
import { benchmarkData } from '../data/mockData';
import { Zap } from 'lucide-react';

export const BenchmarkSection: React.FC = () => {
  const [results, setResults] = useState<BenchmarkResults | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);

  const runBenchmark = (): void => {
    setIsRunning(true);
    setTimeout(() => {
      setResults(benchmarkData);
      setIsRunning(false);
    }, 2000);
  };

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
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Métriques de qualité</h3>
            <div className="h-80 bg-gradient-to-b from-gray-50 to-white rounded-lg p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={results.metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="processus"
                    angle={-15}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                  />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '2px solid #10b981',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Bar dataKey="précision" fill="#10b981" name="Précision (%)" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="pertinence" fill="#059669" name="Pertinence (%)" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="fidélité" fill="#047857" name="Fidélité (%)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Temps de réponse</h3>
            <div className="h-64 bg-gradient-to-b from-gray-50 to-white rounded-lg p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={results.metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="processus"
                    angle={-15}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                  />
                  <YAxis stroke="#9ca3af" label={{ value: 'Secondes', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '2px solid #ef4444',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Bar dataKey="tempsDeRéponse" fill="#ef4444" name="Temps de réponse (s)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
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
