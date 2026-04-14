import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';
import { BenchmarkSection } from './components/BenchmarkSection';
//import { StepEvaluationSection } from './components/StepEvaluationSection';
import { EvaluationDashboard } from './components/EvaluationDashboard';
import { Message, ProcessType } from './types';
import { BenchmarkResults } from './types/types';
import { processResponses } from './data/mockData';
import { EvaluationResults } from './types/types';

type TabType = 'chat' | 'benchmark' | 'evaluation';

function App() {
  const [selectedProcess, setSelectedProcess] = useState<ProcessType>('LLM simple');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<TabType>('chat');

  // États pour le benchmark (persistants)
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResults | null>(null);
  const [isBenchmarkRunning, setIsBenchmarkRunning] = useState<boolean>(false);

  //état pour l'évaluation
  const [evaluationResults, setEvaluationResults] = useState<EvaluationResults | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const handleSendMessage = async (content: string): Promise<void> => {
    // ... (identique à avant)
    const userMessage: Message = { role: 'user', content };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    if (selectedProcess === 'LLM simple') {
      try {
        const response = await fetch('/api/llm-simple', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: content }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Erreur inconnue');
        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: data.response,
          sources: data.sources || []
        }]);
      } catch (error: any) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erreur : ${error.message}` }]);
      } finally {
        setIsLoading(false);
      }
      return;
    }

    if (selectedProcess === 'RAG') {
      try {
        const response = await fetch('/api/rag-simple', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: content }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Erreur inconnue');
        const assistantMessage: Message = { role: 'assistant', content: data.response };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (error: any) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erreur : ${error.message}` }]);
      } finally {
        setIsLoading(false);
      }
      return;
    }

    if (selectedProcess === 'RAG optimisé') {
      try {
        const response = await fetch('/api/rag-optimized', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: content }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Erreur inconnue');
        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: data.response,
          sources: data.sources || []
        }]);
      } catch (error: any) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erreur : ${error.message}` }]);
      } finally {
        setIsLoading(false);
      }
      return;
    }

    if (selectedProcess === 'RAG + Agent IA') {
      try {
        const response = await fetch('/api/rag-agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: content }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Erreur inconnue');

        // L’agent indique combien d’outils il a utilisés
        const toolsNote = data.raft_stats?.react_iterations
          ? `Agent IA \n\n`
          : '';

        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: `${toolsNote}${data.response}`,
          sources: data.oracle_docs?.map((d: any) => d.source) || data.sources || [],
          raftResult: data,
          processName: 'RAG + Agent IA',
        }]);
      } catch (error: any) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erreur Agent : ${error.message}` }]);
      } finally {
        setIsLoading(false);
      }
      return;
    }

    if (selectedProcess === 'RAG fine-tuné (RAFT)') {
      try {
        const response = await fetch('/api/raft', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: content, num_oracle: 2, num_distractors: 2 }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Erreur inconnue');

        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: data.response,
          sources: data.oracle_docs?.map((d: any) => d.source) || [],
          raftResult: data,
          processName: 'RAG fine-tuné (RAFT)',
        }]);
      } catch (error: any) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erreur RAFT : ${error.message}` }]);
      } finally {
        setIsLoading(false);
      }
      return;
    }

    if (selectedProcess === 'RAG + Multi-agents') {
      try {
        const response = await fetch('/api/rag-multi-agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: content }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Erreur inconnue');

        // En-tête lisible avec nom du modèle et statistiques
        const model = data.model_used ?? 'Modèle IA';
        const loops = data.loops_count ?? 0;
        const latency = data.latency_seconds ? `${data.latency_seconds}s` : '';
        const validated = data.context_passed ? 'Contexte validé' : 'Contexte partiel';
        const maHeader = `Multi-agents\n\n`;

        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: `${maHeader}${data.response}`,
          sources: data.oracle_docs?.map((d: any) => d.source) || data.sources || [],
          raftResult: data,
          processName: 'RAG + Multi-agents',
        }]);
      } catch (error: any) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erreur Multi-Agent : ${error.message}` }]);
      } finally {
        setIsLoading(false);
      }
      return;
    }

    // Autres processus : simulation
    const delays: Record<ProcessType, number> = {
      'LLM simple': 800,
      'RAG': 1500,
      'RAG optimisé': 2000,
      'RAG fine-tuné (RAFT)': 2500,
      'RAG + Agent IA': 3500,
      'RAG + Multi-agents': 4500,
    };
    setTimeout(() => {
      setMessages((prev) => [...prev, { role: 'assistant', content: processResponses[selectedProcess] }]);
      setIsLoading(false);
    }, delays[selectedProcess]);
  };

  const handleProcessChange = (process: ProcessType) => setSelectedProcess(process);
  const handleNewChat = () => setMessages([]);

  const runEvaluation = async () => {
    setIsEvaluating(true);
    try {
      const res = await fetch('/api/evaluations?sample_size=10'); //évaluations sur 10 questions
      const data = await res.json();
      setEvaluationResults(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsEvaluating(false);
    }
  };

  // Dans le rendu, passer evaluationResults et runEvaluation à l'onglet evaluation
  {
    activeTab === 'evaluation' && (
      <EvaluationDashboard
        results={evaluationResults}
        onRunEvaluation={runEvaluation}
        isEvaluating={isEvaluating}
      />
    )
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-green-50 via-white to-blue-50">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} selectedProcess={selectedProcess} />
      <main className="flex-1 overflow-hidden flex flex-col">
        <div className="h-full overflow-y-auto">
          <div className={`mx-auto ${activeTab === 'chat' ? 'p-4 max-w-full px-6' : 'p-8 max-w-6xl'}`}>
            <header className={`${activeTab === 'chat' ? 'mb-4' : 'mb-8'}`}>
              <h1 className="text-4xl font-bold text-gray-900 mb-1">🇸🇳 FinChat Senegal 🇸🇳</h1>
              <p className="text-gray-600">Comparaison des différentes architectures IA pour l'assistance financière</p>
            </header>

            {activeTab === 'chat' && (
              <div className="h-[calc(100vh-160px)]">
                <ChatInterface
                  messages={messages}
                  onSendMessage={handleSendMessage}
                  isLoading={isLoading}
                  currentProcess={selectedProcess}
                  onProcessChange={handleProcessChange}
                  onNewChat={handleNewChat}
                />
              </div>
            )}

            {activeTab === 'benchmark' && (
              <BenchmarkSection
                results={benchmarkResults}
                setResults={setBenchmarkResults}
                isRunning={isBenchmarkRunning}
                setIsRunning={setIsBenchmarkRunning}
              />
            )}

            {activeTab === 'evaluation' && (
              <EvaluationDashboard
                results={evaluationResults}
                onRunEvaluation={runEvaluation}
                isEvaluating={isEvaluating}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;