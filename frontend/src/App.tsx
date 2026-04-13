import { useState , useEffect} from 'react';
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
        
        let assistantContent = data.response;
        // Optionnel: On peut afficher le fait qu'il a utilisé des outils
        if (data.tools_used_count > 0) {
            assistantContent = `*(A cherché dans la base de données)*\n\n${assistantContent}`;
        }
        
        setMessages((prev) => [...prev, { 
          role: 'assistant', 
          content: assistantContent,
          sources: data.sources || [] 
        }]);
      } catch (error: any) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erreur : ${error.message}` }]);
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
        
        let assistantContent = data.response;
        // On affiche le feedback
        if (data.loops_count > 0) {
            assistantContent = `*(Le sous-système a effectué ${data.loops_count} itération(s) de recherche avec l'Évaluateur)*\n\n${assistantContent}`;
        }
        
        setMessages((prev) => [...prev, { 
          role: 'assistant', 
          content: assistantContent,
          sources: data.sources || [] 
        }]);
      } catch (error: any) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erreur : ${error.message}` }]);
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
  {activeTab === 'evaluation' && (
    <EvaluationDashboard 
      results={evaluationResults} 
      onRunEvaluation={runEvaluation} 
      isEvaluating={isEvaluating} 
    />
  )}

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