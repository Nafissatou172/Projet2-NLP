import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';
import { BenchmarkSection } from './components/BenchmarkSection';
import { StepEvaluationSection } from './components/StepEvaluationSection';
import { Message, ProcessType } from './types';
import { processResponses } from './data/mockData';

type TabType = 'chat' | 'benchmark' | 'evaluation';

function App() {
  const [selectedProcess, setSelectedProcess] = useState<ProcessType>('LLM simple');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<TabType>('chat');

  const handleSendMessage = (content: string): void => {
    const userMessage: Message = { role: 'user', content };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    const baseDelay = selectedProcess === 'LLM simple' ? 800 :
                     selectedProcess === 'RAG' ? 1500 :
                     selectedProcess === 'RAG optimisé' ? 2000 :
                     selectedProcess === 'RAG fine-tuné (RAFT)' ? 2500 :
                     selectedProcess === 'RAG + Agent IA' ? 3500 : 4500;

    setTimeout(() => {
      const assistantMessage: Message = {
        role: 'assistant',
        content: processResponses[selectedProcess],
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsLoading(false);
    }, baseDelay);
  };

  const handleProcessChange = (process: ProcessType): void => {
    setSelectedProcess(process);
  };

  const handleNewChat = (): void => {
    setMessages([]);
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-green-50 via-white to-blue-50">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        selectedProcess={selectedProcess}
      />

      <main className="flex-1 overflow-hidden flex flex-col">
        <div className="h-full overflow-y-auto">
          <div className={`mx-auto ${activeTab === 'chat' ? 'p-4 max-w-full px-6' : 'p-8 max-w-6xl'}`}>
            <header className={`${activeTab === 'chat' ? 'mb-4' : 'mb-8'}`}>
              <h1 className="text-4xl font-bold text-gray-900 mb-1">
                🇸🇳 FinChat Senegal 🇸🇳
              </h1>
              <p className="text-gray-600">
                Comparaison des différentes architectures IA pour l'assistance financière
              </p>
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
              <div>
                <BenchmarkSection />
              </div>
            )}

            {activeTab === 'evaluation' && (
              <div>
                <StepEvaluationSection currentProcess={selectedProcess} />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
