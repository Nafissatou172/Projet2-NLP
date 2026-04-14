import React, { useState } from 'react';
import { Send, MessageCircle, Zap, PlusCircle, Brain, ChevronDown, ChevronUp, Bot, Users } from 'lucide-react';
import { Message, ProcessType } from '../types';
import { RaftPanel } from './RaftPanel';

/* ─────────────────────────────────────────────
   Config par mode de processus
───────────────────────────────────────────── */
type PanelMode = 'RAG fine-tuné (RAFT)' | 'RAG + Agent IA' | 'RAG + Multi-agents' | string;

const getPanelConfig = (mode: PanelMode) => {
  switch (mode) {
    case 'RAG + Agent IA':
      return {
        label: 'Analyse Agent IA (ReAct)',
        icon: <Bot className="w-4 h-4" />,
        border: 'border-blue-200',
        gradient: 'from-blue-50 to-sky-50 hover:from-blue-100 hover:to-sky-100',
        text: 'text-blue-700',
        chevron: 'text-blue-400',
      };
    case 'RAG + Multi-agents':
      return {
        label: 'Analyse Multi-agents',
        icon: <Users className="w-4 h-4" />,
        border: 'border-amber-200',
        gradient: 'from-amber-50 to-orange-50 hover:from-amber-100 hover:to-orange-100',
        text: 'text-amber-700',
        chevron: 'text-amber-400',
      };
    default: // 'RAG fine-tuné (RAFT)'
      return {
        label: 'Analyse RAFT',
        icon: <Brain className="w-4 h-4" />,
        border: 'border-violet-200',
        gradient: 'from-violet-50 to-indigo-50 hover:from-violet-100 hover:to-indigo-100',
        text: 'text-violet-700',
        chevron: 'text-violet-400',
      };
  }
};

/* ─────────────────────────────────────────────
   RaftAccordion — panneau dépliable adaptatif
───────────────────────────────────────────── */
const RaftAccordion: React.FC<{ result: any; mode?: PanelMode }> = ({ result, mode = 'RAG fine-tuné (RAFT)' }) => {
  const [open, setOpen] = useState(false);
  const cfg = getPanelConfig(mode);

  return (
    <div className={`border-2 ${cfg.border} rounded-xl overflow-hidden shadow-sm`}>
      <button
        className={`w-full flex items-center justify-between px-4 py-2.5 bg-gradient-to-r ${cfg.gradient} transition-colors`}
        onClick={() => setOpen(!open)}
      >
        <div className={`flex items-center gap-2 text-sm font-semibold ${cfg.text}`}>
          {cfg.icon}
          {cfg.label} — {result.raft_stats?.num_oracle_docs ?? 0} oracle
          &nbsp;·&nbsp;{result.raft_stats?.num_distractor_docs ?? 0} distracteurs
          &nbsp;·&nbsp;{result.latency_seconds}s
        </div>
        {open
          ? <ChevronUp className={`w-4 h-4 ${cfg.chevron}`} />
          : <ChevronDown className={`w-4 h-4 ${cfg.chevron}`} />}
      </button>
      {open && (
        <div className="p-4 bg-white">
          <RaftPanel result={result} />
        </div>
      )}
    </div>
  );
};

interface ChatInterfaceProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
  isLoading: boolean;
  currentProcess: ProcessType;
  onProcessChange: (process: ProcessType) => void;
  onNewChat: () => void;
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
  'RAG optimisé': 'RG avec Reranking coss-encoder',
  'RAG fine-tuné (RAFT)': 'RAG avec fine-tuning',
  'RAG + Agent IA': 'RAG avec agent autonome',
  'RAG + Multi-agents': 'Orchestration multi-agents',
};

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ messages, onSendMessage, isLoading, currentProcess, onProcessChange, onNewChat }) => {
  const [input, setInput] = useState<string>('');

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    setInput(event.target.value);
  };

  const handleProcessChange = (event: React.ChangeEvent<HTMLSelectElement>): void => {
    onProcessChange(event.target.value as ProcessType);
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 h-full flex flex-col border border-green-100">
      <div className="flex items-center justify-between mb-4 pb-4 border-b border-green-100">
        <div className="flex items-center gap-2">
          <MessageCircle className="w-6 h-6 text-green-600" />
          <h2 className="text-xl font-bold text-gray-900">Assistant IA en finance personnelle</h2>
        </div>
        <button
          onClick={onNewChat}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-600 to-green-500 text-white text-sm font-semibold rounded-lg hover:from-green-700 hover:to-green-600 transition-all shadow-md hover:shadow-lg"
        >
          <PlusCircle className="w-4 h-4" />
          Nouvelle conversation
        </button>
      </div>

      {/* Process Selector intégré */}
      <div className="mb-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-100">
        <div className="flex items-center gap-2 mb-2">
          <Zap className="w-4 h-4 text-green-600" />
          <label htmlFor="process-select-chat" className="text-sm font-semibold text-gray-800">
            Processus IA
          </label>
        </div>
        <select
          id="process-select-chat"
          value={currentProcess}
          onChange={handleProcessChange}
          className="w-full px-3 py-2 border-2 border-green-200 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white text-gray-900 shadow-sm transition-all text-sm"
        >
          {processes.map((process) => (
            <option key={process} value={process}>
              {process}
            </option>
          ))}
        </select>
        <p className="mt-2 text-xs text-gray-600">
          <span className="font-semibold text-green-700">Description:</span> {processDescriptions[currentProcess]}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto mb-4 space-y-4 p-4 bg-gradient-to-b from-gray-50 to-white rounded-lg">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <MessageCircle className="w-12 h-12 mb-2 text-green-200" />
            <p className="text-center">Posez votre question sur les finances personnelles...</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
            >
              <div
                className={`max-w-[80%] ${message.role === 'user'
                    ? 'px-4 py-3 rounded-xl bg-gradient-to-r from-green-600 to-green-500 text-white shadow-md'
                    : 'w-full'
                  }`}
              >
                {message.role === 'user' ? (
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
                ) : (
                  <div className="space-y-3">
                    {/* Bulle de réponse principale */}
                    <div className="px-4 py-3 rounded-xl bg-gradient-to-r from-gray-100 to-white text-gray-900 border-2 border-green-100 shadow-sm">
                      <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>

                      {/* Sources (non-RAFT) */}
                      {!message.raftResult && message.sources && message.sources.length > 0 && (
                        <div className="mt-3 pt-2 border-t border-green-200">
                          <p className="text-xs font-semibold text-green-700 flex items-center gap-1">
                            📄 Sources utilisées :
                          </p>
                          <ul className="text-xs text-gray-600 mt-1 space-y-1">
                            {message.sources.map((src, idx) => (
                              <li key={idx} className="flex items-start gap-1">
                                <span>•</span>
                                <span className="break-all">{src}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>

                    {/* Panneau analyse dépliable (RAFT / Agent / Multi-agents) */}
                    {message.raftResult && (
                      <RaftAccordion result={message.raftResult} mode={message.processName} />
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[75%] px-4 py-3 rounded-xl bg-gradient-to-r from-gray-100 to-white border-2 border-green-100">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-green-600 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-green-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-green-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 mt-auto">
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          placeholder="Tapez votre question..."
          disabled={isLoading}
          className="flex-1 px-4 py-3 border-2 border-green-200 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 disabled:bg-gray-100 disabled:cursor-not-allowed bg-white text-gray-900 transition-all"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-6 py-3 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-lg hover:from-green-700 hover:to-green-600 disabled:from-gray-300 disabled:to-gray-300 disabled:cursor-not-allowed flex items-center gap-2 transition-all font-semibold shadow-md hover:shadow-lg"
        >
          <Send size={18} />
          Envoyer
        </button>
      </form>
    </div>
  );
};
