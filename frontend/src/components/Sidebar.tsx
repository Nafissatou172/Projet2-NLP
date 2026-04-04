import React from 'react';
import { MessageCircle, BarChart3, CheckCircle } from 'lucide-react';
import { ProcessType } from '../types';

type TabType = 'chat' | 'benchmark' | 'evaluation';

interface SidebarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  selectedProcess: ProcessType;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange, selectedProcess }) => {
  const tabs: Array<{ id: TabType; label: string; icon: React.ReactNode }> = [
    { id: 'chat', label: 'Chat', icon: <MessageCircle className="w-5 h-5" /> },
    { id: 'benchmark', label: 'Benchmark', icon: <BarChart3 className="w-5 h-5" /> },
    { id: 'evaluation', label: 'Évaluation', icon: <CheckCircle className="w-5 h-5" /> },
  ];

  return (
    <aside className="w-64 bg-gradient-to-b from-green-600 to-green-700 text-white h-screen shadow-lg flex flex-col">
      <div className="p-6 border-b border-green-500">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center">
            <span className="text-green-600 font-bold">₹</span>
          </div>
          FinChat-SN
        </h2>
        <p className="text-sm text-green-100 mt-2">Processus: {selectedProcess}</p>
      </div>

      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-white text-green-600 shadow-lg font-semibold'
                  : 'text-green-100 hover:bg-green-500 hover:text-white'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </nav>

      <div className="p-4 border-t border-green-500">
        <div className="bg-green-500 bg-opacity-50 rounded-lg p-3 text-sm text-green-50">
          <p className="font-semibold mb-1">Conseil du jour</p>
          <p>Diversifiez votre portefeuille pour réduire les risques.</p>
        </div>
      </div>
    </aside>
  );
};
