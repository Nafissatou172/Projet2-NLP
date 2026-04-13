import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Brain, FileText, AlertTriangle, CheckCircle2, Zap, Clock, Hash } from 'lucide-react';

/* ─────────────────────────────────────────────
   Types
───────────────────────────────────────────── */
interface DocItem {
  text: string;
  source: string;
}

interface RaftStats {
  num_oracle_docs: number;
  num_distractor_docs: number;
  total_docs_presented: number;
  oracle_ratio: number;
  has_reasoning: boolean;
  has_structured_response: boolean;
}

export interface RaftResult {
  question: string;
  response: string;
  reasoning: string;
  model_used: string;
  oracle_docs: DocItem[];
  distractor_docs: DocItem[];
  raft_stats: RaftStats;
  latency_seconds: number;
  input_tokens: number;
  output_tokens: number;
}

interface RaftPanelProps {
  result: RaftResult;
}

/* ─────────────────────────────────────────────
   Sub-components
───────────────────────────────────────────── */
const Pill: React.FC<{ label: string; value: string | number; color: string }> = ({ label, value, color }) => (
  <div className={`flex flex-col items-center px-4 py-2 rounded-xl ${color}`}>
    <span className="text-xs font-semibold opacity-70 uppercase tracking-wide">{label}</span>
    <span className="text-lg font-bold">{value}</span>
  </div>
);

const Collapsible: React.FC<{ title: string; icon: React.ReactNode; defaultOpen?: boolean; children: React.ReactNode; accent?: string }> = ({
  title, icon, defaultOpen = false, children, accent = 'border-gray-200',
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`border-2 ${accent} rounded-xl overflow-hidden`}>
      <button
        className="w-full flex items-center justify-between px-4 py-3 bg-white hover:bg-gray-50 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2 font-semibold text-gray-800 text-sm">
          {icon}
          {title}
        </div>
        {open ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
      </button>
      {open && <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">{children}</div>}
    </div>
  );
};

/* ─────────────────────────────────────────────
   Main RaftPanel component
───────────────────────────────────────────── */
export const RaftPanel: React.FC<RaftPanelProps> = ({ result }) => {
  const { raft_stats: s } = result;

  return (
    <div className="space-y-4 text-sm">

      {/* ── Header stats bar ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Pill label="Docs oracle" value={s.num_oracle_docs}       color="bg-emerald-50 text-emerald-700" />
        <Pill label="Distracteurs" value={s.num_distractor_docs}   color="bg-rose-50 text-rose-700" />
        <Pill label="Ratio oracle" value={`${Math.round(s.oracle_ratio * 100)}%`} color="bg-blue-50 text-blue-700" />
        <Pill label="Latence"       value={`${result.latency_seconds}s`}           color="bg-purple-50 text-purple-700" />
      </div>

      {/* ── Meta badges ── */}
      <div className="flex flex-wrap gap-2">
        <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-700">
          <Zap className="w-3 h-3" /> {result.model_used}
        </span>
        <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-700">
          <Hash className="w-3 h-3" /> {result.input_tokens} → {result.output_tokens} tokens
        </span>
        <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-700">
          <Clock className="w-3 h-3" /> {s.total_docs_presented} docs présentés
        </span>
        {s.has_structured_response && (
          <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-700">
            <CheckCircle2 className="w-3 h-3" /> Réponse structurée
          </span>
        )}
        {s.has_reasoning && (
          <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold bg-violet-100 text-violet-700">
            <Brain className="w-3 h-3" /> Raisonnement CoT
          </span>
        )}
      </div>

      {/* ── Chain-of-Thought reasoning ── */}
      {result.reasoning && (
        <Collapsible
          title="Raisonnement Chain-of-Thought"
          icon={<Brain className="w-4 h-4 text-violet-600" />}
          defaultOpen={true}
          accent="border-violet-200"
        >
          <pre className="whitespace-pre-wrap font-mono text-xs text-gray-700 leading-relaxed max-h-64 overflow-y-auto">
            {result.reasoning}
          </pre>
        </Collapsible>
      )}

      {/* ── Oracle documents ── */}
      {result.oracle_docs && result.oracle_docs.length > 0 && (
        <Collapsible
          title={`Documents Oracle (${result.oracle_docs.length}) — Sources pertinentes`}
          icon={<CheckCircle2 className="w-4 h-4 text-emerald-600" />}
          defaultOpen={true}
          accent="border-emerald-200"
        >
          <div className="space-y-3">
            {result.oracle_docs.map((doc, i) => (
              <div key={i} className="p-3 bg-white rounded-lg border border-emerald-100 shadow-sm">
                <div className="flex items-center gap-1 mb-1">
                  <FileText className="w-3 h-3 text-emerald-500" />
                  <span className="text-xs font-semibold text-emerald-700 truncate">{doc.source}</span>
                  <span className="ml-auto text-xs text-gray-400">#{i + 1}</span>
                </div>
                <p className="text-xs text-gray-600 leading-relaxed line-clamp-4">{doc.text}</p>
              </div>
            ))}
          </div>
        </Collapsible>
      )}

      {/* ── Distractor documents ── */}
      {result.distractor_docs && result.distractor_docs.length > 0 && (
        <Collapsible
          title={`Distracteurs (${result.distractor_docs.length}) — Documents ignorés`}
          icon={<AlertTriangle className="w-4 h-4 text-rose-500" />}
          accent="border-rose-200"
        >
          <div className="space-y-3">
            {result.distractor_docs.map((doc, i) => (
              <div key={i} className="p-3 bg-white rounded-lg border border-rose-100 shadow-sm">
                <div className="flex items-center gap-1 mb-1">
                  <FileText className="w-3 h-3 text-rose-400" />
                  <span className="text-xs font-semibold text-rose-600 truncate">{doc.source}</span>
                  <span className="ml-auto text-xs text-gray-400">#{i + 1}</span>
                </div>
                <p className="text-xs text-gray-500 leading-relaxed line-clamp-3 opacity-70">{doc.text}</p>
              </div>
            ))}
          </div>
        </Collapsible>
      )}
    </div>
  );
};
