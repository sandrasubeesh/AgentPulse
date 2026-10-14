import React, { useState } from 'react'
import { ShieldCheck, Eye } from 'lucide-react'
import type { SessionSummary } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { JsonViewerModal } from '../components/JsonViewerModal'

interface RiskTabProps {
  sessions: SessionSummary[]
}

export const RiskTab: React.FC<RiskTabProps> = ({ sessions }) => {
  const [selectedSession, setSelectedSession] = useState<SessionSummary | null>(null)

  return (
    <div className="space-y-6">
      {/* Risk Engine Model & Formula Card */}
      <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
        <div className="flex items-center space-x-2 mb-3">
          <ShieldCheck className="w-5 h-5 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-100">Deterministic Multi-Signal Risk Scoring Engine</h3>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed mb-4">
          AgentPulse computes an interpretable composite risk score (0–100) aggregating evidence across three independent detection layers. The score is strictly deterministic and clamped to [0, 100].
        </p>

        {/* 4 Score Factors */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-300">Rule Detection</span>
              <span className="font-mono text-xs font-bold text-amber-400">+30 pts</span>
            </div>
            <p className="text-[11px] text-slate-400">At least one prompt injection or tool misuse pattern confirmed.</p>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-300">Isolation Forest</span>
              <span className="font-mono text-xs font-bold text-purple-400">+25 pts</span>
            </div>
            <p className="text-[11px] text-slate-400">Unsupervised behavioral anomaly detected against baseline.</p>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-300">Random Forest</span>
              <span className="font-mono text-xs font-bold text-cyan-400">+25 pts</span>
            </div>
            <p className="text-[11px] text-slate-400">Supervised model predicts session behavior is SUSPICIOUS.</p>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-300">Triple Agreement</span>
              <span className="font-mono text-xs font-bold text-rose-400">+10 pts</span>
            </div>
            <p className="text-[11px] text-slate-400">Bonus points when all three independent detectors indicate threat.</p>
          </div>
        </div>

        {/* Thresholds Table */}
        <div className="mt-4 pt-4 border-t border-slate-800/80 flex flex-wrap gap-4 text-xs">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
            <span className="text-slate-300">0–29: LOW Risk &rarr; Action: <strong className="text-emerald-400">ALLOW</strong></span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
            <span className="text-slate-300">30–59: MEDIUM Risk &rarr; Action: <strong className="text-amber-400">ALERT</strong></span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-400" />
            <span className="text-slate-300">60–100: HIGH Risk &rarr; Action: <strong className="text-rose-400">HUMAN_REVIEW</strong></span>
          </div>
        </div>
      </div>

      {/* Sessions Risk Table */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800 overflow-hidden shadow-sm">
        <div className="p-4 border-b border-slate-800 bg-slate-950/60 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            <span>Session Risk Assessments</span>
          </h3>
          <span className="text-xs text-slate-400 font-mono">{sessions.length} sessions evaluated</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-950/40 text-slate-400 uppercase text-[11px] tracking-wider">
                <th className="py-3 px-4 font-semibold">Session Prompt</th>
                <th className="py-3 px-4 font-semibold">Timestamp</th>
                <th className="py-3 px-4 font-semibold">Risk Score</th>
                <th className="py-3 px-4 font-semibold">Risk Level</th>
                <th className="py-3 px-4 font-semibold">Response Action</th>
                <th className="py-3 px-4 font-semibold">Correlated</th>
                <th className="py-3 px-4 font-semibold text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {sessions.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-slate-500">
                    No sessions recorded yet.
                  </td>
                </tr>
              ) : (
                sessions.map((s) => (
                  <tr key={s.session_id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 max-w-xs truncate text-slate-200 font-medium">
                      "{s.prompt}"
                    </td>
                    <td className="py-3 px-4 text-slate-400 font-mono whitespace-nowrap text-[11px]">
                      {new Date(s.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-3 px-4 font-mono font-bold">
                      <span
                        className={
                          s.risk_score >= 60
                            ? 'text-rose-400'
                            : s.risk_score >= 30
                            ? 'text-amber-400'
                            : 'text-emerald-400'
                        }
                      >
                        {s.risk_score}/100
                      </span>
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap">
                      <StatusBadge type="risk" value={s.risk_level} />
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap">
                      <StatusBadge type="action" value={s.response_action} />
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap">
                      {s.correlated ? (
                        <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800/60 text-[10px] font-semibold">
                          CORRELATED
                        </span>
                      ) : (
                        <span className="text-slate-500 font-mono text-[11px]">Single / Clear</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right whitespace-nowrap">
                      <button
                        onClick={() => setSelectedSession(s)}
                        className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 text-[11px] transition-colors inline-flex items-center space-x-1"
                      >
                        <Eye className="w-3 h-3 text-cyan-400" />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {selectedSession && (
        <JsonViewerModal
          title={`Risk Assessment: ${selectedSession.session_id}`}
          data={selectedSession}
          onClose={() => setSelectedSession(null)}
        />
      )}
    </div>
  )
}
