import React, { useState } from 'react'
import { ShieldAlert, Terminal, AlertTriangle, CheckCircle2, Eye } from 'lucide-react'
import type { RuntimeEvent } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { JsonViewerModal } from '../components/JsonViewerModal'

interface ThreatsTabProps {
  events: RuntimeEvent[]
}

export const ThreatsTab: React.FC<ThreatsTabProps> = ({ events }) => {
  const [selectedThreat, setSelectedThreat] = useState<RuntimeEvent | null>(null)

  // Filter security detection events
  const threatEvents = events.filter(
    (e) =>
      e.event_type === 'SECURITY_DETECTION' ||
      (e.event_type === 'SECURITY_ANALYSIS' && (e.detected === true || e.status === 'detected'))
  )

  const promptThreats = threatEvents.filter(
    (e) =>
      e.attack_type?.includes('PROMPT') ||
      (typeof e.input === 'object' && e.input?.attack_type?.includes('PROMPT')) ||
      e.reason?.toLowerCase().includes('prompt')
  )

  const toolThreats = threatEvents.filter(
    (e) =>
      e.attack_type?.includes('TOOL') ||
      (typeof e.input === 'object' && e.input?.attack_type?.includes('TOOL')) ||
      e.tool !== null ||
      e.reason?.toLowerCase().includes('tool')
  )

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 uppercase tracking-wider mb-1">
            <span>Total Threats Flagged</span>
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-rose-400">{threatEvents.length}</div>
          <p className="text-[11px] text-slate-500 mt-1">Rule-based pattern matching detections</p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 uppercase tracking-wider mb-1">
            <span>Prompt Injection Attempts</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400">{promptThreats.length}</div>
          <p className="text-[11px] text-slate-500 mt-1">Jailbreaks, system instruction overrides</p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 uppercase tracking-wider mb-1">
            <span>Tool Misuse Detections</span>
            <Terminal className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-cyan-400">{toolThreats.length}</div>
          <p className="text-[11px] text-slate-500 mt-1">Directory traversal, path escapes, invalid params</p>
        </div>
      </div>

      {/* Threats Feed / Table */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800 overflow-hidden shadow-sm">
        <div className="p-4 border-b border-slate-800 bg-slate-950/60 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-semibold text-slate-100">Deterministic Threat Detection Log</h3>
          </div>
          <span className="text-xs text-slate-400 font-mono">{threatEvents.length} detected instances</span>
        </div>

        <div className="divide-y divide-slate-800/80">
          {threatEvents.length === 0 ? (
            <div className="p-12 text-center text-slate-500 flex flex-col items-center justify-center">
              <CheckCircle2 className="w-10 h-10 text-emerald-500/50 mb-2" />
              <p className="text-sm font-medium text-slate-400">No rule-based threats detected.</p>
              <p className="text-xs text-slate-600 mt-1 max-w-sm">
                The agent has not encountered any prompt injection patterns or unauthorized tool invocations in the current logs.
              </p>
            </div>
          ) : (
            threatEvents.map((threat) => {
              const attackType =
                threat.attack_type ||
                (typeof threat.input === 'object' ? threat.input?.attack_type : null) ||
                'RULE_DETECTION'
              const severity =
                threat.severity ||
                (typeof threat.input === 'object' ? threat.input?.severity : null) ||
                'HIGH'
              const confidence =
                threat.confidence ??
                (typeof threat.input === 'object' ? threat.input?.confidence : 1.0)
              const reason =
                threat.reason ||
                (typeof threat.input === 'object' ? threat.input?.reason : '') ||
                'Suspicious signature matched'

              return (
                <div
                  key={threat.event_id}
                  className="p-4 hover:bg-slate-800/30 transition-colors flex flex-col sm:flex-row sm:items-start justify-between gap-4"
                >
                  <div className="space-y-1.5 flex-1">
                    <div className="flex items-center space-x-2.5">
                      <span className="font-mono text-xs font-bold text-amber-300 bg-amber-950/60 border border-amber-800/60 px-2 py-0.5 rounded">
                        {attackType}
                      </span>
                      <StatusBadge type="risk" value={severity} />
                      <span className="text-[11px] text-slate-400 font-mono">
                        Conf: {(Number(confidence) * 100).toFixed(0)}%
                      </span>
                      <span className="text-[11px] text-slate-500">
                        {new Date(threat.timestamp).toLocaleString()}
                      </span>
                    </div>

                    <p className="text-xs text-slate-200 font-medium">{reason}</p>

                    {threat.tool && (
                      <div className="text-[11px] text-slate-400 flex items-center space-x-1">
                        <span>Target Tool:</span>
                        <span className="font-mono text-cyan-400 bg-slate-800 px-1.5 py-0.2 rounded">
                          {threat.tool}
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setSelectedThreat(threat)}
                      className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded-lg border border-slate-700 transition-colors flex items-center space-x-1.5"
                    >
                      <Eye className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Inspect Evidence</span>
                    </button>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* Threat Inspection Modal */}
      {selectedThreat && (
        <JsonViewerModal
          title={`Threat Evidence: ${selectedThreat.event_id.slice(0, 8)}`}
          data={selectedThreat}
          onClose={() => setSelectedThreat(null)}
        />
      )}
    </div>
  )
}
