import React from 'react'
import {
  ShieldAlert,
  Activity,
  Layers,
  Brain,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  ArrowRight,
  TrendingUp,
} from 'lucide-react'
import type { MetricsResponse, SessionSummary } from '../types'
import { StatusBadge } from '../components/StatusBadge'

interface OverviewTabProps {
  metrics: MetricsResponse | null
  sessions: SessionSummary[]
  onSelectSession: (sessionId: string) => void
  onNavigateTab: (tab: string) => void
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
  metrics,
  sessions,
  onSelectSession,
  onNavigateTab,
}) => {
  if (!metrics) {
    return (
      <div className="flex items-center justify-center p-12 text-slate-400">
        <div className="flex items-center space-x-3">
          <Activity className="w-5 h-5 animate-spin text-cyan-500" />
          <span>Loading telemetry metrics...</span>
        </div>
      </div>
    )
  }

  const lowRisk = metrics.risk_distribution['LOW'] || 0
  const medRisk = metrics.risk_distribution['MEDIUM'] || 0
  const highRisk = metrics.risk_distribution['HIGH'] || 0
  const total = Math.max(1, metrics.total_sessions)

  const lowPct = Math.round((lowRisk / total) * 100)
  const medPct = Math.round((medRisk / total) * 100)
  const highPct = Math.round((highRisk / total) * 100)

  // Recent alerts (sessions that triggered detections, anomalies, or suspicious ML)
  const alertSessions = sessions.filter(
    (s) => s.rule_detections > 0 || s.risk_score > 0 || s.ml_prediction === 'SUSPICIOUS' || s.correlated
  ).slice(0, 5)

  return (
    <div className="space-y-6">
      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Sessions */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 shadow-sm relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Monitored Sessions</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-white">{metrics.total_sessions}</span>
            <span className="text-xs text-slate-400">({metrics.total_events} raw events)</span>
          </div>
          <div className="mt-3 text-[11px] text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-2">
            <span>Runtime JSONL Active</span>
            <button
              onClick={() => onNavigateTab('events')}
              className="text-cyan-400 hover:text-cyan-300 flex items-center space-x-1"
            >
              <span>View Log</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Rule-Based Threat Detections */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 shadow-sm relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Threat Detections</span>
            <ShieldAlert className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-amber-400">{metrics.rule_detections_count}</span>
            <span className="text-xs text-slate-400">rule flags</span>
          </div>
          <div className="mt-3 text-[11px] text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-2">
            <span>Prompt & Tool Misuse</span>
            <button
              onClick={() => onNavigateTab('threats')}
              className="text-amber-400 hover:text-amber-300 flex items-center space-x-1"
            >
              <span>Inspect</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Behavioral Anomalies */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 shadow-sm relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Behavioral Anomalies</span>
            <Activity className="w-4 h-4 text-purple-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-purple-400">{metrics.anomaly_count}</span>
            <span className="text-xs text-slate-400">Isolation Forest flags</span>
          </div>
          <div className="mt-3 text-[11px] text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-2">
            <span>Unsupervised Baseline</span>
            <button
              onClick={() => onNavigateTab('behavior')}
              className="text-purple-400 hover:text-purple-300 flex items-center space-x-1"
            >
              <span>Behavior</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* ML Supervised Trust Index */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 shadow-sm relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">ML Suspicious Sessions</span>
            <Brain className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-white">{metrics.suspicious_predictions_count}</span>
            <span className="text-xs text-slate-400">Random Forest</span>
          </div>
          <div className="mt-3 text-[11px] text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-2">
            <span>Avg Risk: <strong className="text-cyan-300 font-mono">{metrics.average_risk_score}</strong>/100</span>
            <button
              onClick={() => onNavigateTab('risk')}
              className="text-emerald-400 hover:text-emerald-300 flex items-center space-x-1"
            >
              <span>Risk Matrix</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Center Section: Risk Meter & Distribution Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Distribution Card */}
        <div className="lg:col-span-2 p-5 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              <span>Session Risk Distribution & Response Actions</span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">{metrics.total_sessions} Total Sessions</span>
          </div>

          {/* Stacked Bar */}
          <div className="h-4 w-full rounded-full bg-slate-800 flex overflow-hidden p-0.5 border border-slate-700/60 mb-4">
            <div
              style={{ width: `${lowPct}%` }}
              className="h-full bg-emerald-500 rounded-l transition-all"
              title={`LOW Risk: ${lowRisk} (${lowPct}%)`}
            />
            <div
              style={{ width: `${medPct}%` }}
              className="h-full bg-amber-500 transition-all"
              title={`MEDIUM Risk: ${medRisk} (${medPct}%)`}
            />
            <div
              style={{ width: `${highPct}%` }}
              className="h-full bg-rose-500 rounded-r transition-all"
              title={`HIGH Risk: ${highRisk} (${highPct}%)`}
            />
          </div>

          {/* Category breakdown pills */}
          <div className="grid grid-cols-3 gap-3 text-center mb-6">
            <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/40">
              <div className="flex items-center justify-center space-x-1.5 text-emerald-400 text-xs font-semibold mb-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>LOW (0–29)</span>
              </div>
              <div className="text-xl font-bold font-mono text-white">{lowRisk}</div>
              <div className="text-[11px] text-slate-400">Action: ALLOW ({lowPct}%)</div>
            </div>

            <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/40">
              <div className="flex items-center justify-center space-x-1.5 text-amber-400 text-xs font-semibold mb-1">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>MEDIUM (30–59)</span>
              </div>
              <div className="text-xl font-bold font-mono text-white">{medRisk}</div>
              <div className="text-[11px] text-slate-400">Action: ALERT ({medPct}%)</div>
            </div>

            <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/40">
              <div className="flex items-center justify-center space-x-1.5 text-rose-400 text-xs font-semibold mb-1">
                <AlertOctagon className="w-3.5 h-3.5" />
                <span>HIGH (60–100)</span>
              </div>
              <div className="text-xl font-bold font-mono text-white">{highRisk}</div>
              <div className="text-[11px] text-slate-400">Action: REVIEW ({highPct}%)</div>
            </div>
          </div>

          {/* Event Types Breakdown Mini Table */}
          <div className="border-t border-slate-800/80 pt-4">
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2.5">
              Runtime Telemetry Event Breakdown
            </h4>
            <div className="flex flex-wrap gap-2">
              {Object.entries(metrics.event_type_distribution).map(([etype, count]) => (
                <div
                  key={etype}
                  className="px-2.5 py-1 rounded bg-slate-800/80 border border-slate-700/50 flex items-center space-x-2 text-xs"
                >
                  <span className="font-mono text-slate-300">{etype}:</span>
                  <span className="font-mono font-bold text-cyan-400">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Alerts Feed */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <ShieldAlert className="w-4 h-4 text-amber-400" />
              <span>Flagged Sessions</span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">{alertSessions.length} active</span>
          </div>

          <div className="space-y-3 flex-1 overflow-y-auto max-h-[340px]">
            {alertSessions.length === 0 ? (
              <div className="h-48 flex flex-col items-center justify-center text-center text-slate-500">
                <CheckCircle2 className="w-8 h-8 text-emerald-500/50 mb-2" />
                <span className="text-xs">No flagged threats detected yet.</span>
                <span className="text-[11px] text-slate-600 mt-1">All logged sessions evaluated as clean.</span>
              </div>
            ) : (
              alertSessions.map((s) => (
                <div
                  key={s.session_id}
                  onClick={() => onSelectSession(s.session_id)}
                  className="p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 cursor-pointer transition-all group"
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <StatusBadge type="risk" value={s.risk_level} />
                    <span className="text-[11px] font-mono text-slate-400">Score: {s.risk_score}</span>
                  </div>
                  <p className="text-xs text-slate-200 line-clamp-1 font-medium group-hover:text-cyan-300 transition-colors">
                    "{s.prompt}"
                  </p>
                  <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400">
                    <span>{new Date(s.timestamp).toLocaleTimeString()}</span>
                    <span className="text-cyan-400 flex items-center space-x-0.5">
                      <span>Inspect</span>
                      <ArrowRight className="w-2.5 h-2.5" />
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
