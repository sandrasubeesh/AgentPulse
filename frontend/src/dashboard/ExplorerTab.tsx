import React, { useState, useEffect } from 'react'
import {
  Layers,
  Code2,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Terminal,
  Activity,
  Brain,
  ShieldAlert,
  ShieldCheck,
  Search,
} from 'lucide-react'
import type { SessionSummary, RuntimeEvent, SessionDetailResponse } from '../types'
import { api } from '../api'
import { StatusBadge } from '../components/StatusBadge'
import { JsonViewerModal } from '../components/JsonViewerModal'

interface ExplorerTabProps {
  sessions: SessionSummary[]
  selectedSessionId: string | null
  onSelectSessionId: (id: string) => void
}

export const ExplorerTab: React.FC<ExplorerTabProps> = ({
  sessions,
  selectedSessionId,
  onSelectSessionId,
}) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [sessionDetail, setSessionDetail] = useState<SessionDetailResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [inspectEvent, setInspectEvent] = useState<RuntimeEvent | null>(null)

  const activeId = selectedSessionId || (sessions.length > 0 ? sessions[0].session_id : null)

  useEffect(() => {
    if (!activeId) {
      setSessionDetail(null)
      return
    }

    setLoading(true)
    api
      .getSessionDetail(activeId)
      .then((detail) => {
        setSessionDetail(detail)
      })
      .catch((err) => {
        console.error('Failed to load session details:', err)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [activeId])

  const filteredSessions = sessions.filter((s) => {
    if (!searchTerm) return true
    const term = searchTerm.toLowerCase()
    return (
      s.prompt.toLowerCase().includes(term) ||
      s.session_id.toLowerCase().includes(term) ||
      s.risk_level.toLowerCase().includes(term)
    )
  })

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'USER_REQUEST':
        return <Layers className="w-4 h-4 text-cyan-400" />
      case 'SECURITY_ANALYSIS':
      case 'SECURITY_DETECTION':
        return <ShieldAlert className="w-4 h-4 text-amber-400" />
      case 'TOOL_CALL':
      case 'TOOL_RESULT':
      case 'TOOL_ERROR':
        return <Terminal className="w-4 h-4 text-sky-400" />
      case 'AGENT_RESPONSE':
        return <CheckCircle2 className="w-4 h-4 text-emerald-400" />
      case 'BEHAVIORAL_ANOMALY':
        return <Activity className="w-4 h-4 text-purple-400" />
      case 'ML_BEHAVIOR_PREDICTION':
        return <Brain className="w-4 h-4 text-teal-400" />
      case 'SECURITY_CORRELATION':
        return <ShieldCheck className="w-4 h-4 text-indigo-400" />
      case 'RISK_ASSESSMENT':
        return <AlertTriangle className="w-4 h-4 text-rose-400" />
      default:
        return <Clock className="w-4 h-4 text-slate-400" />
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[600px]">
      {/* Left Column: Sessions List */}
      <div className="lg:col-span-4 rounded-xl bg-slate-900/80 border border-slate-800 p-4 flex flex-col h-[700px]">
        <div className="mb-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
              Sessions ({sessions.length})
            </h3>
            <span className="text-[11px] text-slate-400">Click to replay</span>
          </div>
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Filter sessions..."
              className="w-full pl-8 pr-3 py-1.5 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto space-y-2 pr-1">
          {filteredSessions.length === 0 ? (
            <div className="p-8 text-center text-xs text-slate-500">No sessions match filter.</div>
          ) : (
            filteredSessions.map((s) => {
              const isSelected = s.session_id === activeId
              return (
                <div
                  key={s.session_id}
                  onClick={() => onSelectSessionId(s.session_id)}
                  className={`p-3 rounded-lg border transition-all cursor-pointer text-left ${
                    isSelected
                      ? 'bg-slate-800 border-cyan-500/80 shadow-sm shadow-cyan-950'
                      : 'bg-slate-950/40 border-slate-800/80 hover:bg-slate-850'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <StatusBadge type="risk" value={s.risk_level} />
                    <span className="text-[11px] font-mono text-slate-400">
                      Score: <strong className="text-white">{s.risk_score}</strong>
                    </span>
                  </div>
                  <p className="text-xs text-slate-200 font-medium line-clamp-2">"{s.prompt}"</p>
                  <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500">
                    <span>{new Date(s.timestamp).toLocaleTimeString()}</span>
                    <span>{s.event_count} events</span>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* Right Column: Replay Timeline */}
      <div className="lg:col-span-8 rounded-xl bg-slate-900/80 border border-slate-800 p-5 flex flex-col h-[700px]">
        {loading ? (
          <div className="flex-1 flex items-center justify-center text-slate-400 space-x-2">
            <Activity className="w-5 h-5 animate-spin text-cyan-400" />
            <span className="text-xs">Loading session event timeline...</span>
          </div>
        ) : !sessionDetail ? (
          <div className="flex-1 flex items-center justify-center text-slate-500 text-xs">
            Select a session from the left pane to explore its full event timeline.
          </div>
        ) : (
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Header info */}
            <div className="border-b border-slate-800 pb-3 mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-xs text-cyan-400 font-bold">
                    Session: {sessionDetail.summary.session_id.slice(0, 16)}...
                  </span>
                  <StatusBadge type="action" value={sessionDetail.summary.response_action} />
                </div>
                <p className="text-xs text-slate-300 mt-1 line-clamp-1 italic">
                  "{sessionDetail.summary.prompt}"
                </p>
              </div>
              <div className="flex items-center space-x-3 text-xs text-slate-400">
                <span>Risk Score: <strong className="text-white font-mono">{sessionDetail.summary.risk_score}</strong></span>
                <span>Events: <strong className="text-white font-mono">{sessionDetail.events.length}</strong></span>
              </div>
            </div>

            {/* Chronological Event Steps */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-2">
              {sessionDetail.events.map((ev, index) => {
                const payloadStr =
                  typeof ev.input === 'string'
                    ? ev.input
                    : typeof ev.input === 'object'
                    ? JSON.stringify(ev.input)
                    : String(ev.input ?? '')

                return (
                  <div
                    key={ev.event_id}
                    className="flex items-start space-x-3 p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 hover:border-slate-700 transition-colors"
                  >
                    {/* Step indicator */}
                    <div className="flex flex-col items-center">
                      <div className="w-7 h-7 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center shrink-0">
                        {getEventIcon(ev.event_type)}
                      </div>
                      <span className="text-[10px] font-mono text-slate-500 mt-1">#{index + 1}</span>
                    </div>

                    {/* Step details */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center space-x-2">
                          <span className="text-xs font-bold font-mono text-slate-200">
                            {ev.event_type}
                          </span>
                          {ev.tool && (
                            <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] font-mono text-cyan-300">
                              tool: {ev.tool}
                            </span>
                          )}
                          <StatusBadge type="status" value={ev.status || 'success'} />
                        </div>
                        <span className="text-[10px] font-mono text-slate-500">
                          {new Date(ev.timestamp).toLocaleTimeString()}
                        </span>
                      </div>

                      {/* Content preview */}
                      <p className="text-xs text-slate-300 line-clamp-3 font-sans break-words bg-slate-900/60 p-2 rounded border border-slate-800/50 mt-1">
                        {payloadStr || <span className="text-slate-600 italic">No payload</span>}
                      </p>

                      {/* Extra info flags if present */}
                      {(ev.attack_type || ev.anomaly_score !== undefined || ev.predicted_label || ev.risk_score !== undefined) && (
                        <div className="mt-1.5 flex flex-wrap gap-2 text-[10px] font-mono text-slate-400">
                          {ev.attack_type && <span className="text-amber-400 font-semibold">Attack: {ev.attack_type}</span>}
                          {ev.anomaly_score !== undefined && <span>Anomaly Score: {Number(ev.anomaly_score).toFixed(4)}</span>}
                          {ev.predicted_label && <span>ML Label: {ev.predicted_label}</span>}
                          {ev.risk_score !== undefined && <span>Risk: {ev.risk_score} ({ev.risk_level})</span>}
                        </div>
                      )}
                    </div>

                    {/* Inspect button */}
                    <button
                      onClick={() => setInspectEvent(ev)}
                      title="Inspect full JSON"
                      className="p-1.5 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded transition-colors shrink-0"
                    >
                      <Code2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {inspectEvent && (
        <JsonViewerModal
          title={`Step Inspector: ${inspectEvent.event_type} (${inspectEvent.event_id.slice(0, 8)})`}
          data={inspectEvent}
          onClose={() => setInspectEvent(null)}
        />
      )}
    </div>
  )
}
