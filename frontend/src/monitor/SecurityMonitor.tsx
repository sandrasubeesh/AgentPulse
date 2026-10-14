import React, { useState, useEffect } from 'react'
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  Brain,
  AlertTriangle,
  CheckCircle2,
  Code2,
  Clock,
  Layers,
  ChevronDown,
  ChevronUp,
  RefreshCw,
} from 'lucide-react'
import type { RuntimeEvent, SessionDetailResponse, SessionSummary } from '../types'
import { api } from '../api'
import { StatusBadge } from '../components/StatusBadge'
import { JsonViewerModal } from '../components/JsonViewerModal'

interface SecurityMonitorProps {
  sessionId: string | null
  sessions: SessionSummary[]
  totalEventsCount: number
  onRefresh: () => void
  refreshing: boolean
}

export const SecurityMonitor: React.FC<SecurityMonitorProps> = ({
  sessionId,
  sessions,
  totalEventsCount,
  onRefresh,
  refreshing,
}) => {
  const [detail, setDetail] = useState<SessionDetailResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [inspectEvent, setInspectEvent] = useState<RuntimeEvent | null>(null)
  const [expandDetails, setExpandDetails] = useState(false)
  const [inspectFullSession, setInspectFullSession] = useState(false)

  // Target session ID: either passed in or the newest session from sessions list
  const activeId = sessionId || (sessions.length > 0 ? sessions[0].session_id : null)

  useEffect(() => {
    if (!activeId) {
      setDetail(null)
      return
    }

    setLoading(true)
    api
      .getSessionDetail(activeId)
      .then((data) => {
        setDetail(data)
      })
      .catch((err) => {
        console.warn('Failed to load session detail for monitor:', err)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [activeId])

  if (!activeId || (!detail && !loading)) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-center text-slate-400 bg-[#0B1120]">
        <div className="w-12 h-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-3 text-cyan-400">
          <ShieldCheck className="w-6 h-6" />
        </div>
        <h3 className="text-sm font-semibold text-slate-200">Security Monitor Standing By</h3>
        <p className="text-xs text-slate-500 mt-1 max-w-xs">
          Send a message in the chat on the left to initiate a runtime session and inspect live security evaluations.
        </p>
      </div>
    )
  }

  const summary = detail?.summary
  const events = detail?.events || []

  const riskScore = summary ? summary.risk_score : 0
  const riskLevel = summary ? summary.risk_level : 'LOW'
  const responseAction = summary ? summary.response_action : 'ALLOW'
  const isHighRisk = riskScore >= 60 || riskLevel === 'HIGH'
  const isMedRisk = riskScore >= 30 || riskLevel === 'MEDIUM'

  // Extract threat detections from events (prioritize explicit SECURITY_DETECTION)
  const explicitDetections = events.filter((e) => e.event_type === 'SECURITY_DETECTION')
  const detections =
    explicitDetections.length > 0
      ? explicitDetections
      : events.filter(
          (e) =>
            e.event_type === 'SECURITY_ANALYSIS' &&
            (e.detected === true || e.status === 'detected' || (typeof e.input === 'object' && e.input?.detected === true))
        )

  // Extract ML & Isolation Forest events
  const anomalyEv = events.find((e) => e.event_type === 'BEHAVIORAL_ANOMALY')
  const mlEv = events.find((e) => e.event_type === 'ML_BEHAVIOR_PREDICTION')
  const corrEv = events.find((e) => e.event_type === 'SECURITY_CORRELATION')
  const riskEv = events.find((e) => e.event_type === 'RISK_ASSESSMENT')

  const anomalyDetected =
    anomalyEv?.detected === true || (typeof anomalyEv?.input === 'object' && anomalyEv?.input?.detected === true)
  const anomalyScore = anomalyEv?.anomaly_score ?? anomalyEv?.input?.anomaly_score ?? 0.0

  const mlLabel = mlEv?.predicted_label || mlEv?.input?.predicted_label || summary?.ml_prediction || 'NORMAL'
  const mlConfidence = mlEv?.confidence ?? mlEv?.input?.confidence ?? 0.0

  const isCorrelated = summary?.correlated || corrEv?.correlated || corrEv?.input?.correlated || false
  const scoreBreakdown = riskEv?.score_breakdown || riskEv?.input?.score_breakdown || {}

  return (
    <div className="h-full flex flex-col bg-[#0B1120] text-slate-100 overflow-hidden font-sans border-l border-slate-800/80">
      {/* Top Monitor Header */}
      <div className="px-5 py-3 border-b border-slate-800 bg-[#0F172A]/90 backdrop-blur-xs flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-2.5">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-cyan-950 border border-cyan-800/60 text-cyan-400">
            <Activity className="w-4 h-4 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-100">
                AgentPulse Security Monitor
              </h2>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
            </div>
            <p className="text-[10px] text-slate-400 font-mono">
              Session: <span className="text-cyan-400">{activeId.slice(0, 8)}</span> • {events.length} session events
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-[11px] font-mono text-slate-400 hidden sm:inline">
            Total Logged: <strong className="text-slate-200">{totalEventsCount}</strong>
          </span>
          <button
            onClick={onRefresh}
            disabled={refreshing}
            title="Refresh Security Telemetry"
            className="p-1.5 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded-md border border-slate-800 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Scrollable Monitor Body */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4">
        {/* 1. Primary Risk Score & Response Action Card */}
        <div
          className={`p-4 rounded-xl border transition-all ${
            isHighRisk
              ? 'bg-rose-950/20 border-rose-500/40 shadow-sm shadow-rose-950/40'
              : isMedRisk
              ? 'bg-amber-950/20 border-amber-500/40 shadow-sm shadow-amber-950/40'
              : 'bg-slate-900/90 border-slate-800'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
              <span>Current Session Risk Evaluation</span>
            </span>
            <div className="flex items-center space-x-1.5">
              <StatusBadge type="risk" value={riskLevel} />
              <StatusBadge type="action" value={responseAction} />
            </div>
          </div>

          <div className="flex items-baseline space-x-3 my-2">
            <span className="text-3xl font-bold font-mono text-white">
              {riskScore}
              <span className="text-sm font-normal text-slate-400"> / 100</span>
            </span>
            <span className="text-xs font-medium text-slate-300">
              {riskLevel} Risk &bull; Enforcement:{' '}
              <strong
                className={
                  responseAction === 'HUMAN_REVIEW'
                    ? 'text-rose-400'
                    : responseAction === 'ALERT'
                    ? 'text-amber-400'
                    : 'text-emerald-400'
                }
              >
                {responseAction}
              </strong>
            </span>
          </div>

          {/* Progress bar */}
          <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden mb-2">
            <div
              style={{ width: `${Math.min(100, Math.max(4, riskScore))}%` }}
              className={`h-full transition-all duration-500 ${
                isHighRisk ? 'bg-rose-500' : isMedRisk ? 'bg-amber-500' : 'bg-emerald-500'
              }`}
            />
          </div>

          <p className="text-xs text-slate-300 leading-relaxed pt-1">
            {riskEv?.reason || summary?.prompt ? (
              <span>
                {riskEv?.reason ||
                  (riskScore === 0
                    ? 'Clean session: no known attack patterns, anomalies, or suspicious behaviors flagged.'
                    : 'Security signals correlated to determine composite risk level.')}
              </span>
            ) : (
              'Awaiting session evaluation results...'
            )}
          </p>
        </div>

        {/* 2. Security Threat Detections Card */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span>Deterministic Threat Detections</span>
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                detections.length > 0
                  ? 'bg-rose-950 text-rose-300 border border-rose-800/60'
                  : 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/40'
              }`}
            >
              {detections.length} Flagged
            </span>
          </div>

          {detections.length === 0 ? (
            <div className="flex items-center space-x-2 py-1 text-xs text-emerald-400">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Prompt and tool misuse rule checks passed cleanly.</span>
            </div>
          ) : (
            <div className="space-y-2 pt-1">
              {detections.map((d) => {
                const attackType =
                  d.attack_type ||
                  (typeof d.input === 'object' ? d.input?.attack_type : null) ||
                  'SECURITY_DETECTION'
                const severity =
                  d.severity || (typeof d.input === 'object' ? d.input?.severity : null) || 'HIGH'
                const reason =
                  d.reason || (typeof d.input === 'object' ? d.input?.reason : null) || 'Signature matched'
                return (
                  <div
                    key={d.event_id}
                    className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-500/30 flex items-start justify-between gap-2 text-xs"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono font-bold text-rose-300">{attackType}</span>
                        <StatusBadge type="risk" value={severity} />
                      </div>
                      <p className="text-slate-300 text-[11px]">{reason}</p>
                    </div>
                    <button
                      onClick={() => setInspectEvent(d)}
                      title="Inspect event JSON"
                      className="p-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded shrink-0 transition-colors"
                    >
                      <Code2 className="w-3 h-3" />
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* 3. Behavioral ML & Correlation Telemetry */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Isolation Forest Card */}
          <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-1">
                <Activity className="w-3.5 h-3.5 text-purple-400" />
                <span>Isolation Forest</span>
              </span>
              <span
                className={`px-1.5 py-0.2 rounded text-[10px] font-mono font-bold ${
                  anomalyDetected ? 'bg-purple-950 text-purple-300' : 'bg-slate-800 text-slate-300'
                }`}
              >
                {anomalyDetected ? 'ANOMALY' : 'CLEAR'}
              </span>
            </div>
            <div className="text-xs font-mono">
              Score: <strong className="text-white">{Number(anomalyScore).toFixed(4)}</strong>
            </div>
            <p className="text-[11px] text-slate-400 leading-tight">
              {anomalyDetected
                ? 'Statistical deviation from baseline session patterns.'
                : 'Consistent with learned benign baseline.'}
            </p>
          </div>

          {/* Random Forest Card */}
          <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-1">
                <Brain className="w-3.5 h-3.5 text-teal-400" />
                <span>Random Forest ML</span>
              </span>
              <StatusBadge type="prediction" value={mlLabel} />
            </div>
            <div className="text-xs font-mono">
              Confidence: <strong className="text-white">{(Number(mlConfidence) * 100).toFixed(0)}%</strong>
            </div>
            <p className="text-[11px] text-slate-400 leading-tight">
              Supervised classification of runtime event volume & tool metrics.
            </p>
          </div>
        </div>

        {/* Correlation Status Banner */}
        <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2">
            <span
              className={`w-2 h-2 rounded-full ${
                isCorrelated ? 'bg-rose-400 animate-pulse' : 'bg-slate-500'
              }`}
            />
            <span className="text-slate-300">
              Multi-Signal Correlation:{' '}
              <strong className={isCorrelated ? 'text-rose-400 font-bold' : 'text-slate-400 font-normal'}>
                {isCorrelated ? 'Multi-Detector Agreement Confirmed' : 'Single / Independent Signal'}
              </strong>
            </span>
          </div>
          {isCorrelated && (
            <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-300 text-[10px] font-mono font-bold">
              +10 BONUS
            </span>
          )}
        </div>

        {/* 4. Chronological Step Sequence */}
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 bg-slate-950/60 flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Session Event Timeline ({events.length} steps)</span>
            </span>
            <span className="text-[10px] text-slate-500 font-mono">Ordered Replay</span>
          </div>

          <div className="divide-y divide-slate-800/60 max-h-64 overflow-y-auto">
            {events.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-500">
                No events recorded for this session yet.
              </div>
            ) : (
              events.map((ev, index) => {
                const payloadPreview =
                  typeof ev.input === 'string'
                    ? ev.input
                    : typeof ev.input === 'object'
                    ? JSON.stringify(ev.input)
                    : String(ev.input ?? '')

                return (
                  <div
                    key={ev.event_id}
                    className="p-2.5 hover:bg-slate-850/50 flex items-center justify-between text-xs transition-colors group"
                  >
                    <div className="flex items-center space-x-2 min-w-0 flex-1 pr-2">
                      <span className="text-[10px] font-mono text-slate-500 w-5 shrink-0">
                        #{index + 1}
                      </span>
                      <StatusBadge type="event" value={ev.event_type} />
                      {ev.tool && (
                        <span className="px-1.5 py-0.2 rounded bg-slate-800 text-[10px] font-mono text-cyan-300 shrink-0">
                          {ev.tool}
                        </span>
                      )}
                      <span className="text-slate-400 font-sans truncate text-[11px]">
                        {payloadPreview || '-'}
                      </span>
                    </div>

                    <div className="flex items-center space-x-2 shrink-0">
                      <StatusBadge type="status" value={ev.status || 'success'} />
                      <button
                        onClick={() => setInspectEvent(ev)}
                        title="Inspect step payload"
                        className="p-1 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                      >
                        <Code2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* 5. Simple Expandable Session Details Section */}
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 overflow-hidden">
          <button
            onClick={() => setExpandDetails(!expandDetails)}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-850/50 transition-colors"
          >
            <span className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              <span>Technical Session Breakdown</span>
            </span>
            <div className="flex items-center space-x-1 text-xs text-slate-400">
              <span>{expandDetails ? 'Hide' : 'Expand'}</span>
              {expandDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </div>
          </button>

          {expandDetails && (
            <div className="p-4 border-t border-slate-800 space-y-3 text-xs bg-slate-950/40">
              <div className="space-y-1 font-mono text-[11px]">
                <div className="flex justify-between py-1 border-b border-slate-800/80">
                  <span className="text-slate-400">Session ID:</span>
                  <span className="text-cyan-300 break-all select-all">{activeId}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/80">
                  <span className="text-slate-400">Agent:</span>
                  <span className="text-slate-200">agent_001 (LangGraph ReAct)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/80">
                  <span className="text-slate-400">Rule Detection Points:</span>
                  <span className="text-white">+{scoreBreakdown.rule_based ?? 0} pts</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/80">
                  <span className="text-slate-400">Isolation Forest Points:</span>
                  <span className="text-white">+{scoreBreakdown.isolation_forest ?? 0} pts</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/80">
                  <span className="text-slate-400">Random Forest Points:</span>
                  <span className="text-white">+{scoreBreakdown.random_forest ?? 0} pts</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">Triple Agreement Bonus:</span>
                  <span className="text-white">+{scoreBreakdown.triple_correlation_bonus ?? 0} pts</span>
                </div>
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  onClick={() => setInspectFullSession(true)}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition-colors flex items-center space-x-1.5 text-xs"
                >
                  <Code2 className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Inspect Full Session JSON</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Inspect Single Event Modal */}
      {inspectEvent && (
        <JsonViewerModal
          title={`Event Step: ${inspectEvent.event_type} (${inspectEvent.event_id.slice(0, 8)})`}
          data={inspectEvent}
          onClose={() => setInspectEvent(null)}
        />
      )}

      {/* Inspect Full Session Modal */}
      {inspectFullSession && detail && (
        <JsonViewerModal
          title={`Full Session Payload: ${activeId}`}
          data={detail}
          onClose={() => setInspectFullSession(false)}
        />
      )}
    </div>
  )
}
