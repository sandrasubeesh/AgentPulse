import React, { useState } from 'react'
import { Brain, Activity, Sparkles, Code2 } from 'lucide-react'
import type { RuntimeEvent } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { JsonViewerModal } from '../components/JsonViewerModal'

interface BehaviorTabProps {
  events: RuntimeEvent[]
}

export const BehaviorTab: React.FC<BehaviorTabProps> = ({ events }) => {
  const [selectedEvent, setSelectedEvent] = useState<RuntimeEvent | null>(null)

  // Behavioral Anomaly events (Isolation Forest)
  const anomalyEvents = events.filter((e) => e.event_type === 'BEHAVIORAL_ANOMALY')

  // ML Prediction events (Random Forest)
  const mlEvents = events.filter((e) => e.event_type === 'ML_BEHAVIOR_PREDICTION')

  const anomalyCount = anomalyEvents.filter((e) => e.detected === true).length
  const suspiciousMlCount = mlEvents.filter(
    (e) => e.predicted_label === 'SUSPICIOUS' || e.prediction === 1
  ).length

  return (
    <div className="space-y-6">
      {/* Dual Engine Info Header */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Isolation Forest Card */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 relative overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <div className="p-2 rounded-lg bg-purple-950/60 border border-purple-800/60 text-purple-400">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-slate-100">Isolation Forest Engine</h4>
                <p className="text-[11px] text-slate-400">Unsupervised Behavioral Anomaly Detection</p>
              </div>
            </div>
            <StatusBadge type="status" value="ACTIVE" />
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            Evaluates operational event distributions against the pre-trained benign baseline without relying on hardcoded attack signatures. Flags sessions exhibiting structural or temporal deviations.
          </p>

          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">Anomalies Detected:</span>
            <span className="font-mono font-bold text-purple-400">{anomalyCount}</span>
          </div>
        </div>

        {/* Random Forest Card */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 relative overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <div className="p-2 rounded-lg bg-cyan-950/60 border border-cyan-800/60 text-cyan-400">
                <Brain className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-slate-100">Random Forest Classifier</h4>
                <p className="text-[11px] text-slate-400">Supervised Behavioral Trust Classification</p>
              </div>
            </div>
            <StatusBadge type="status" value="ACTIVE" />
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            Classifies sessions as <code className="text-teal-300">NORMAL</code> or <code className="text-amber-300">SUSPICIOUS</code> based on engineered runtime features including event volume, tool error rates, and duration.
          </p>

          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">Suspicious Classifications:</span>
            <span className="font-mono font-bold text-amber-400">{suspiciousMlCount}</span>
          </div>
        </div>
      </div>

      {/* Feature Engineering Reference Card */}
      <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
        <div className="flex items-center space-x-2 mb-3">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Engineered Behavioral Features Extracted at Runtime
          </h4>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 text-xs">
          <div className="p-2.5 rounded bg-slate-800/60 border border-slate-700/50">
            <div className="text-[10px] text-slate-400 uppercase">Feature 1</div>
            <div className="font-mono font-medium text-slate-200 mt-0.5">total_events</div>
          </div>
          <div className="p-2.5 rounded bg-slate-800/60 border border-slate-700/50">
            <div className="text-[10px] text-slate-400 uppercase">Feature 2</div>
            <div className="font-mono font-medium text-slate-200 mt-0.5">tool_call_count</div>
          </div>
          <div className="p-2.5 rounded bg-slate-800/60 border border-slate-700/50">
            <div className="text-[10px] text-slate-400 uppercase">Feature 3</div>
            <div className="font-mono font-medium text-slate-200 mt-0.5">tool_error_count</div>
          </div>
          <div className="p-2.5 rounded bg-slate-800/60 border border-slate-700/50">
            <div className="text-[10px] text-slate-400 uppercase">Feature 4</div>
            <div className="font-mono font-medium text-slate-200 mt-0.5">unique_tools</div>
          </div>
          <div className="p-2.5 rounded bg-slate-800/60 border border-slate-700/50">
            <div className="text-[10px] text-slate-400 uppercase">Feature 5</div>
            <div className="font-mono font-medium text-slate-200 mt-0.5">search_calls</div>
          </div>
          <div className="p-2.5 rounded bg-slate-800/60 border border-slate-700/50">
            <div className="text-[10px] text-slate-400 uppercase">Feature 6</div>
            <div className="font-mono font-medium text-slate-200 mt-0.5">session_duration</div>
          </div>
        </div>
      </div>

      {/* Behavioral Logs Table */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800 overflow-hidden shadow-sm">
        <div className="p-4 border-b border-slate-800 bg-slate-950/60 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
            <Activity className="w-4 h-4 text-purple-400" />
            <span>Behavioral Model Outputs</span>
          </h3>
          <span className="text-xs text-slate-400 font-mono">
            {mlEvents.length} evaluations recorded
          </span>
        </div>

        <div className="divide-y divide-slate-800/70 font-mono text-xs">
          {mlEvents.length === 0 ? (
            <div className="p-12 text-center text-slate-500 font-sans">
              No behavioral evaluation events recorded yet.
            </div>
          ) : (
            mlEvents.map((ml) => {
              const label = ml.predicted_label || ml.input?.predicted_label || 'NORMAL'
              const conf = ml.confidence ?? ml.input?.confidence ?? 0
              const features = ml.feature_summary || ml.input?.feature_summary || {}

              return (
                <div
                  key={ml.event_id}
                  className="p-4 hover:bg-slate-800/30 transition-colors flex flex-col md:flex-row md:items-center justify-between gap-3"
                >
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <StatusBadge type="prediction" value={label} />
                      <span className="text-slate-400 text-xs">
                        Confidence: <strong className="text-white">{(Number(conf) * 100).toFixed(0)}%</strong>
                      </span>
                      <span className="text-[11px] text-slate-500 font-sans">
                        {new Date(ml.timestamp).toLocaleTimeString()}
                      </span>
                    </div>

                    <div className="text-[11px] text-slate-400 font-sans flex flex-wrap gap-x-4 gap-y-1 pt-1">
                      <span>Events: <strong className="text-slate-300 font-mono">{features.total_events ?? '-'}</strong></span>
                      <span>Tools: <strong className="text-slate-300 font-mono">{features.tool_call_count ?? '-'}</strong></span>
                      <span>Errors: <strong className="text-slate-300 font-mono">{features.tool_error_count ?? 0}</strong></span>
                      <span>Duration: <strong className="text-slate-300 font-mono">{Number(features.session_duration_seconds ?? 0).toFixed(2)}s</strong></span>
                    </div>
                  </div>

                  <button
                    onClick={() => setSelectedEvent(ml)}
                    className="self-start md:self-auto px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded border border-slate-700 text-xs font-sans flex items-center space-x-1"
                  >
                    <Code2 className="w-3.5 h-3.5" />
                    <span>Raw Features</span>
                  </button>
                </div>
              )
            })
          )}
        </div>
      </div>

      {selectedEvent && (
        <JsonViewerModal
          title={`Behavioral ML Event: ${selectedEvent.event_id.slice(0, 8)}`}
          data={selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}
    </div>
  )
}
