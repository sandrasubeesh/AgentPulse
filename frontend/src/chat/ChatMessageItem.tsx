import React, { useState } from 'react'
import {
  ShieldCheck,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  User,
  Bot,
} from 'lucide-react'
import type { ChatMessage } from '../types'

interface ChatMessageItemProps {
  message: ChatMessage
  onOpenDashboardSession?: (sessionId: string) => void
}

export const ChatMessageItem: React.FC<ChatMessageItemProps> = ({
  message,
  onOpenDashboardSession,
}) => {
  const isUser = message.role === 'user'
  const [expandedSecurity, setExpandedSecurity] = useState(false)

  const security = message.security
  const isSuspicious = security && (security.risk_score >= 30 || security.rule_based_detections > 0)

  return (
    <div className={`py-4 transition-colors ${isUser ? 'bg-transparent' : 'bg-transparent'}`}>
      <div className="max-w-3xl mx-auto px-4 flex space-x-3.5">
        {/* Avatar */}
        <div className="shrink-0 mt-0.5">
          {isUser ? (
            <div className="w-7 h-7 rounded-full bg-[#E5E3DC] text-[#44403C] flex items-center justify-center font-medium text-xs">
              <User className="w-4 h-4 text-[#57534E]" />
            </div>
          ) : (
            <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-[#D97706] to-[#F59E0B] text-white flex items-center justify-center shadow-xs">
              <Bot className="w-4 h-4 text-white" />
            </div>
          )}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0 space-y-2">
          {/* Header */}
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold text-[#1C1917]">
              {isUser ? 'You' : 'AgentPulse Assistant'}
            </span>
            <span className="text-[10px] text-[#A8A29E]">
              {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>

          {/* Text */}
          <div className="text-sm text-[#292524] leading-relaxed whitespace-pre-wrap font-sans selection:bg-[#FDE68A]">
            {message.content}
          </div>

          {/* Security Telemetry Badge (for Assistant responses) */}
          {!isUser && security && (
            <div className="mt-3 pt-2">
              <div
                className={`rounded-xl border p-3 transition-all ${
                  isSuspicious
                    ? 'bg-[#FEF3C7]/40 border-[#FCD34D] text-[#92400E]'
                    : 'bg-[#F5F5F0] border-[#E6E4DD] text-[#57534E]'
                }`}
              >
                {/* Header row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {isSuspicious ? (
                      <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                    ) : (
                      <ShieldCheck className="w-4 h-4 text-teal-600 shrink-0" />
                    )}
                    <span className="text-xs font-semibold">
                      Security Trust: {security.risk_score}/100 ({security.risk_level})
                    </span>
                    <span
                      className={`px-1.5 py-0.2 rounded text-[10px] font-mono uppercase font-bold tracking-wider ${
                        security.response_action === 'ALLOW'
                          ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                          : security.response_action === 'ALERT'
                          ? 'bg-amber-100 text-amber-800 border border-amber-300'
                          : 'bg-rose-100 text-rose-800 border border-rose-300'
                      }`}
                    >
                      {security.response_action}
                    </span>
                  </div>

                  <button
                    onClick={() => setExpandedSecurity(!expandedSecurity)}
                    className="text-xs text-[#78716C] hover:text-[#1C1917] flex items-center space-x-1"
                  >
                    <span>{expandedSecurity ? 'Hide Details' : 'View Signals'}</span>
                    {expandedSecurity ? (
                      <ChevronUp className="w-3.5 h-3.5" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>

                {/* Expanded Details */}
                {expandedSecurity && (
                  <div className="mt-2.5 pt-2 border-t border-[#E6E4DD]/80 space-y-2 text-xs">
                    <p className="text-[11px] text-[#44403C] italic">{security.reason}</p>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-[11px]">
                      <div className="p-1.5 rounded bg-white/70 border border-[#E6E4DD]">
                        <span className="text-[10px] text-[#78716C] block uppercase font-sans">Rules</span>
                        <span className="font-bold">{security.rule_based_detections} detected</span>
                      </div>
                      <div className="p-1.5 rounded bg-white/70 border border-[#E6E4DD]">
                        <span className="text-[10px] text-[#78716C] block uppercase font-sans">Isolation Forest</span>
                        <span className="font-bold">{security.behavioral_anomaly_detected ? 'Anomaly' : 'Clean'}</span>
                      </div>
                      <div className="p-1.5 rounded bg-white/70 border border-[#E6E4DD]">
                        <span className="text-[10px] text-[#78716C] block uppercase font-sans">Random Forest</span>
                        <span className="font-bold">{security.ml_prediction}</span>
                      </div>
                      <div className="p-1.5 rounded bg-white/70 border border-[#E6E4DD]">
                        <span className="text-[10px] text-[#78716C] block uppercase font-sans">ML Conf</span>
                        <span className="font-bold">{(security.ml_confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>

                    {message.session_id && onOpenDashboardSession && (
                      <div className="pt-2 flex justify-end">
                        <button
                          onClick={() => onOpenDashboardSession(message.session_id!)}
                          className="flex items-center space-x-1.5 px-3 py-1 bg-[#1C1917] hover:bg-[#292524] text-white rounded-lg text-xs font-medium shadow-xs transition-colors"
                        >
                          <span>Inspect in Security Dashboard</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
