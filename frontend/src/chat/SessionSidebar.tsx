import React from 'react'
import { Plus, MessageSquare, Trash2, ArrowUpRight, ShieldCheck, Clock } from 'lucide-react'
import type { SessionSummary } from '../types'

interface SessionSidebarProps {
  sessions: SessionSummary[]
  activeSessionId: string | null
  onSelectSession: (id: string) => void
  onNewChat: () => void
  onClearHistory: () => void
  onOpenDashboardSession: (id: string) => void
}

export const SessionSidebar: React.FC<SessionSidebarProps> = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onClearHistory,
  onOpenDashboardSession,
}) => {
  return (
    <aside className="w-64 sm:w-72 bg-[#F3F2EE] border-r border-[#E6E4DD] flex flex-col h-full select-none text-[#2A2926]">
      {/* New Chat Button */}
      <div className="p-3.5 border-b border-[#E6E4DD]">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center space-x-2 px-3.5 py-2.5 bg-[#E8E6DF] hover:bg-[#DEDCD4] text-[#1C1917] rounded-xl font-medium text-xs shadow-xs transition-colors"
        >
          <Plus className="w-4 h-4 text-[#78716C]" />
          <span>New Conversation</span>
        </button>
      </div>

      {/* History List */}
      <div className="flex-1 overflow-y-auto px-2 py-3 space-y-1">
        <div className="px-2 pb-1.5 text-[10px] font-semibold text-[#8C877D] uppercase tracking-wider">
          Recent Agent Sessions
        </div>

        {sessions.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-[#A8A29E]">
            No recent conversations yet.
          </div>
        ) : (
          sessions.map((s) => {
            const isActive = s.session_id === activeSessionId
            const isAlert = s.risk_score >= 30
            return (
              <div
                key={s.session_id}
                onClick={() => onSelectSession(s.session_id)}
                className={`group relative flex flex-col p-2.5 rounded-lg text-left text-xs transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-[#E3E1D9] text-[#1C1917] font-medium'
                    : 'text-[#44403C] hover:bg-[#EBE9E2]'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center space-x-1.5 truncate flex-1">
                    <MessageSquare className="w-3.5 h-3.5 shrink-0 text-[#78716C]" />
                    <span className="truncate">{s.prompt || 'Untitled session'}</span>
                  </div>
                  {isAlert && (
                    <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0 ml-1" title="Security Alert" />
                  )}
                </div>

                <div className="flex items-center justify-between text-[10px] text-[#8C877D] mt-0.5">
                  <span className="flex items-center space-x-1">
                    <Clock className="w-2.5 h-2.5" />
                    <span>{new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  </span>

                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onOpenDashboardSession(s.session_id)
                    }}
                    title="Inspect in Security Dashboard"
                    className="opacity-0 group-hover:opacity-100 text-[#78716C] hover:text-[#06B6D4] flex items-center space-x-0.5 transition-opacity"
                  >
                    <span>SOC</span>
                    <ArrowUpRight className="w-3 h-3" />
                  </button>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-[#E6E4DD] flex items-center justify-between text-xs text-[#78716C]">
        <div className="flex items-center space-x-1.5 text-[11px]">
          <ShieldCheck className="w-3.5 h-3.5 text-teal-600" />
          <span>Digital Twin Monitored</span>
        </div>
        <button
          onClick={onClearHistory}
          title="Clear current chat view"
          className="p-1 hover:text-rose-600 rounded transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
    </aside>
  )
}
