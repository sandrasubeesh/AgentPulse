import React, { useState } from 'react'
import {
  LayoutDashboard,
  ListFilter,
  ShieldAlert,
  Activity,
  ShieldCheck,
  Compass,
  Server,
} from 'lucide-react'
import type { MetricsResponse, SessionSummary, RuntimeEvent, SystemStatusResponse } from '../types'
import { OverviewTab } from './OverviewTab'
import { EventsTab } from './EventsTab'
import { ThreatsTab } from './ThreatsTab'
import { BehaviorTab } from './BehaviorTab'
import { RiskTab } from './RiskTab'
import { ExplorerTab } from './ExplorerTab'
import { SystemTab } from './SystemTab'

interface DashboardLayoutProps {
  metrics: MetricsResponse | null
  sessions: SessionSummary[]
  events: RuntimeEvent[]
  systemStatus: SystemStatusResponse | null
  onRefresh: () => void
  refreshing: boolean
  initialSessionId?: string | null
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  metrics,
  sessions,
  events,
  systemStatus,
  onRefresh,
  refreshing,
  initialSessionId = null,
}) => {
  const [activeTab, setActiveTab] = useState<string>(initialSessionId ? 'explorer' : 'overview')
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(initialSessionId)

  const handleSelectSession = (id: string) => {
    setSelectedSessionId(id)
    setActiveTab('explorer')
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'events', label: 'Runtime Events', icon: ListFilter, count: events.length },
    { id: 'threats', label: 'Threat Detection', icon: ShieldAlert, alert: (metrics?.rule_detections_count ?? 0) > 0 },
    { id: 'behavior', label: 'Behavioral Analysis', icon: Activity },
    { id: 'risk', label: 'Risk Assessment', icon: ShieldCheck },
    { id: 'explorer', label: 'Session Explorer', icon: Compass, count: sessions.length },
    { id: 'system', label: 'System Status', icon: Server },
  ]

  return (
    <div className="min-h-[calc(100vh-60px)] bg-[#0B1120] text-slate-100 flex flex-col">
      {/* Subnav Tab Bar */}
      <div className="border-b border-slate-800 bg-[#0F172A]/80 sticky top-[57px] z-30 px-4 lg:px-8">
        <div className="max-w-7xl mx-auto flex items-center space-x-1 overflow-x-auto py-2 no-scrollbar">
          {tabs.map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                  isActive
                    ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
                {tab.count !== undefined && (
                  <span
                    className={`ml-1 px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                      isActive ? 'bg-cyan-900/60 text-cyan-300' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {tab.count}
                  </span>
                )}
                {tab.alert && (
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Main Tab Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-8">
        {activeTab === 'overview' && (
          <OverviewTab
            metrics={metrics}
            sessions={sessions}
            onSelectSession={handleSelectSession}
            onNavigateTab={setActiveTab}
          />
        )}
        {activeTab === 'events' && (
          <EventsTab events={events} onRefresh={onRefresh} refreshing={refreshing} />
        )}
        {activeTab === 'threats' && <ThreatsTab events={events} />}
        {activeTab === 'behavior' && <BehaviorTab events={events} />}
        {activeTab === 'risk' && <RiskTab sessions={sessions} />}
        {activeTab === 'explorer' && (
          <ExplorerTab
            sessions={sessions}
            selectedSessionId={selectedSessionId}
            onSelectSessionId={setSelectedSessionId}
          />
        )}
        {activeTab === 'system' && (
          <SystemTab
            systemStatus={systemStatus}
            onRefresh={onRefresh}
            refreshing={refreshing}
          />
        )}
      </main>
    </div>
  )
}
