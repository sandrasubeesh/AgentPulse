import { useState, useEffect, useCallback } from 'react'
import { Navbar } from './components/Navbar'
import { ChatLayout } from './chat/ChatLayout'
import { SecurityMonitor } from './monitor/SecurityMonitor'
import { api } from './api'
import type {
  SessionSummary,
  RuntimeEvent,
  SystemStatusResponse,
} from './types'

export function App() {
  const [layoutMode, setLayoutMode] = useState<'split' | 'chat' | 'monitor'>('split')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [events, setEvents] = useState<RuntimeEvent[]>([])
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  // Load all telemetry from actual backend API endpoints
  const loadData = useCallback(async () => {
    setRefreshing(true)
    try {
      const [statusRes, sessionsRes, eventsRes] = await Promise.all([
        api.getSystemStatus().catch((err) => {
          console.warn('Status fetch error:', err)
          return null
        }),
        api.getSessions().catch((err) => {
          console.warn('Sessions fetch error:', err)
          return []
        }),
        api.getEvents({ limit: 200 }).catch((err) => {
          console.warn('Events fetch error:', err)
          return []
        }),
      ])

      if (statusRes) setSystemStatus(statusRes)
      if (sessionsRes) {
        setSessions(sessionsRes)
        // If no active session yet and we have sessions, select the latest
        setActiveSessionId((prev) => prev || (sessionsRes.length > 0 ? sessionsRes[0].session_id : null))
      }
      if (eventsRes) setEvents(eventsRes)
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    loadData()
    // Poll telemetry every 10 seconds for real-time live updates
    const interval = setInterval(loadData, 10000)
    return () => clearInterval(interval)
  }, [loadData])

  const handleSessionSelect = (sessionId: string) => {
    setActiveSessionId(sessionId)
  }

  return (
    <div className="h-screen w-screen flex flex-col font-sans antialiased bg-[#0B1120] overflow-hidden">
      {/* Top Header */}
      <Navbar
        layoutMode={layoutMode}
        onSelectLayoutMode={setLayoutMode}
        systemStatus={systemStatus}
        onRefresh={loadData}
        refreshing={refreshing}
      />

      {/* Main Side-by-Side Workspace */}
      <div className="flex-1 flex flex-col lg:flex-row w-full overflow-hidden">
        {/* Left Panel: AI Agent Chat */}
        {(layoutMode === 'split' || layoutMode === 'chat') && (
          <div
            className={`flex flex-col h-[50vh] lg:h-full overflow-hidden border-b lg:border-b-0 lg:border-r border-[#E6E4DD]/60 ${
              layoutMode === 'chat' ? 'w-full' : 'w-full lg:w-[53%] shrink-0'
            }`}
          >
            <ChatLayout
              sessions={sessions}
              systemStatus={systemStatus}
              onOpenDashboardSession={handleSessionSelect}
              onRefreshData={loadData}
              onSessionSelect={handleSessionSelect}
              activeSessionIdProp={activeSessionId}
            />
          </div>
        )}

        {/* Right Panel: Simplified Security Monitor */}
        {(layoutMode === 'split' || layoutMode === 'monitor') && (
          <div
            className={`flex flex-col h-[50vh] lg:h-full overflow-hidden bg-[#0B1120] ${
              layoutMode === 'monitor' ? 'w-full' : 'w-full lg:w-[47%] flex-1'
            }`}
          >
            <SecurityMonitor
              sessionId={activeSessionId}
              sessions={sessions}
              totalEventsCount={systemStatus?.log_file_events_count || events.length}
              onRefresh={loadData}
              refreshing={refreshing}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default App
