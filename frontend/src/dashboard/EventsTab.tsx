import React, { useState, useMemo } from 'react'
import { Search, Filter, Code2, RefreshCw } from 'lucide-react'
import type { RuntimeEvent } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { JsonViewerModal } from '../components/JsonViewerModal'

interface EventsTabProps {
  events: RuntimeEvent[]
  onRefresh: () => void
  refreshing: boolean
}

export const EventsTab: React.FC<EventsTabProps> = ({ events, onRefresh, refreshing }) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedType, setSelectedType] = useState<string>('ALL')
  const [inspectEvent, setInspectEvent] = useState<RuntimeEvent | null>(null)

  const eventTypes = useMemo(() => {
    const set = new Set<string>()
    events.forEach((e) => {
      if (e.event_type) set.add(e.event_type)
    })
    return ['ALL', ...Array.from(set).sort()]
  }, [events])

  const filteredEvents = useMemo(() => {
    return events.filter((e) => {
      if (selectedType !== 'ALL' && e.event_type !== selectedType) {
        return false
      }
      if (!searchTerm) return true

      const term = searchTerm.toLowerCase()
      const typeMatch = e.event_type?.toLowerCase().includes(term)
      const toolMatch = e.tool?.toLowerCase().includes(term)
      const inputStr = typeof e.input === 'string' ? e.input : JSON.stringify(e.input ?? '')
      const inputMatch = inputStr.toLowerCase().includes(term)
      const reasonMatch = e.reason?.toLowerCase().includes(term)
      const statusMatch = e.status?.toLowerCase().includes(term)

      return typeMatch || toolMatch || inputMatch || reasonMatch || statusMatch
    })
  }, [events, selectedType, searchTerm])

  const formatInputPreview = (input: any): string => {
    if (input === null || input === undefined) return '-'
    if (typeof input === 'string') return input
    if (typeof input === 'object') {
      if (input.message) return String(input.message)
      if (input.query) return `query: ${input.query}`
      if (input.filename) return `file: ${input.filename}`
      if (input.reason) return String(input.reason)
      if (input.risk_score !== undefined) return `Risk: ${input.risk_score} (${input.risk_level})`
      return JSON.stringify(input)
    }
    return String(input)
  }

  return (
    <div className="space-y-4">
      {/* Controls Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 p-4 rounded-xl bg-slate-900/80 border border-slate-800">
        <div className="flex items-center space-x-3 flex-1">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search runtime events by tool, query, status, or payload..."
              className="w-full pl-9 pr-4 py-1.5 bg-slate-950 border border-slate-700/80 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
            />
          </div>

          {/* Type Filter */}
          <div className="flex items-center space-x-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="px-2.5 py-1.5 bg-slate-950 border border-slate-700/80 rounded-lg text-xs text-slate-200 focus:outline-none focus:border-cyan-500 transition-colors"
            >
              {eventTypes.map((type) => (
                <option key={type} value={type}>
                  {type === 'ALL' ? 'All Event Types' : type}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center space-x-3 justify-end text-xs text-slate-400">
          <span>
            Showing <strong className="text-white font-mono">{filteredEvents.length}</strong> of{' '}
            <span className="font-mono">{events.length}</span> events
          </span>
          <button
            onClick={onRefresh}
            disabled={refreshing}
            className="flex items-center space-x-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-200 font-medium transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-cyan-400' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Events Table */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800 overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-950/70 text-slate-400 uppercase tracking-wider text-[11px]">
                <th className="py-3 px-4 font-semibold">Timestamp</th>
                <th className="py-3 px-4 font-semibold">Event Type</th>
                <th className="py-3 px-4 font-semibold">Tool</th>
                <th className="py-3 px-4 font-semibold">Payload / Preview</th>
                <th className="py-3 px-4 font-semibold">Status</th>
                <th className="py-3 px-4 font-semibold text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredEvents.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-12 text-center text-slate-500 font-sans">
                    No runtime events matching the filter criteria.
                  </td>
                </tr>
              ) : (
                filteredEvents.map((ev) => (
                  <tr
                    key={ev.event_id}
                    className="hover:bg-slate-800/40 transition-colors group cursor-pointer"
                    onClick={() => setInspectEvent(ev)}
                  >
                    {/* Timestamp */}
                    <td className="py-3 px-4 text-slate-400 whitespace-nowrap text-[11px]">
                      {new Date(ev.timestamp).toLocaleTimeString()}
                    </td>

                    {/* Event Type */}
                    <td className="py-3 px-4 whitespace-nowrap">
                      <StatusBadge type="event" value={ev.event_type} />
                    </td>

                    {/* Tool */}
                    <td className="py-3 px-4 whitespace-nowrap">
                      {ev.tool ? (
                        <span className="px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700/60 text-[11px]">
                          {ev.tool}
                        </span>
                      ) : (
                        <span className="text-slate-600">-</span>
                      )}
                    </td>

                    {/* Payload Preview */}
                    <td className="py-3 px-4 text-slate-300 font-sans max-w-md truncate">
                      {formatInputPreview(ev.input)}
                    </td>

                    {/* Status */}
                    <td className="py-3 px-4 whitespace-nowrap">
                      <StatusBadge type="status" value={ev.status || 'success'} />
                    </td>

                    {/* Inspect Action */}
                    <td className="py-3 px-4 text-right whitespace-nowrap">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setInspectEvent(ev)
                        }}
                        className="px-2 py-1 bg-slate-800 hover:bg-cyan-950 hover:text-cyan-300 border border-slate-700 hover:border-cyan-800 rounded text-slate-300 transition-colors inline-flex items-center space-x-1"
                      >
                        <Code2 className="w-3.5 h-3.5" />
                        <span className="text-[10px]">JSON</span>
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* JSON Modal */}
      {inspectEvent && (
        <JsonViewerModal
          title={`Event Inspector: ${inspectEvent.event_type} (${inspectEvent.event_id.slice(0, 8)})`}
          data={inspectEvent}
          onClose={() => setInspectEvent(null)}
        />
      )}
    </div>
  )
}
