import React from 'react'
import { Activity, RefreshCw, Cpu, Columns2, MessageSquare, ShieldCheck } from 'lucide-react'
import type { SystemStatusResponse } from '../types'

interface NavbarProps {
  layoutMode: 'split' | 'chat' | 'monitor'
  onSelectLayoutMode: (mode: 'split' | 'chat' | 'monitor') => void
  systemStatus: SystemStatusResponse | null
  onRefresh: () => void
  refreshing: boolean
}

export const Navbar: React.FC<NavbarProps> = ({
  layoutMode,
  onSelectLayoutMode,
  systemStatus,
  onRefresh,
  refreshing,
}) => {
  const isOnline = systemStatus?.status === 'online'
  const isOllamaOk = systemStatus?.ollama_connected

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-[#0B1120] px-4 lg:px-6 py-2.5 transition-colors">
      <div className="flex items-center justify-between w-full">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-teal-600 shadow-md shadow-cyan-500/20 text-white font-bold text-base">
            <Activity className="w-4 h-4 text-white animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-bold tracking-tight text-white">AgentPulse</span>
              <span className="hidden sm:inline-block px-1.5 py-0.2 text-[9px] font-semibold uppercase tracking-wider bg-cyan-950 text-cyan-400 border border-cyan-800/60 rounded">
                Unified Workspace
              </span>
            </div>
            <p className="text-[10px] text-slate-400 hidden md:block">
              Runtime Security & Behavioral Trust Framework for AI Agents
            </p>
          </div>
        </div>

        {/* Workspace Layout Switcher (Desktop controls) */}
        <div className="hidden lg:flex items-center p-0.5 bg-slate-900 border border-slate-800 rounded-lg text-xs">
          <button
            onClick={() => onSelectLayoutMode('split')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
              layoutMode === 'split'
                ? 'bg-cyan-600 text-white shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Columns2 className="w-3.5 h-3.5" />
            <span>Side-by-Side</span>
          </button>
          <button
            onClick={() => onSelectLayoutMode('chat')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
              layoutMode === 'chat'
                ? 'bg-slate-800 text-amber-300 shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Chat Focus</span>
          </button>
          <button
            onClick={() => onSelectLayoutMode('monitor')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
              layoutMode === 'monitor'
                ? 'bg-slate-800 text-cyan-300 shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Monitor Focus</span>
          </button>
        </div>

        {/* Telemetry & Refresh */}
        <div className="flex items-center space-x-2.5">
          {/* Ollama Status */}
          <div className="flex items-center space-x-1.5 px-2 py-1 bg-slate-900 border border-slate-800 rounded-md text-xs">
            <Cpu className="w-3.5 h-3.5 text-slate-400" />
            <span className="hidden sm:inline text-slate-400">Ollama:</span>
            <span
              className={`w-2 h-2 rounded-full ${
                isOllamaOk ? 'bg-emerald-400' : 'bg-rose-500'
              }`}
            />
            <span className={`font-mono text-[11px] ${isOllamaOk ? 'text-emerald-400' : 'text-rose-400'}`}>
              {systemStatus ? (isOllamaOk ? systemStatus.ollama_model : 'Offline') : 'Checking...'}
            </span>
          </div>

          {/* Events Count */}
          <div className="hidden sm:flex items-center space-x-1.5 text-xs text-slate-400">
            <span
              className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-amber-400'}`}
            />
            <span className="font-mono text-[11px]">
              {systemStatus?.log_file_events_count ?? 0} events
            </span>
          </div>

          {/* Refresh Action */}
          <button
            onClick={onRefresh}
            title="Refresh All Telemetry"
            disabled={refreshing}
            className="p-1.5 text-slate-400 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-md transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  )
}
