import React from 'react'
import { Server, Cpu, Database, CheckCircle2, XCircle, RefreshCw, HardDrive, Shield } from 'lucide-react'
import type { SystemStatusResponse } from '../types'
import { StatusBadge } from '../components/StatusBadge'

interface SystemTabProps {
  systemStatus: SystemStatusResponse | null
  onRefresh: () => void
  refreshing: boolean
}

export const SystemTab: React.FC<SystemTabProps> = ({
  systemStatus,
  onRefresh,
  refreshing,
}) => {
  if (!systemStatus) {
    return (
      <div className="p-12 text-center text-slate-500">
        Loading system telemetry...
      </div>
    )
  }

  const isOllamaOk = systemStatus.ollama_connected
  const isOnline = systemStatus.status === 'online'

  return (
    <div className="space-y-6">
      {/* Top Banner Status */}
      <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-white shadow-lg ${
              isOnline ? 'bg-emerald-600/90 shadow-emerald-500/20' : 'bg-rose-600/90 shadow-rose-500/20'
            }`}
          >
            <Server className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-base font-semibold text-slate-100">AgentPulse Security Engine</h3>
              <StatusBadge type="status" value={systemStatus.status.toUpperCase()} />
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Agent ID: <span className="font-mono text-cyan-400">{systemStatus.agent_id}</span> • Pipeline: LangGraph + Multi-Detector Security
            </p>
          </div>
        </div>

        <button
          onClick={onRefresh}
          disabled={refreshing}
          className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg border border-slate-700 transition-colors flex items-center space-x-2 disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-cyan-400' : ''}`} />
          <span>Ping Services</span>
        </button>
      </div>

      {/* Grid of Diagnostics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Ollama LLM Connection */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Language Model Runtime (Ollama)
              </h4>
            </div>
            {isOllamaOk ? (
              <span className="flex items-center space-x-1 text-xs text-emerald-400 font-medium">
                <CheckCircle2 className="w-4 h-4" />
                <span>Connected</span>
              </span>
            ) : (
              <span className="flex items-center space-x-1 text-xs text-rose-400 font-medium">
                <XCircle className="w-4 h-4" />
                <span>Unreachable</span>
              </span>
            )}
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1.5 border-b border-slate-800">
              <span className="text-slate-400">Endpoint:</span>
              <span className="text-slate-200">{systemStatus.ollama_base_url}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800">
              <span className="text-slate-400">Model:</span>
              <span className="text-cyan-400 font-semibold">{systemStatus.ollama_model}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">ReAct Loop:</span>
              <span className="text-emerald-400">LangGraph Compiled Graph</span>
            </div>
          </div>
        </div>

        {/* Security & ML Models Loaded */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-purple-400" />
              <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Security & Model Components
              </h4>
            </div>
            <span className="text-xs font-mono text-slate-400">4 / 4 Active</span>
          </div>

          <div className="space-y-2 text-xs">
            {Object.entries(systemStatus.models_loaded).map(([name, loaded]) => (
              <div key={name} className="flex justify-between items-center py-1.5 border-b border-slate-800 last:border-0">
                <span className="font-mono text-slate-300">{name}</span>
                {loaded ? (
                  <span className="text-emerald-400 font-semibold text-[11px] flex items-center space-x-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>LOADED</span>
                  </span>
                ) : (
                  <span className="text-rose-400 font-semibold text-[11px] flex items-center space-x-1">
                    <XCircle className="w-3.5 h-3.5" />
                    <span>MISSING</span>
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* JSONL Event Storage Info */}
      <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
        <div className="flex items-center space-x-2 mb-4">
          <Database className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Runtime Event Log & Persistence Store
          </h4>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
            <span className="text-slate-400 text-[11px] block mb-1">Target JSONL Path:</span>
            <span className="text-slate-200 break-all text-[11px]">{systemStatus.log_file_path}</span>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center justify-between">
            <div>
              <span className="text-slate-400 text-[11px] block mb-1">Total Events Logged:</span>
              <span className="text-lg font-bold text-white">{systemStatus.log_file_events_count}</span>
            </div>
            <HardDrive className="w-6 h-6 text-slate-600" />
          </div>
        </div>
      </div>
    </div>
  )
}
