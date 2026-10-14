import React from 'react'

interface StatusBadgeProps {
  type: 'risk' | 'action' | 'status' | 'event' | 'prediction'
  value: string | number
  className?: string
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ type, value, className = '' }) => {
  const str = String(value).toUpperCase()

  let colorClasses = 'bg-slate-800 text-slate-300 border-slate-700'

  if (type === 'risk') {
    if (str === 'HIGH' || (typeof value === 'number' && value >= 60)) {
      colorClasses = 'bg-rose-500/15 text-rose-400 border-rose-500/30'
    } else if (str === 'MEDIUM' || (typeof value === 'number' && value >= 30)) {
      colorClasses = 'bg-amber-500/15 text-amber-400 border-amber-500/30'
    } else {
      colorClasses = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
    }
  } else if (type === 'action') {
    if (str === 'HUMAN_REVIEW') {
      colorClasses = 'bg-rose-500/20 text-rose-300 border-rose-500/40'
    } else if (str === 'ALERT') {
      colorClasses = 'bg-amber-500/20 text-amber-300 border-amber-500/40'
    } else {
      colorClasses = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
    }
  } else if (type === 'prediction') {
    if (str === 'SUSPICIOUS') {
      colorClasses = 'bg-amber-500/20 text-amber-300 border-amber-500/40'
    } else {
      colorClasses = 'bg-teal-500/15 text-teal-300 border-teal-500/30'
    }
  } else if (type === 'status') {
    if (str === 'SUCCESS' || str === 'CLEAR' || str === 'ALLOW') {
      colorClasses = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
    } else if (str === 'DETECTED' || str === 'ERROR' || str === 'REJECTED') {
      colorClasses = 'bg-rose-500/15 text-rose-400 border-rose-500/30'
    } else if (str === 'INITIATED') {
      colorClasses = 'bg-sky-500/15 text-sky-400 border-sky-500/30'
    }
  } else if (type === 'event') {
    if (str.includes('SECURITY') || str.includes('RISK')) {
      colorClasses = 'bg-purple-500/15 text-purple-300 border-purple-500/30'
    } else if (str.includes('TOOL')) {
      colorClasses = 'bg-amber-500/15 text-amber-300 border-amber-500/30'
    } else if (str.includes('BEHAVIOR') || str.includes('ML')) {
      colorClasses = 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30'
    } else {
      colorClasses = 'bg-slate-700/40 text-slate-300 border-slate-600/40'
    }
  }

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold tracking-wide border ${colorClasses} ${className}`}
    >
      {String(value)}
    </span>
  )
}
