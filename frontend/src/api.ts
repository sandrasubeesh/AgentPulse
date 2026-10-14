import type {
  ChatResponse,
  MetricsResponse,
  RuntimeEvent,
  SessionDetailResponse,
  SessionSummary,
  SystemStatusResponse,
} from './types'

const BASE_URL = '' // Uses Vite proxy in development or direct host in production

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!res.ok) {
    let errorMsg = `HTTP ${res.status}: ${res.statusText}`
    try {
      const errData = await res.json()
      if (errData && errData.detail) {
        errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail)
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(errorMsg)
  }

  return (await res.json()) as T
}

export const api = {
  getSystemStatus: () => fetchJson<SystemStatusResponse>('/api/status'),
  getMetrics: () => fetchJson<MetricsResponse>('/api/metrics'),
  getSessions: () => fetchJson<SessionSummary[]>('/api/sessions'),
  getSessionDetail: (sessionId: string) => fetchJson<SessionDetailResponse>(`/api/sessions/${encodeURIComponent(sessionId)}`),
  getEvents: (params?: { limit?: number; event_type?: string; session_id?: string }) => {
    const q = new URLSearchParams()
    if (params?.limit) q.set('limit', String(params.limit))
    if (params?.event_type) q.set('event_type', params.event_type)
    if (params?.session_id) q.set('session_id', params.session_id)
    const queryString = q.toString() ? `?${q.toString()}` : ''
    return fetchJson<RuntimeEvent[]>(`/api/events${queryString}`)
  },
  sendChatMessage: (message: string) =>
    fetchJson<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message }),
    }),
}
