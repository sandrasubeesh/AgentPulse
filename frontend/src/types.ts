export interface SecurityAssessmentSummary {
  risk_score: number
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | string
  response_action: 'ALLOW' | 'ALERT' | 'HUMAN_REVIEW' | string
  rule_based_detections: number
  behavioral_anomaly_detected: boolean
  anomaly_score: number
  ml_prediction: 'NORMAL' | 'SUSPICIOUS' | string
  ml_confidence: number
  signals_used: string[]
  reason: string
  score_breakdown: Record<string, number>
}

export interface ChatResponse {
  response: string
  session_id: string
  timestamp: string
  security: SecurityAssessmentSummary
  event_count: number
  events: RuntimeEvent[]
}

export interface RuntimeEvent {
  event_id: string
  timestamp: string
  agent_id: string
  event_type: string
  tool?: string | null
  input?: any
  status?: string
  detected?: boolean
  attack_type?: string | null
  severity?: string
  confidence?: number
  reason?: string
  risk_score?: number
  risk_level?: string
  response_action?: string
  anomaly_score?: number
  predicted_label?: string
  session_id?: string
  [key: string]: any
}

export interface SessionSummary {
  session_id: string
  timestamp: string
  prompt: string
  response: string
  event_count: number
  risk_score: number
  risk_level: string
  response_action: string
  correlated: boolean
  ml_prediction: string
  anomaly_score: number
  rule_detections: number
}

export interface SessionDetailResponse {
  summary: SessionSummary
  events: RuntimeEvent[]
}

export interface MetricsResponse {
  total_sessions: number
  total_events: number
  rule_detections_count: number
  anomaly_count: number
  suspicious_predictions_count: number
  risk_distribution: Record<string, number>
  action_distribution: Record<string, number>
  average_risk_score: number
  event_type_distribution: Record<string, number>
}

export interface SystemStatusResponse {
  status: 'online' | 'degraded' | 'offline' | string
  ollama_connected: boolean
  ollama_model: string
  ollama_base_url: string
  models_loaded: Record<string, boolean>
  log_file_path: string
  log_file_events_count: number
  agent_id: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  session_id?: string
  security?: SecurityAssessmentSummary
  event_count?: number
}
